"""
FastCache Wrapper for Wan Video DiT

Modular wrapper that implements FastCache without modifying DiTBlock internals.
Wraps around the block's forward pass and manages caching decisions.

Key principles:
- Non-invasive: Works as a wrapper, doesn't modify DiTBlock
- Modular: Can be enabled/disabled independently  
- Compatible: Works alongside PAB, TeaCache, AdaCache, TokenCache, etc.

FastCache Strategy:
1. Statistical + Adaptive thresholds determine if we use cache
2. If cache hit: Use linear projection instead of full transformer
3. If cache miss: Compute motion saliency
   - Motion tokens: Full transformer computation
   - Static tokens: Linear projection (lightweight)
4. Update cache and background state with EMA

References:
- FastCache paper: Hidden-state-level caching for DiT
- xDiT: xfuser/model_executor/accelerator/fastcache.py
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple
from .fastcache_mgr import (
    FastCacheConfig,
    should_use_cache,
    compute_motion_saliency,
    update_background_state,
    print_fastcache_stats,
)


class FastCacheBlockWrapper:
    """
    Wrapper for a single DiTBlock to add FastCache functionality.
    
    Manages:
    - Cache hit/miss decision based on statistical + adaptive thresholds
    - Motion saliency computation for token-level caching
    - Linear projection for static content
    - Background state with EMA
    - Per-block statistics
    """
    
    def __init__(
        self,
        block_id: int,
        hidden_size: int,
        config: FastCacheConfig,
    ):
        """
        Initialize FastCache wrapper for a block.
        
        Args:
            block_id: Block index
            hidden_size: Hidden dimension size
            config: FastCache configuration
        """
        self.block_id = block_id
        self.hidden_size = hidden_size
        self.config = config
        
        # Cache states
        self.prev_hidden_states = None
        self.bg_hidden_states = None
        
        # Statistics
        self.cache_hits = 0
        self.total_steps = 0
        self.motion_token_ratio = []  # Track motion/static ratio
        
        # Linear projections (learnable) - will be created on first use with correct dtype
        self.cache_projection = None
        self.static_token_projection = None
        self._projections_initialized = False
    
    def _ensure_projections(self, reference_tensor: torch.Tensor):
        """
        Ensure linear projections are initialized with correct dtype and device.
        
        Args:
            reference_tensor: A tensor to infer dtype and device from
        """
        if self._projections_initialized:
            return
        
        device = reference_tensor.device
        dtype = reference_tensor.dtype
        
        # Create projections with correct dtype and device
        self.cache_projection = nn.Linear(
            self.hidden_size, self.hidden_size, bias=True,
            device=device, dtype=dtype
        )
        self.static_token_projection = nn.Linear(
            self.hidden_size, self.hidden_size, bias=True,
            device=device, dtype=dtype
        )
        
        # Initialize projections with identity-like mapping
        with torch.no_grad():
            nn.init.eye_(self.cache_projection.weight)
            nn.init.zeros_(self.cache_projection.bias)
            nn.init.eye_(self.static_token_projection.weight)
            nn.init.zeros_(self.static_token_projection.bias)
        
        self._projections_initialized = True
    
    def should_compute(
        self,
        hidden_states: torch.Tensor,
        timestep: int,
        max_timesteps: int = 1000
    ) -> Tuple[bool, Dict[str, float]]:
        """
        Decide if we should compute or use cache.
        
        Returns:
            (should_compute, debug_info) tuple
        """
        if not self.config.enabled:
            return True, {}
        
        if self.prev_hidden_states is None:
            # First step: always compute
            return True, {}
        
        # Use statistical + adaptive thresholds
        use_cache, debug_info = should_use_cache(
            hidden_states,
            self.prev_hidden_states,
            timestep,
            self.config,
            max_timesteps
        )
        
        return not use_cache, debug_info
    
    def forward_with_cache(
        self,
        block: nn.Module,
        hidden_states: torch.Tensor,
        context: torch.Tensor,
        timestep: int,
        max_timesteps: int = 1000,
        **kwargs
    ) -> torch.Tensor:
        """
        Forward pass with FastCache.
        
        Strategy:
        1. Check if we should use cache (statistical + adaptive threshold)
        2. If cache hit: Apply cache_projection
        3. If cache miss:
           - Compute motion saliency
           - Motion tokens: Full transformer
           - Static tokens: static_token_projection
        4. Update cache states
        
        Args:
            block: The DiTBlock to wrap
            hidden_states: Input hidden states
            context: Context tensor
            timestep: Current timestep
            max_timesteps: Maximum timesteps for normalization
            **kwargs: Additional block arguments
        
        Returns:
            Output hidden states
        """
        # Ensure projections are initialized with correct dtype/device
        self._ensure_projections(hidden_states)
        
        self.total_steps += 1
        
        # Decide if we should compute or use cache
        should_compute, debug_info = self.should_compute(
            hidden_states, timestep, max_timesteps
        )
        
        if not should_compute:
            # ===== Cache Hit: Use linear projection =====
            self.cache_hits += 1
            output = self.cache_projection(hidden_states)
            
            if self.config.verbose and self.block_id == 0:
                print(f"  [FastCache Block {self.block_id}] Cache HIT "
                      f"(delta={debug_info.get('delta', 0):.4f}, "
                      f"thresh={debug_info.get('final_threshold', 0):.4f})")
            
            return output
        
        # ===== Cache Miss: Compute with motion-aware processing =====
        
        # Compute motion saliency
        motion_saliency, motion_mask, static_mask = compute_motion_saliency(
            hidden_states,
            self.prev_hidden_states,
            self.config.motion_threshold
        )
        
        # Track motion ratio for statistics
        motion_ratio = motion_mask.float().mean().item()
        self.motion_token_ratio.append(motion_ratio)
        
        if self.config.verbose and self.block_id == 0:
            print(f"  [FastCache Block {self.block_id}] Cache MISS "
                  f"(delta={debug_info.get('delta', 0):.4f}, "
                  f"motion_ratio={motion_ratio:.2%})")
        
        # Strategy based on motion ratio
        if motion_ratio > 0.5:
            # High motion: Process all tokens through full transformer
            output = block(hidden_states, context, **kwargs)
        else:
            # Mixed motion: Process motion and static tokens separately
            batch_size, seq_len, hidden_dim = hidden_states.shape
            
            # Clone for output reconstruction
            output = hidden_states.clone()
            
            # Process motion tokens through full transformer (if any)
            if motion_mask.any():
                # Extract motion tokens
                motion_indices = motion_mask.nonzero(as_tuple=False)
                
                # For simplicity, if motion tokens exist, do full computation
                # (Fine-grained per-token processing would require modifying block internals)
                output = block(hidden_states, context, **kwargs)
                
                # Apply static projection only to static tokens
                if static_mask.any():
                    static_output = self.static_token_projection(hidden_states)
                    # Blend: keep motion tokens from full computation, 
                    # replace static tokens with projection
                    output = torch.where(
                        static_mask.unsqueeze(-1).expand_as(output),
                        static_output,
                        output
                    )
            else:
                # All tokens are static: use static projection
                output = self.static_token_projection(hidden_states)
        
        # Update cache states
        self.prev_hidden_states = hidden_states.detach().clone()
        self.bg_hidden_states = update_background_state(
            hidden_states,
            self.bg_hidden_states,
            self.config.ema_alpha
        )
        
        return output
    
    def reset(self):
        """Reset cache states"""
        self.prev_hidden_states = None
        self.bg_hidden_states = None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics for this block"""
        if self.total_steps == 0:
            return {
                'cache_hit_ratio': 0.0,
                'avg_motion_ratio': 0.0,
                'total_steps': 0
            }
        
        return {
            'cache_hit_ratio': self.cache_hits / self.total_steps,
            'avg_motion_ratio': sum(self.motion_token_ratio) / len(self.motion_token_ratio) if self.motion_token_ratio else 0.0,
            'total_steps': self.total_steps
        }


class FastCacheState:
    """
    Global FastCache state manager.
    
    Manages all block wrappers and provides global statistics.
    """
    
    def __init__(self, config: FastCacheConfig, num_blocks: int):
        """
        Initialize global FastCache state.
        
        Args:
            config: FastCache configuration
            num_blocks: Total number of transformer blocks
        """
        self.config = config
        self.num_blocks = num_blocks
        self.wrappers: Dict[int, FastCacheBlockWrapper] = {}
        self.enabled = config.enabled
    
    def get_or_create_wrapper(
        self,
        block_id: int,
        hidden_size: int
    ) -> FastCacheBlockWrapper:
        """Get or create wrapper for a block"""
        if block_id not in self.wrappers:
            self.wrappers[block_id] = FastCacheBlockWrapper(
                block_id, hidden_size, self.config
            )
        return self.wrappers[block_id]
    
    def should_cache_block(self, block_id: int) -> bool:
        """Check if a block should use FastCache"""
        if not self.enabled:
            return False
        
        if self.config.cache_blocks is None:
            # Default: cache middle blocks
            start = self.num_blocks // 3
            end = 2 * self.num_blocks // 3
            return start <= block_id < end
        
        return block_id in self.config.cache_blocks
    
    def reset(self):
        """Reset all wrappers"""
        for wrapper in self.wrappers.values():
            wrapper.reset()
    
    def print_statistics(self):
        """Print global statistics"""
        if not self.wrappers:
            print("⚠️  FastCache: No blocks cached")
            return
        
        total_hits = sum(w.cache_hits for w in self.wrappers.values())
        total_steps = sum(w.total_steps for w in self.wrappers.values())
        
        layer_stats = {
            block_id: wrapper.cache_hits
            for block_id, wrapper in self.wrappers.items()
        }
        
        print_fastcache_stats(total_hits, total_steps, layer_stats)
        
        # Print detailed per-block statistics
        print("\n📈 Detailed Block Statistics:")
        print(f"{'Block':<10} {'Cache Hit':>12} {'Motion Ratio':>14} {'Steps':>8}")
        print("-" * 50)
        for block_id in sorted(self.wrappers.keys()):
            stats = self.wrappers[block_id].get_stats()
            print(f"  {block_id:<8} {stats['cache_hit_ratio']:>11.2%} "
                  f"{stats['avg_motion_ratio']:>13.2%} {stats['total_steps']:>8}")
        print()


# ============================================================================
# Global FastCache State Management
# ============================================================================

_fastcache_state: Optional[FastCacheState] = None


def initialize_fastcache(config: FastCacheConfig, num_blocks: int):
    """Initialize global FastCache state"""
    global _fastcache_state
    _fastcache_state = FastCacheState(config, num_blocks)


def get_fastcache_wrapper(block_id: int, hidden_size: int = 1536) -> Optional[FastCacheBlockWrapper]:
    """Get FastCache wrapper for a block"""
    global _fastcache_state
    if _fastcache_state is None or not _fastcache_state.enabled:
        return None
    
    if not _fastcache_state.should_cache_block(block_id):
        return None
    
    return _fastcache_state.get_or_create_wrapper(block_id, hidden_size)


def get_fastcache_state() -> Optional[FastCacheState]:
    """Get global FastCache state"""
    return _fastcache_state


def is_fastcache_enabled() -> bool:
    """Check if FastCache is enabled"""
    return _fastcache_state is not None and _fastcache_state.enabled


def reset_fastcache():
    """Reset all FastCache states"""
    global _fastcache_state
    if _fastcache_state is not None:
        _fastcache_state.reset()


def finalize_fastcache():
    """Print final statistics and cleanup"""
    global _fastcache_state
    if _fastcache_state is not None:
        _fastcache_state.print_statistics()


# ============================================================================
# Block Forward Function with FastCache
# ============================================================================

def block_forward_with_fastcache(
    block: nn.Module,
    x: torch.Tensor,
    context: torch.Tensor,
    t_mod: torch.Tensor,
    freqs: torch.Tensor,
    timestep: int,
    max_timesteps: int = 1000,
    block_id: int = 0,
    hidden_size: int = 1536,
    **kwargs
) -> torch.Tensor:
    """
    Forward pass through a DiTBlock with FastCache.
    
    This is the main entry point for FastCache acceleration.
    
    Args:
        block: DiTBlock instance
        x: Input tensor
        context: Context tensor
        t_mod: Time modulation
        freqs: Frequency embeddings
        timestep: Current timestep
        max_timesteps: Maximum timesteps
        block_id: Block index
        hidden_size: Hidden dimension
        **kwargs: Additional block arguments
    
    Returns:
        Output tensor from block
    """
    # Get wrapper
    wrapper = get_fastcache_wrapper(block_id, hidden_size)
    
    if wrapper is None:
        # FastCache not enabled for this block, use normal forward
        return block(x, context, t_mod, freqs, **kwargs)
    
    # Forward with FastCache
    return wrapper.forward_with_cache(
        block,
        x,
        context,
        timestep,
        max_timesteps,
        t_mod=t_mod,
        freqs=freqs,
        **kwargs
    )
