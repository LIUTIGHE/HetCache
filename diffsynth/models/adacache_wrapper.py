"""
AdaCache Wrapper for Wan Video DiT

Modular wrapper that implements AdaCache without modifying DiTBlock internals.
Wraps around the block's forward pass and manages caching decisions.

Key principles:
- Non-invasive: Works as a wrapper, doesn't modify DiTBlock
- Modular: Can be enabled/disabled independently
- Compatible: Works alongside PAB, TeaCache, TokenCache, etc.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any


class AdaCacheBlockWrapper:
    """
    Wrapper for a single DiTBlock to add AdaCache functionality.
    
    Manages:
    - Spatial attention caching
    - Cross-attention + MLP caching
    - Cache rate adaptation based on feature differences
    - Motion regularization (optional)
    """
    
    def __init__(
        self,
        block_idx: int,
        config: 'AdaCacheConfig',
    ):
        self.block_idx = block_idx
        self.config = config
        
        # Caches
        self.spatial_cache = None
        self.cross_mlp_cache = None
        
        # State
        self.prev_rate = 1
        self.next_compute_step = 2
        self.prev_motion = 1.0
        self.recomputed_steps = []
        
        # Whether this block should use caching
        self.should_cache = block_idx in config.cache_blocks
    
    def reset(self):
        """Reset cache for new sequence"""
        self.spatial_cache = None
        self.cross_mlp_cache = None
        self.prev_rate = 1
        self.next_compute_step = 2
        self.prev_motion = 1.0
        self.recomputed_steps = []
    
    def should_compute_spatial(self, current_step: int) -> bool:
        """Check if we should compute spatial attention or use cache"""
        if not self.should_cache:
            return True
        
        # Always compute first/last steps
        skip_steps = [1, self.config.num_steps - 1, self.config.num_steps]
        if current_step in skip_steps:
            return True
        
        # Check if cache module includes spatial
        if self.config.cache_module not in ['spatial', 'both']:
            return True
        
        # Check if we've reached the compute step
        if current_step >= self.next_compute_step or self.spatial_cache is None:
            return True
        
        return False
    
    def should_compute_cross_mlp(self, current_step: int) -> bool:
        """Check if we should compute cross-attn+MLP or use cache"""
        if not self.should_cache:
            return True
        
        # Always compute first/last steps
        skip_steps = [1, self.config.num_steps - 1, self.config.num_steps]
        if current_step in skip_steps:
            return True
        
        # Check if cache module includes cross_mlp
        if self.config.cache_module not in ['cross_mlp', 'both']:
            return True
        
        # Check if we've reached the compute step
        if current_step >= self.next_compute_step or self.cross_mlp_cache is None:
            return True
        
        return False
    
    def update_cache_rate(
        self,
        cached_features: torch.Tensor,
        current_features: torch.Tensor,
        current_step: int,
        ada_state: 'AdaCacheState',
        spatial_size: int = None,
        verbose: bool = False,
    ):
        """
        Compute cache difference and update cache rate.
        
        Args:
            cached_features: Previously cached features
            current_features: Newly computed features
            current_step: Current sampling step
            ada_state: Global AdaCache state
            spatial_size: Spatial dimension S (for motion computation)
            verbose: Print debug info
        """
        from .adacache_mgr import (
            compute_cache_diff,
            select_cache_rate_from_codebook,
            compute_motion_score,
            apply_motion_regularization,
        )
        
        # Compute cache difference
        cache_diff = compute_cache_diff(
            cached_features,
            current_features,
            self.prev_rate,
            norm_ord=1,
        )
        
        # Sync across layers (update running average)
        cache_diff = (cache_diff + ada_state.count * ada_state.avg_diff) / (ada_state.count + 1)
        ada_state.avg_diff = cache_diff
        ada_state.count += 1
        
        # Apply motion regularization if enabled
        if self.config.enable_moreg and spatial_size is not None:
            moreg_start, moreg_end = self.config.moreg_steps
            
            if moreg_start <= current_step <= moreg_end:
                # Compute motion score
                motion = compute_motion_score(
                    current_features,
                    num_frames=current_features.shape[1] // spatial_size,
                    spatial_size=spatial_size,
                    strides=self.config.moreg_strides,
                    norm_ord=1,
                )
                
                # Apply motion regularization
                cache_diff = apply_motion_regularization(
                    cache_diff,
                    motion,
                    self.prev_motion,
                    self.prev_rate,
                    self.config.moreg_hyp,
                    self.config.mograd_mul,
                )
                
                self.prev_motion = motion
        
        # Select new cache rate from codebook
        new_rate = select_cache_rate_from_codebook(
            cache_diff,
            self.config.codebook,
        )
        
        # Update state
        self.prev_rate = new_rate
        self.next_compute_step = current_step + new_rate
        ada_state.cache_rate = new_rate
        
        if verbose and self.block_idx == self.config.cache_blocks[-1]:
            module_name = "spatial" if self.config.cache_module == "spatial" else "cross+mlp"
            print(f'{module_name} - step {str(current_step).zfill(3)} - cache_diff {cache_diff:.3f} - rate {new_rate}')
        
        return new_rate


def block_forward_with_adacache(
    block: nn.Module,
    x: torch.Tensor,
    context: torch.Tensor,
    t_mod: torch.Tensor,
    freqs: torch.Tensor,
    wrapper: AdaCacheBlockWrapper,
    ada_state: 'AdaCacheState',
    current_step: int,
    spatial_size: int = None,
    **kwargs,
) -> torch.Tensor:
    """
    Forward pass through DiTBlock with AdaCache wrapper.
    
    This function wraps around the block's forward pass and manages caching.
    It does NOT modify the block's internal logic.
    
    Args:
        block: The DiTBlock to wrap
        x: Input tensor
        context: Cross-attention context
        t_mod: Time modulation
        freqs: RoPE frequencies
        wrapper: AdaCache wrapper for this block
        ada_state: Global AdaCache state
        current_step: Current sampling step
        spatial_size: Spatial dimension S
        **kwargs: Additional arguments to pass to block
    
    Returns:
        Output tensor
    """
    config = wrapper.config
    
    # Check if we should compute or use cache
    compute_spatial = wrapper.should_compute_spatial(current_step)
    compute_cross_mlp = wrapper.should_compute_cross_mlp(current_step)
    
    # Case 1: Compute everything normally
    if compute_spatial and compute_cross_mlp:
        # Normal forward pass
        result = block(x, context, t_mod, freqs, **kwargs)
        
        # Extract output (handle tuple returns)
        if isinstance(result, tuple):
            output = result[0]
        else:
            output = result
        
        # Update caches if this block should cache
        if wrapper.should_cache:
            # For spatial: we cache the attention output before cross-attn
            # For cross_mlp: we cache the combined residual
            # To avoid modifying block internals, we'll cache the final output
            # and compute cache_diff on it
            
            if config.cache_module in ['spatial', 'both']:
                if wrapper.spatial_cache is not None:
                    # Update cache rate based on difference
                    wrapper.update_cache_rate(
                        wrapper.spatial_cache,
                        output,
                        current_step,
                        ada_state,
                        spatial_size,
                        verbose=True,
                    )
                
                wrapper.spatial_cache = output.clone()
            
            if config.cache_module in ['cross_mlp', 'both']:
                wrapper.cross_mlp_cache = output.clone()
            
            wrapper.recomputed_steps.append(current_step)
        
        return result
    
    # Case 2: Skip spatial, compute cross_mlp
    elif not compute_spatial and compute_cross_mlp:
        # This is complex - we'd need to modify block internals
        # For simplicity, use full cache (compute everything or nothing)
        # Fall back to computing
        result = block(x, context, t_mod, freqs, **kwargs)
        if isinstance(result, tuple):
            output = result[0]
        else:
            output = result
        return result
    
    # Case 3: Skip both, use cache
    else:
        # Reuse cached output
        if wrapper.spatial_cache is not None:
            # Add cached residual to input
            cached_residual = wrapper.spatial_cache - x
            output = x + cached_residual
            
            # Return in same format as block would
            if 'return_attention_weights' in kwargs and kwargs['return_attention_weights']:
                return output, None, None
            elif 'indices_compute' in kwargs and kwargs['indices_compute'] is not None:
                return output, {}
            else:
                return output
        else:
            # No cache available, compute normally
            result = block(x, context, t_mod, freqs, **kwargs)
            return result


class AdaCacheState:
    """Global state for AdaCache across all blocks"""
    
    def __init__(self, config: 'AdaCacheConfig'):
        self.config = config
        self.current_step = 0
        self.avg_diff = 0.0
        self.count = 0
        self.cache_rate = 1
    
    def reset(self):
        """Reset for new sampling sequence"""
        self.current_step = 0
        self.avg_diff = 0.0
        self.count = 0
        self.cache_rate = 1
    
    def step(self):
        """Advance to next step"""
        self.current_step += 1
        if self.current_step > self.config.num_steps:
            self.current_step = 0
    
    def reset_per_step(self):
        """Reset per-step counters (call at start of each timestep)"""
        self.avg_diff = 0.0
        self.count = 0


class AdaCacheConfig:
    """Configuration for AdaCache"""
    
    def __init__(
        self,
        enabled: bool = True,
        cache_module: str = 'spatial',
        cache_blocks: list = None,
        codebook: dict = None,
        num_steps: int = 30,
        enable_moreg: bool = False,
        moreg_steps: tuple = (3, 27),
        moreg_strides: list = None,
        moreg_hyp: tuple = (0.385, 8, 1, 2),
        mograd_mul: float = 10.0,
    ):
        self.enabled = enabled
        self.cache_module = cache_module
        self.cache_blocks = cache_blocks or [13]
        self.num_steps = num_steps
        
        # Default codebook
        if codebook is None:
            if num_steps <= 30:
                self.codebook = {
                    0.08: 6,
                    0.16: 5,
                    0.24: 4,
                    0.32: 3,
                    0.40: 2,
                    1.00: 1,
                }
            else:
                self.codebook = {
                    0.03: 12,
                    0.05: 10,
                    0.07: 8,
                    0.09: 6,
                    0.11: 4,
                    1.00: 3,
                }
        else:
            self.codebook = codebook
        
        self.enable_moreg = enable_moreg
        self.moreg_steps = moreg_steps
        self.moreg_strides = moreg_strides or [1]
        self.moreg_hyp = moreg_hyp
        self.mograd_mul = mograd_mul


# Global state
_ADACACHE_WRAPPERS = {}
_ADACACHE_STATE = None


def initialize_adacache(config: AdaCacheConfig, num_blocks: int):
    """Initialize AdaCache wrappers for all blocks"""
    global _ADACACHE_WRAPPERS, _ADACACHE_STATE
    
    _ADACACHE_STATE = AdaCacheState(config)
    _ADACACHE_WRAPPERS = {}
    
    for block_idx in range(num_blocks):
        _ADACACHE_WRAPPERS[block_idx] = AdaCacheBlockWrapper(block_idx, config)
    
    print(f"✅ AdaCache initialized: {config.cache_module} caching on blocks {config.cache_blocks}")


def get_adacache_wrapper(block_idx: int) -> Optional[AdaCacheBlockWrapper]:
    """Get wrapper for a specific block"""
    global _ADACACHE_WRAPPERS
    return _ADACACHE_WRAPPERS.get(block_idx)


def get_adacache_state() -> Optional[AdaCacheState]:
    """Get global AdaCache state"""
    global _ADACACHE_STATE
    return _ADACACHE_STATE


def reset_adacache():
    """Reset all AdaCache state"""
    global _ADACACHE_WRAPPERS, _ADACACHE_STATE
    
    if _ADACACHE_STATE is not None:
        _ADACACHE_STATE.reset()
    
    for wrapper in _ADACACHE_WRAPPERS.values():
        wrapper.reset()


def is_adacache_enabled() -> bool:
    """Check if AdaCache is enabled"""
    global _ADACACHE_STATE
    return _ADACACHE_STATE is not None and _ADACACHE_STATE.config.enabled


def finalize_adacache():
    """Print statistics and cleanup"""
    global _ADACACHE_WRAPPERS, _ADACACHE_STATE
    
    if _ADACACHE_STATE is None:
        return
    
    # Print statistics for cached blocks
    for block_idx in _ADACACHE_STATE.config.cache_blocks:
        wrapper = _ADACACHE_WRAPPERS.get(block_idx)
        if wrapper and wrapper.recomputed_steps:
            print(f'Block {block_idx}: ({len(wrapper.recomputed_steps)}/{_ADACACHE_STATE.config.num_steps}) '
                  f'recomputed steps {wrapper.recomputed_steps}')
