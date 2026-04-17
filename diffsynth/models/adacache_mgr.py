"""
AdaCache Manager for Wan Video DiT

Adaptive caching mechanism that dynamically decides when to recompute vs. reuse
cached attention/MLP outputs based on feature differences.

Key concepts:
- cache_diff: Normalized difference between cached and current results
- codebook: Maps cache_diff thresholds to skip rates
- cache_rate: How many steps to skip before recomputing (adaptive)
- MoReg (optional): Motion regularization that adjusts cache_rate based on motion

References:
- AdaCache paper and OpenSora implementation
"""

import torch
import torch.nn as nn


class AdaCacheConfig:
    """Configuration for AdaCache acceleration"""
    
    def __init__(
        self,
        enabled: bool = True,
        cache_module: str = 'spatial',  # 'spatial', 'cross_mlp', or 'both'
        cache_blocks: list = None,  # Which blocks to cache (e.g., [13, 14, 15])
        codebook: dict = None,  # Maps cache_diff threshold to skip rate
        num_steps: int = 30,
        
        # Motion regularization (MoReg) - optional
        enable_moreg: bool = False,
        moreg_steps: tuple = (3, 27),  # Step range for MoReg
        moreg_strides: list = None,  # Frame strides for motion calc
        moreg_hyp: tuple = (0.385, 8, 1, 2),  # (scale, power, divisor, unused)
        mograd_mul: float = 10.0,  # Motion gradient multiplier
    ):
        self.enabled = enabled
        self.cache_module = cache_module
        self.cache_blocks = cache_blocks or [13]  # Default: middle layer
        self.num_steps = num_steps
        
        # Default codebook for 30-step sampling
        # Format: {cache_diff_threshold: skip_rate}
        # Lower cache_diff → more skipping → faster but less accurate
        if codebook is None:
            if num_steps <= 30:
                # Fast codebook for 30 steps
                self.codebook = {
                    0.08: 6,  # Very similar → skip 6 steps
                    0.16: 5,
                    0.24: 4,
                    0.32: 3,
                    0.40: 2,
                    1.00: 1,  # Very different → recompute every step
                }
            else:
                # Slower codebook for 100 steps
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
        
        # Motion regularization
        self.enable_moreg = enable_moreg
        self.moreg_steps = moreg_steps
        self.moreg_strides = moreg_strides or [1]
        self.moreg_hyp = moreg_hyp
        self.mograd_mul = mograd_mul


class AdaCacheState:
    """
    Global state manager for AdaCache across all blocks.
    
    Tracks:
    - Current step in sampling
    - Average cache_diff across layers (for inter-layer synchronization)
    - Global cache_rate that all blocks can reference
    """
    
    def __init__(self, config: AdaCacheConfig):
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
        """Advance to next sampling step"""
        self.current_step += 1
        if self.current_step > self.config.num_steps:
            self.reset()
    
    def update_cache_diff(self, new_diff: float):
        """Update running average of cache_diff across layers"""
        self.avg_diff = (self.avg_diff * self.count + new_diff) / (self.count + 1)
        self.count += 1
        
    def get_ada_dict(self):
        """Get dictionary for passing to blocks"""
        return {
            'avg_diff': self.avg_diff,
            'count': self.count,
            'attn_cache_rate': self.cache_rate,
            'new_attn_cache_rate': self.cache_rate,
        }


def compute_cache_diff(
    cached: torch.Tensor,
    current: torch.Tensor,
    prev_rate: int,
    norm_ord: int = 1,
) -> float:
    """
    Compute normalized difference between cached and current features.
    
    Args:
        cached: Cached feature tensor [B, N, C]
        current: Current computed feature tensor [B, N, C]
        prev_rate: Previous cache rate (for normalization)
        norm_ord: Norm order (1 for L1, 2 for L2)
    
    Returns:
        Normalized cache_diff in range [0, inf), typically [0, 1]
    """
    diff_norm = (cached - current).norm(dim=(0, 1, 2), p=norm_ord)
    current_norm = current.norm(dim=(0, 1, 2), p=norm_ord)
    
    # Avoid division by zero
    if current_norm < 1e-8:
        return 0.0
    
    cache_diff = diff_norm / current_norm
    # Normalize by prev_rate to account for accumulation
    cache_diff = cache_diff / prev_rate
    
    return cache_diff.item()


def compute_motion_score(
    features: torch.Tensor,
    num_frames: int,
    spatial_size: int,
    strides: list = [1],
    norm_ord: int = 1,
) -> float:
    """
    Compute motion score across temporal frames.
    
    Args:
        features: Feature tensor [B, T*S, C]
        num_frames: Number of frames T
        spatial_size: Spatial patch count S
        strides: Frame strides to compute motion over
        norm_ord: Norm order
    
    Returns:
        Motion score (normalized)
    """
    moreg = 0.0
    
    for stride in strides:
        # Compare features between frames stride apart
        start_idx = stride * spatial_size
        feat_curr = features[:, start_idx:, :]
        feat_prev = features[:, :-start_idx, :]
        
        motion_norm = (feat_curr - feat_prev).norm(p=norm_ord)
        total_norm = feat_curr.norm(p=norm_ord) + feat_prev.norm(p=norm_ord)
        
        if total_norm < 1e-8:
            continue
            
        moreg_i = motion_norm / total_norm
        moreg += moreg_i
    
    moreg = moreg / len(strides)
    return moreg.item()


def select_cache_rate_from_codebook(
    cache_diff: float,
    codebook: dict,
) -> int:
    """
    Select cache rate from codebook based on cache_diff threshold.
    
    Args:
        cache_diff: Computed cache difference
        codebook: Dict mapping thresholds to rates
    
    Returns:
        Cache rate (number of steps to skip)
    """
    thresholds = sorted(codebook.keys())
    
    for threshold in thresholds:
        if cache_diff < threshold:
            return codebook[threshold]
    
    # If cache_diff exceeds all thresholds, use slowest rate
    return codebook[thresholds[-1]]


def apply_motion_regularization(
    cache_diff: float,
    motion_score: float,
    prev_motion: float,
    prev_rate: int,
    moreg_hyp: tuple,
    mograd_mul: float,
) -> float:
    """
    Apply motion regularization to cache_diff.
    
    High motion → increase cache_diff → lower cache_rate → more recomputation
    
    Args:
        cache_diff: Base cache difference
        motion_score: Current motion magnitude
        prev_motion: Previous motion magnitude
        prev_rate: Previous cache rate
        moreg_hyp: (scale, power, divisor, unused)
        mograd_mul: Motion gradient multiplier
    
    Returns:
        Adjusted cache_diff
    """
    # Normalize motion score around 1.0
    scale, power, divisor, _ = moreg_hyp
    moreg = ((1 / scale * motion_score) ** power) / divisor
    
    # Motion gradient (acceleration)
    mograd = mograd_mul * (moreg - prev_motion) / prev_rate
    moreg = moreg + abs(mograd)
    
    # Apply to cache_diff
    adjusted_diff = cache_diff * moreg
    
    return adjusted_diff


# Global state singleton
_ADACACHE_STATE = None


def get_adacache_state() -> AdaCacheState:
    """Get global AdaCache state"""
    global _ADACACHE_STATE
    if _ADACACHE_STATE is None:
        raise RuntimeError("AdaCache state not initialized. Call set_adacache_state() first.")
    return _ADACACHE_STATE


def set_adacache_state(config: AdaCacheConfig):
    """Initialize global AdaCache state"""
    global _ADACACHE_STATE
    _ADACACHE_STATE = AdaCacheState(config)


def reset_adacache_state():
    """Reset AdaCache state"""
    global _ADACACHE_STATE
    if _ADACACHE_STATE is not None:
        _ADACACHE_STATE.reset()


def is_adacache_enabled() -> bool:
    """Check if AdaCache is enabled"""
    global _ADACACHE_STATE
    return _ADACACHE_STATE is not None and _ADACACHE_STATE.config.enabled
