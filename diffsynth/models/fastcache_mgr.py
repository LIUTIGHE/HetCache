"""
FastCache Manager for Wan Video DiT

Configuration and utility functions for FastCache acceleration.

Key concepts from FastCache paper:
1. Statistical Threshold: Chi-square based threshold for cache decision
2. Adaptive Threshold: Timestep-aware threshold adjustment
3. Motion Saliency: Token-level motion detection
4. Linear Approximation: Replace transformer with learned projection for static content
5. Block-level & Token-level caching: Two-stage caching strategy

References:
- FastCache paper: Hidden-state-level caching for DiT inference
- xDiT implementation: xfuser/model_executor/accelerator/fastcache.py
"""

import torch
import math
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class FastCacheConfig:
    """Configuration for FastCache acceleration"""
    
    # Core FastCache parameters
    enabled: bool = True
    cache_ratio_threshold: float = 0.05  # Base threshold for cache decision
    motion_threshold: float = 0.1  # Token-level motion detection threshold
    significance_level: float = 0.05  # Statistical significance level (chi-square)
    
    # Cache scope
    cache_blocks: list = None  # Which blocks to cache (e.g., [12, 13, 14])
    
    # Adaptive threshold parameters
    beta0: float = 0.01  # Base adaptive threshold
    beta1: float = 0.5   # Variance coefficient
    beta2: float = -0.002  # Timestep coefficient (linear)
    beta3: float = 0.00005  # Timestep coefficient (quadratic)
    
    # EMA for background state
    ema_alpha: float = 0.9  # EMA coefficient for background state
    
    # Advanced features (optional)
    enable_enhanced_linear_approx: bool = False  # Enhanced linear approximation
    enable_adacorrection: bool = False  # AdaCorrection for bias correction
    adacorr_gamma: float = 1.0  # AdaCorrection sensitivity
    adacorr_lambda: float = 1.0  # AdaCorrection spatial weight
    
    # For debugging
    verbose: bool = False


def compute_relative_change(current: torch.Tensor, previous: torch.Tensor) -> float:
    """
    Compute relative change (delta) between current and previous hidden states.
    
    Formula: δ = ||H_t - H_{t-1}||_F / ||H_{t-1}||_F
    
    Args:
        current: Current hidden states [batch, seq_len, hidden_dim]
        previous: Previous hidden states [batch, seq_len, hidden_dim]
    
    Returns:
        Relative change as a scalar
    """
    if previous is None:
        return float('inf')
    
    # Frobenius norm of difference
    diff_norm = torch.norm(current - previous, p='fro')
    prev_norm = torch.norm(previous, p='fro')
    
    if prev_norm == 0:
        return float('inf')
    
    return (diff_norm / prev_norm).item()


def get_statistical_threshold(
    hidden_states: torch.Tensor,
    significance_level: float = 0.05
) -> float:
    """
    Compute statistical threshold based on chi-square distribution.
    
    Formula: threshold = sqrt(χ²_{ND,1-α} / (N*D))
    where N = num_tokens, D = hidden_dim, α = significance_level
    
    For large DOF, χ²_{ND,1-α} ≈ ND + z_{1-α} * sqrt(2*ND)
    
    Args:
        hidden_states: Input hidden states [batch, seq_len, hidden_dim]
        significance_level: Statistical significance level (default: 0.05)
    
    Returns:
        Statistical threshold as a scalar
    """
    n, d = hidden_states.shape[1], hidden_states.shape[2]  # seq_len, hidden_dim
    dof = n * d  # degrees of freedom
    
    # Z-score for given significance level (95% confidence = 1.96)
    z = 1.96 if significance_level == 0.05 else torch.erfinv(
        torch.tensor(2 * (1 - significance_level) - 1)
    ).item() * math.sqrt(2)
    
    # Chi-square threshold approximation
    chi2_threshold = dof + z * math.sqrt(2 * dof)
    
    # Convert to relative change threshold
    statistical_threshold = math.sqrt(chi2_threshold / dof)
    
    return statistical_threshold


def get_adaptive_threshold(
    variance_score: float,
    timestep: int,
    beta0: float = 0.01,
    beta1: float = 0.5,
    beta2: float = -0.002,
    beta3: float = 0.00005,
    max_timesteps: int = 1000
) -> float:
    """
    Calculate adaptive threshold based on variance and timestep.
    
    Formula: T_adaptive = β0 + β1*δ + β2*t_norm + β3*t_norm²
    where t_norm = timestep / max_timesteps
    
    Intuition:
    - Early steps (high t): Lower threshold → more recomputation → higher quality
    - Later steps (low t): Higher threshold → more caching → faster inference
    - High variance: Lower threshold → more recomputation
    
    Args:
        variance_score: Computed relative change (delta)
        timestep: Current denoising timestep
        beta0, beta1, beta2, beta3: Adaptive threshold coefficients
        max_timesteps: Maximum timesteps for normalization
    
    Returns:
        Adaptive threshold as a scalar
    """
    # Normalize timestep to [0, 1]
    normalized_timestep = timestep / max_timesteps
    
    # Compute adaptive threshold
    adaptive_threshold = (
        beta0 + 
        beta1 * variance_score + 
        beta2 * normalized_timestep + 
        beta3 * (normalized_timestep ** 2)
    )
    
    return adaptive_threshold


def should_use_cache(
    hidden_states: torch.Tensor,
    prev_hidden_states: torch.Tensor,
    timestep: int,
    config: FastCacheConfig,
    max_timesteps: int = 1000
) -> Tuple[bool, Dict[str, float]]:
    """
    Determine if cached states should be used based on statistical test.
    
    Decision logic:
    1. Compute delta = relative_change(current, previous)
    2. Compute statistical_threshold (chi-square based)
    3. Compute adaptive_threshold (timestep & variance aware)
    4. Final threshold = max(cache_ratio_threshold, min(statistical, adaptive))
    5. Use cache if delta <= final_threshold
    
    Args:
        hidden_states: Current hidden states
        prev_hidden_states: Previous hidden states
        timestep: Current timestep
        config: FastCache configuration
        max_timesteps: Maximum timesteps for normalization
    
    Returns:
        (use_cache, debug_info) tuple
    """
    if prev_hidden_states is None:
        return False, {}
    
    # 1. Compute relative change
    delta = compute_relative_change(hidden_states, prev_hidden_states)
    
    # 2. Compute statistical threshold
    statistical_threshold = get_statistical_threshold(
        hidden_states,
        significance_level=config.significance_level
    )
    
    # 3. Compute adaptive threshold
    adaptive_threshold = get_adaptive_threshold(
        variance_score=delta,
        timestep=timestep,
        beta0=config.beta0,
        beta1=config.beta1,
        beta2=config.beta2,
        beta3=config.beta3,
        max_timesteps=max_timesteps
    )
    
    # 4. Combine thresholds
    final_threshold = max(
        config.cache_ratio_threshold,
        min(statistical_threshold, adaptive_threshold)
    )
    
    # 5. Cache decision
    use_cache = delta <= final_threshold
    
    # Debug info
    debug_info = {
        'delta': delta,
        'statistical_threshold': statistical_threshold,
        'adaptive_threshold': adaptive_threshold,
        'final_threshold': final_threshold,
        'use_cache': use_cache
    }
    
    return use_cache, debug_info


def compute_motion_saliency(
    hidden_states: torch.Tensor,
    prev_hidden_states: torch.Tensor,
    motion_threshold: float = 0.1
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute motion saliency for each token.
    
    Formula: S_t[i] = ||H_t[i] - H_{t-1}[i]||²
    
    Tokens are classified as:
    - Motion tokens: S_t[i] > motion_threshold
    - Static tokens: S_t[i] <= motion_threshold
    
    Args:
        hidden_states: Current hidden states [batch, seq_len, hidden_dim]
        prev_hidden_states: Previous hidden states [batch, seq_len, hidden_dim]
        motion_threshold: Threshold for motion detection
    
    Returns:
        (motion_saliency, motion_mask, static_mask) tuple
        - motion_saliency: Normalized saliency per token [batch, seq_len]
        - motion_mask: Boolean mask for motion tokens [batch, seq_len]
        - static_mask: Boolean mask for static tokens [batch, seq_len]
    """
    if prev_hidden_states is None:
        # First step: all tokens are "motion"
        batch_size, seq_len = hidden_states.shape[0], hidden_states.shape[1]
        motion_saliency = torch.ones(batch_size, seq_len, device=hidden_states.device)
        motion_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=hidden_states.device)
        static_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=hidden_states.device)
        return motion_saliency, motion_mask, static_mask
    
    # Compute token-wise squared difference
    token_diffs = (hidden_states - prev_hidden_states) ** 2
    motion_saliency = token_diffs.sum(dim=-1)  # [batch, seq_len]
    
    # Normalize saliency to [0, 1]
    max_saliency = motion_saliency.max()
    if max_saliency > 0:
        motion_saliency = motion_saliency / max_saliency
    
    # Create masks
    motion_mask = motion_saliency > motion_threshold
    static_mask = ~motion_mask
    
    return motion_saliency, motion_mask, static_mask


def update_background_state(
    current_hidden_states: torch.Tensor,
    bg_hidden_states: Optional[torch.Tensor],
    ema_alpha: float = 0.9
) -> torch.Tensor:
    """
    Update background state with exponential moving average.
    
    Formula: BG_t = α * BG_{t-1} + (1-α) * H_t
    
    Args:
        current_hidden_states: Current hidden states
        bg_hidden_states: Previous background state (None for first step)
        ema_alpha: EMA coefficient (default: 0.9)
    
    Returns:
        Updated background state
    """
    if bg_hidden_states is None:
        return current_hidden_states.detach().clone()
    
    return ema_alpha * bg_hidden_states + (1 - ema_alpha) * current_hidden_states.detach()


def print_fastcache_stats(
    cache_hits: int,
    total_steps: int,
    layer_stats: Optional[Dict[int, int]] = None
):
    """
    Print FastCache statistics.
    
    Args:
        cache_hits: Total number of cache hits
        total_steps: Total number of steps
        layer_stats: Per-layer cache hit counts (optional)
    """
    if total_steps == 0:
        print("⚠️  FastCache: No steps recorded")
        return
    
    hit_ratio = cache_hits / total_steps
    print(f"\n{'='*80}")
    print(f"📊 FastCache Statistics")
    print(f"{'='*80}")
    print(f"  Overall Cache Hit Ratio: {hit_ratio:.2%} ({cache_hits}/{total_steps})")
    
    if layer_stats:
        print(f"\n  Per-Layer Statistics:")
        for layer_idx, hits in sorted(layer_stats.items()):
            layer_ratio = hits / total_steps if total_steps > 0 else 0.0
            print(f"    Layer {layer_idx:2d}: {layer_ratio:.2%} ({hits}/{total_steps})")
    
    print(f"{'='*80}\n")
