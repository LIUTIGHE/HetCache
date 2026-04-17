"""
HetCache: Training-Free Acceleration for Diffusion-Based Video Editing
via Heterogeneous Caching.

Paper: https://arxiv.org/abs/2603.24260

Quick start:
    from hetcache import HetCache, TeaCache, WanVideoPipeline
"""

__version__ = "1.0.0"

from diffsynth.pipelines.wan_video_new import (
    MaskedTokenCache as HetCache,
    TeaCache,
    TokenTeaCache,
    WanVideoPipeline,
)

__all__ = [
    "HetCache",
    "TeaCache",
    "TokenTeaCache",
    "WanVideoPipeline",
]
