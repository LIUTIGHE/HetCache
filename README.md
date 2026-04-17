# HetCache: Training-Free Acceleration for Diffusion-Based Video Editing via Heterogeneous Caching

<p align="center">
  <a href="https://arxiv.org/abs/2603.24260"><img src="https://img.shields.io/badge/arXiv-2603.24260-b31b1b.svg" alt="arXiv"></a>
  <a href="#license"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg" alt="PyTorch"></a>
</p>

> **HetCache: Training-Free Acceleration for Diffusion-Based Video Editing via Heterogeneous Caching**
>
> Tianyi Liu, Ye Lu, Linfeng Zhang, Chen Cai, Jianjun Gao, Yi Wang, Kim-Hui Yap, Lap-Pui Chau
>
> **CVPR 2026** | [Paper (arXiv)](https://arxiv.org/abs/2603.24260)

---

## Overview

HetCache is a **training-free** method to accelerate diffusion-based video editing pipelines (e.g., inpainting, outpainting) by exploiting the **heterogeneous** token structure in masked video-to-video generation.

### Key Idea

In masked video editing, different spatial tokens play fundamentally different roles:

| Token Type | Description | Compute Ratio |
|---|---|---|
| **Generative** | Inside the editing mask — requires full denoising | 100% (always computed) |
| **Margin** | Near the mask boundary — ensures seamless blending | ~70% (subsampled) |
| **Context** | Background / far from mask — already known content | ~5% (sparse sampling) |

HetCache introduces a **three-level timestep decision** mechanism combined with **heterogeneous token caching**:

1. **Full Compute** (large feature change): All tokens are computed.
2. **Partial Compute** (moderate change): Only generative + sampled margin/context tokens are computed; the rest are retrieved from cache.
3. **Full Reuse** (small change): Entire output is reused from cache.

### Two-Stage Context Sampling

For the small fraction of context tokens that are computed, HetCache uses a two-stage sampling strategy:
- **Stage 1 — K-Means Clustering**: Groups context tokens by semantic similarity to ensure spatial diversity.
- **Stage 2 — Attention-Interaction Scoring**: Ranks tokens within each cluster by their attention interaction strength with generative tokens, selecting the most relevant ones.

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/HetCache.git
cd HetCache

# Create environment
conda create -n hetcache python=3.10 -y
conda activate hetcache

# Install dependencies
pip install -r requirements.txt
```

### Model Download

HetCache uses [Wan2.1-VACE](https://huggingface.co/Wan-AI/Wan2.1-VACE-1.3B) as the base model. It will be downloaded automatically on first run via `modelscope` / HuggingFace.

---

## Quick Start

### HetCache (Ours)

```bash
python inference.py \
    --mode hetcache \
    --video data/real.mp4 \
    --mask data/masks.mp4 \
    --frames 33 --steps 50 \
    --cache-thresh 0.05 \
    --context-ratio 0.05 \
    --margin-ratio 0.7 \
    --use-kmeans --kmeans-clusters 16 \
    --use-attention-interaction \
    --apply-mask-to-input
```

### Baseline (No Acceleration)

```bash
python inference.py \
    --mode baseline \
    --video data/real.mp4 \
    --mask data/masks.mp4 \
    --frames 33 --steps 50 \
    --apply-mask-to-input
```

### TeaCache

```bash
python inference.py \
    --mode teacache \
    --video data/real.mp4 \
    --mask data/masks.mp4 \
    --frames 33 --steps 50 \
    --cache-thresh 0.05 \
    --apply-mask-to-input
```

### PAB (Pyramid Attention Broadcast)

```bash
python inference.py \
    --mode pab \
    --video data/real.mp4 \
    --mask data/masks.mp4 \
    --frames 33 --steps 50 \
    --apply-mask-to-input
```

### AdaCache

```bash
python inference.py \
    --mode adacache \
    --video data/real.mp4 \
    --mask data/masks.mp4 \
    --frames 33 --steps 50 \
    --apply-mask-to-input
```

### FastCache

```bash
python inference.py \
    --mode fastcache \
    --video data/real.mp4 \
    --mask data/masks.mp4 \
    --frames 33 --steps 50 \
    --apply-mask-to-input
```

---

## Available Modes

| Mode | Description | Timestep Cache | Token Cache |
|---|---|---|---|
| `baseline` | No acceleration | ✗ | ✗ |
| `teacache` | TeaCache timestep-level caching | ✓ | ✗ |
| `hetcache` | **HetCache** (ours) — heterogeneous token caching | ✓ | ✓ |
| `pab` | Pyramid Attention Broadcast | ✓ (broadcast) | ✗ |
| `adacache` | AdaCache adaptive caching | ✓ (codebook) | ✗ |
| `fastcache` | FastCache statistical caching | ✓ (chi-square) | ✗ |

Combine any baseline with TeaCache by appending `_teacache` (e.g., `pab_teacache`).

---

## Results

### Wan2.1-VACE-1.3B — Video Inpainting (VACE-Benchmark, 50 steps)

| Method | PFLOPs ↓ | Time (s) ↓ | Speedup ↑ |
|---|---|---|---|
| Timestep Reduction (baseline) | 72.60 | 238.84 | 1.00× |
| TeaCache-slow (Δ=0.05) | 47.19 | 224.53 | 1.06× |
| TeaCache-fast (Δ=0.02) | 36.30 | 186.45 | 1.28× |
| PAB | 65.02 | 221.64 | 1.08× |
| AdaCache | 58.17 | 223.58 | 1.07× |
| FastCache | 57.41 | 222.81 | 1.07× |
| **HetCache-slow (Δ=0.05)** | **30.68** | **176.31** | **1.35×** |
| **HetCache-fast (Δ=0.02)** | **23.60** | **166.81** | **1.43×** |

> Speedup is computed relative to the 50-step timestep reduction baseline. All methods also include the timestep reduction from the base scheduler (100→50 steps corresponds to a 2× speedup over the full 100-step baseline).

### Wan2.1-VACE-1.3B — Video Outpainting (50 steps)

| Method | PFLOPs ↓ | Time (s) ↓ |
|---|---|---|
| Timestep Reduction | 72.60 | 241.02 |
| TeaCache-slow | 47.19 | 225.05 |
| **HetCache-slow** | **30.68** | **177.72** |

---

## Key Hyperparameters

| Parameter | Flag | Default | Description |
|---|---|---|---|
| TeaCache Threshold (Δ) | `--cache-thresh` | 0.05 | L1 distance threshold for timestep skip. Lower = more aggressive. |
| Context Ratio | `--context-ratio` | 0.05 | Fraction of context tokens to compute (rest are cached). |
| Margin Ratio | `--margin-ratio` | 0.7 | Fraction of margin tokens to compute. |
| K-Means Clusters | `--kmeans-clusters` | 16 | Number of clusters for context sampling diversity. |
| Attention Interaction | `--use-attention-interaction` | off | Enable two-stage (K-Means + Attention) context sampling. |

---

## Method Architecture

```
Timestep t
    │
    ├─ Compute L1 distance Δ(t) via polynomial-rescaled TeaCache
    │
    ├─ If Δ(t) > 1.5×thresh  ──→  FULL COMPUTE (all tokens)
    │
    ├─ If thresh < Δ(t) < 1.5×thresh  ──→  PARTIAL COMPUTE
    │     ├─ Generative tokens: always computed (100%)
    │     ├─ Margin tokens: subsampled (~70%)
    │     └─ Context tokens: sparse two-stage sampling (~5%)
    │           ├─ Stage 1: K-Means clustering (semantic diversity)
    │           └─ Stage 2: Attention scoring (relevance ranking)
    │
    └─ If Δ(t) < thresh  ──→  FULL REUSE (entire output from cache)
```

---

## Project Structure

```
HetCache/
├── inference.py              # Main entry point (CLI)
├── re_PAB_mgr.py             # PAB baseline manager
├── hetcache/
│   └── __init__.py           # Re-exports: HetCache, TeaCache, WanVideoPipeline
├── diffsynth/                # Backend framework (DiffSynth-Studio)
│   ├── pipelines/
│   │   └── wan_video_new.py  # HetCache (MaskedTokenCache), TeaCache, Pipeline
│   ├── models/
│   │   ├── wan_video_dit.py  # DiT model with caching hooks
│   │   ├── adacache_mgr.py   # AdaCache baseline manager
│   │   ├── adacache_wrapper.py
│   │   ├── fastcache_mgr.py  # FastCache baseline manager
│   │   └── fastcache_wrapper.py
│   └── ...
├── data/                     # Example test data
│   ├── real.mp4
│   └── masks.mp4
├── scripts/
│   └── benchmark.sh          # Benchmark all methods
├── requirements.txt
├── setup.py
└── LICENSE
```

---

## Citation

If you find HetCache useful, please cite our paper:

```bibtex
@inproceedings{liu2025hetcache,
    title     = {HetCache: Training-Free Acceleration for Diffusion-Based Video Editing via Heterogeneous Caching},
    author    = {Liu, Tianyi and Lu, Ye and Zhang, Linfeng and Cai, Chen and Gao, Jianjun and Wang, Yi and Yap, Kim-Hui and Chau, Lap-Pui},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year      = {2026}
}
```

---

## Acknowledgments

- [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) — Base diffusion framework
- [Wan2.1-VACE](https://github.com/Wan-Video/Wan2.1) — Base video editing model
- [TeaCache](https://github.com/LiewFeng/TeaCache) — Timestep-level caching baseline
- [PAB](https://github.com/Pyramid-Attention-Broadcast/PAB) — Pyramid Attention Broadcast baseline
- [AdaCache](https://github.com/AdaCache/AdaCache) — Adaptive caching baseline
- [FastCache](https://github.com/FastCache/FastCache) — Statistical caching baseline

## License

This project is released under the [Apache 2.0 License](LICENSE).
