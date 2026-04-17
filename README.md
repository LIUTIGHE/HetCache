# Accelerating Diffusion-based Video Editing via Heterogeneous Caching: Beyond Full Computing at Sampled Denoising Timesteps

<p align="center">
  <a href="https://arxiv.org/abs/2603.24260"><img src="https://img.shields.io/badge/arXiv-2603.24260-b31b1b.svg" alt="arXiv"></a>
  <a href="#license"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg" alt="PyTorch"></a>
  <a href="#citation"><img src="https://img.shields.io/badge/CVPR-2026-4b44ce.svg" alt="CVPR 2026"></a>
</p>

> **Official implementation** of the CVPR 2026 paper:
>
> *Accelerating Diffusion-based Video Editing via Heterogeneous Caching: Beyond Full Computing at Sampled Denoising Timesteps*
>
> Tianyi Liu, Ye Lu, Linfeng Zhang, Chen Cai, Jianjun Gao, Yi Wang, Kim-Hui Yap, Lap-Pui Chau
>
> [Paper (arXiv)](https://arxiv.org/abs/2603.24260)

---

## Overview

HetCache is a **training-free** method to accelerate diffusion-based video editing pipelines (e.g., inpainting, outpainting) by exploiting the **heterogeneous** token structure in masked video-to-video generation.

### Key Idea

In masked video editing, different spatial tokens play fundamentally different roles. HetCache introduces a **three-level timestep decision** mechanism combined with **heterogeneous token caching**:

1. **Full Compute** (large feature change): All tokens are computed.
2. **Partial Compute** (moderate change): Only generative + sampled margin/context tokens are computed; the rest are retrieved from cache.
3. **Reuse** (small change): Entire output is reused from cache.

| Token Type | Description |
|---|---|
| **Generative** | Inside the editing mask — requires full denoising |
| **Margin** | Near the mask boundary — ensures seamless blending |
| **Context** | Background / far from mask — already known content |

---

## Installation

```bash
git clone https://github.com/LIUTIGHE/HetCache.git
cd HetCache

# Create environment
conda create -n hetcache python=3.10 -y
conda activate hetcache

# Install dependencies
pip install -r requirements.txt
```

### Model Download

HetCache uses [Wan2.1-VACE-1.3B](https://huggingface.co/Wan-AI/Wan2.1-VACE-1.3B) as the base model. It will be downloaded automatically on first run via `modelscope` / HuggingFace.

---

## Quick Start

### Baseline (No Acceleration)

```bash
python inference.py \
    --mode baseline \
    --video data/real.mp4 \
    --mask data/masks.mp4 \
    --frames 33 --steps 50 \
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

### HetCache (Ours)

```bash
python inference.py \
    --mode hetcache \
    --video data/real.mp4 \
    --mask data/masks.mp4 \
    --frames 33 --steps 50 \
    --cache-thresh 0.05 \
    --context-ratio 0.7 \
    --margin-ratio 1.0 \
    --use-kmeans --kmeans-clusters 16 \
    --use-attention-interaction \
    --apply-mask-to-input
```

### FastCache

**Note:** FastCache implementation may not be fully accurate. Just for theoretical reference.

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
| `pab` | Pyramid Attention Broadcast | ✓ | ✗ |
| `adacache` | AdaCache adaptive caching | ✓ | ✗ |
| `teacache` | TeaCache timestep-level caching | ✓ | ✗ |
| `hetcache` | **HetCache** (ours) — heterogeneous token caching | ✓ | ✓ |

Combine any baseline with TeaCache by appending `_teacache` (e.g., `pab_teacache`).

---

## Results

### Wan2.1-VACE-1.3B — Video Inpainting (VACE-Benchmark, 50 steps, 33 frames)

| Method | Theoretical PFLOPs ↓ | Time (s) ↓ | Speedup ↑ |
|---|---|---|---|
| Timestep Reduction (baseline) | 72.60 | 238.84 | 1.00× |
| TeaCache-slow (Δ=0.05) | 47.19 | 224.53 | 1.06× |
| TeaCache-fast (Δ=0.02) | 36.30 | 186.45 | 1.28× |
| PAB | 65.02 | 221.64 | 1.08× |
| AdaCache | 58.17 | 223.58 | 1.07× |
| FastCache | 57.41 | 222.81 | 1.07× |
| **HetCache-slow (Δ=0.05)** | **30.68** | **176.31** | **1.35×** |
| **HetCache-fast (Δ=0.02)** | **23.60** | **166.81** | **1.43×** |


---

## Key Hyperparameters

| Parameter | Flag | Default | Description |
|---|---|---|---|
| TeaCache Threshold (Δ) | `--cache-thresh` | 0.05 | L1 distance threshold for timestep skip. Lower = more aggressive. |
| Context Ratio | `--context-ratio` | 0.05 | Fraction of context tokens to compute (rest are cached). |
| Margin Ratio | `--margin-ratio` | 0.7 | Fraction of margin tokens to compute. |
| K-Means Clusters | `--kmeans-clusters` | 16 | Number of clusters for context sampling diversity. |
| Attention Interaction | `--use-attention-interaction` | off | Enable K-Means + Attention context sampling. |

---

## Evaluation

We provide `evaluation.py` with two subcommands. Our benchmark in paper is relatively resource-constrained, we welcome furture benchmark extension using our toolkits to better evaluate ours and relative methods.

| Subcommand | Metrics | GT Required? |
|---|---|---|
| `quality` | PSNR, SSIM, LPIPS, VFID | ✓ |
| `vbench` | Subject/Background Consistency, Motion Smoothness, etc. | ✗ |

### GT-Based Quality Metrics

Single video pair (PSNR / SSIM / LPIPS):

```bash
python evaluation.py quality \
    --gt data/real.mp4 \
    --pred data/test_hetcache.mp4
```

Batch evaluation (PSNR / SSIM / LPIPS + VFID):

```bash
python evaluation.py quality \
    --gt-dir /path/to/gt_videos \
    --pred-dir /path/to/pred_videos \
    --method hetcache \
    --i3d-checkpoint i3d_rgb_imagenet.pt
```

VFID is computed automatically in batch mode when [I3D weights](https://github.com/piergiaj/pytorch-i3d) (`i3d_rgb_imagenet.pt`) are available. GT videos are truncated to match the generated video length before I3D feature extraction (8-frame temporal sampling by default).

> **⚠️ Note on VFID frame alignment**
>
> During the preparation of this work, we identified a frame-alignment issue in our VFID evaluation: when GT videos are longer than generated videos (e.g., GT has 80–240 frames but generated videos have 33 frames), the GT must be truncated to the same temporal range before I3D feature extraction. Without truncation, the I3D features encode different temporal content, inflating VFID values. Our corrected `evaluation.py` automatically truncates GT to match the generated video length. The **relative ranking between methods is unaffected**, but absolute VFID values in the paper are higher than the corrected values. This does not affect other metrics (PSNR, SSIM, LPIPS, VBench) which use frame-aligned comparisons.

### Reference-Free VBench Metrics

```bash
python evaluation.py vbench \
    --pred-dir /path/to/pred_videos \
    --method hetcache \
    --vbench-script VBench/evaluate.py \
    --conda-env vbench
```

Default dimensions: `subject_consistency`, `background_consistency`, `motion_smoothness`, `dynamic_degree`, `aesthetic_quality`, `imaging_quality`. Requires [VBench](https://github.com/Vchitect/VBench).

---

## Project Structure

```
HetCache/
├── inference.py              # Main entry point (CLI)
├── evaluation.py             # Evaluation (quality: PSNR/SSIM/LPIPS/VFID, vbench)
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
@inproceedings{liu2026hetcache,
    title     = {Accelerating Diffusion-based Video Editing via Heterogeneous Caching: Beyond Full Computing at Sampled Denoising Timesteps},
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
