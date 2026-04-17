#!/bin/bash
# ==========================================================================
# HetCache Benchmark Script
#
# Benchmarks all acceleration methods on Wan2.1-VACE-1.3B video inpainting.
# Expected approximate timings (50 steps, 33 frames, 480x832, single GPU):
#   Baseline         : ~238s
#   TeaCache-slow    : ~225s  (Δ=0.05)
#   TeaCache-fast    : ~186s  (Δ=0.02)
#   PAB              : ~222s
#   AdaCache         : ~224s
#   FastCache        : ~223s
#   HetCache-slow    : ~176s  (Δ=0.05)
#   HetCache-fast    : ~167s  (Δ=0.02)
# ==========================================================================

set -e
cd "$(dirname "$0")/.."

VIDEO="data/real.mp4"
MASK="data/masks.mp4"
FRAMES=33
STEPS=50
COMMON="--video $VIDEO --mask $MASK --frames $FRAMES --steps $STEPS --apply-mask-to-input"

echo "============================================================"
echo "  HetCache Benchmark — $(date)"
echo "============================================================"

# 1. Baseline
echo ""
echo "[1/8] Running Baseline..."
python inference.py --mode baseline $COMMON \
    --output data/bench_baseline.mp4

# 2. TeaCache-slow (Δ=0.05)
echo ""
echo "[2/8] Running TeaCache-slow (Δ=0.05)..."
python inference.py --mode teacache $COMMON \
    --cache-thresh 0.05 \
    --output data/bench_teacache_slow.mp4

# 3. TeaCache-fast (Δ=0.02)
echo ""
echo "[3/8] Running TeaCache-fast (Δ=0.02)..."
python inference.py --mode teacache $COMMON \
    --cache-thresh 0.02 \
    --output data/bench_teacache_fast.mp4

# 4. PAB
echo ""
echo "[4/8] Running PAB..."
python inference.py --mode pab $COMMON \
    --output data/bench_pab.mp4

# 5. AdaCache
echo ""
echo "[5/8] Running AdaCache..."
python inference.py --mode adacache $COMMON \
    --output data/bench_adacache.mp4

# 6. FastCache
echo ""
echo "[6/8] Running FastCache..."
python inference.py --mode fastcache $COMMON \
    --output data/bench_fastcache.mp4

# 7. HetCache-slow (Δ=0.05)
echo ""
echo "[7/8] Running HetCache-slow (Δ=0.05)..."
python inference.py --mode hetcache $COMMON \
    --cache-thresh 0.05 \
    --context-ratio 0.05 --margin-ratio 0.7 \
    --use-kmeans --kmeans-clusters 16 \
    --use-generative-ema --use-attention-interaction \
    --output data/bench_hetcache_slow.mp4

# 8. HetCache-fast (Δ=0.02)
echo ""
echo "[8/8] Running HetCache-fast (Δ=0.02)..."
python inference.py --mode hetcache $COMMON \
    --cache-thresh 0.02 \
    --context-ratio 0.05 --margin-ratio 0.7 \
    --use-kmeans --kmeans-clusters 16 \
    --use-generative-ema --use-attention-interaction \
    --output data/bench_hetcache_fast.mp4

echo ""
echo "============================================================"
echo "  Benchmark completed — $(date)"
echo "============================================================"
