#!/usr/bin/env python3
"""
Evaluation script for HetCache.

Subcommands
-----------
quality   — GT-based metrics: PSNR, SSIM, LPIPS, VFID  (requires GT videos)
vbench    — Reference-free metrics via VBench            (no GT needed)

Usage examples:

    # GT-based quality metrics (single pair — PSNR / SSIM / LPIPS only)
    python evaluation.py quality \
        --gt data/real.mp4 \
        --pred data/test_hetcache.mp4

    # GT-based quality metrics (batch — PSNR / SSIM / LPIPS + VFID)
    python evaluation.py quality \
        --gt-dir /path/to/gt_videos \
        --pred-dir /path/to/pred_videos \
        --method hetcache

    # Reference-free VBench metrics
    python evaluation.py vbench \
        --pred-dir /path/to/pred_videos \
        --method hetcache

Note on VFID frame alignment:
    GT videos are automatically **truncated to the first N frames** (matching
    the generated video length) before I3D feature extraction.  I3D temporal
    sampling defaults to 8 frames.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm


# ──────────────────────────────────────────────────────────────────────
#  Video I/O
# ──────────────────────────────────────────────────────────────────────

def read_video_frames(path, max_frames=None):
    """Read video frames as list of uint8 numpy arrays (H, W, 3) in RGB."""
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if max_frames is not None and len(frames) >= max_frames:
            break
    cap.release()
    if not frames:
        raise ValueError(f"No frames read from {path}")
    return frames


def align_frame_counts(gt_frames, pred_frames):
    """Truncate both lists to the shorter length."""
    n = min(len(gt_frames), len(pred_frames))
    return gt_frames[:n], pred_frames[:n]


# ──────────────────────────────────────────────────────────────────────
#  PSNR / SSIM
# ──────────────────────────────────────────────────────────────────────

def compute_psnr(img1, img2):
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return 10.0 * np.log10(255.0 ** 2 / mse)


def compute_ssim_frame(img1, img2):
    """Compute SSIM between two uint8 RGB images (simplified luminance-only)."""
    g1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY).astype(np.float64)
    g2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY).astype(np.float64)
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mu1, mu2 = g1.mean(), g2.mean()
    s1, s2, s12 = g1.var(), g2.var(), np.cov(g1.flat, g2.flat)[0, 1]
    ssim = ((2 * mu1 * mu2 + C1) * (2 * s12 + C2)) / (
        (mu1 ** 2 + mu2 ** 2 + C1) * (s1 + s2 + C2)
    )
    return ssim


# ──────────────────────────────────────────────────────────────────────
#  LPIPS
# ──────────────────────────────────────────────────────────────────────

_lpips_model = None


def get_lpips_model(device="cuda"):
    global _lpips_model
    if _lpips_model is None:
        import lpips
        _lpips_model = lpips.LPIPS(net="alex").to(device)
    return _lpips_model


def compute_lpips_frame(img1, img2, device="cuda"):
    """Compute LPIPS between two uint8 RGB images."""
    model = get_lpips_model(device)

    def to_tensor(img):
        t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0
        return t.to(device)

    with torch.no_grad():
        return model(to_tensor(img1), to_tensor(img2)).item()


# ──────────────────────────────────────────────────────────────────────
#  VFID  (Video Fréchet Inception Distance via I3D)
# ──────────────────────────────────────────────────────────────────────

def load_i3d_model(checkpoint, device="cuda"):
    """Load pretrained I3D model for feature extraction."""
    from i3d_model import InceptionI3d

    model = InceptionI3d(400, in_channels=3)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model = model.to(device).eval()
    return model


def extract_i3d_features(model, video_path, device="cuda",
                         target_frames=8, max_frames=None):
    """
    Extract 1024-d I3D features from a video.

    Args:
        target_frames: Number of frames to feed I3D (temporal sampling).
                       Default 8 for short generated clips.
        max_frames: Truncate video to first N frames BEFORE sampling.
                    Use this to align GT temporal range with generated videos.
    """
    frames = read_video_frames(video_path, max_frames=max_frames)

    # Temporal sampling / padding to target_frames
    if len(frames) < target_frames:
        frames = frames + [frames[-1]] * (target_frames - len(frames))
    elif len(frames) > target_frames:
        indices = np.linspace(0, len(frames) - 1, target_frames, dtype=int)
        frames = [frames[i] for i in indices]

    # Resize to 224x224
    frames = [cv2.resize(f, (224, 224)) for f in frames]

    # (T, H, W, C) -> (1, C, T, H, W), normalise to [-1, 1]
    tensor = np.stack(frames)
    tensor = torch.from_numpy(tensor).permute(0, 3, 1, 2).unsqueeze(0).float()
    tensor = tensor.permute(0, 2, 1, 3, 4).to(device) / 127.5 - 1.0

    with torch.no_grad():
        features = model.extract_features(tensor)

    return features.cpu().numpy().squeeze()  # (1024,)


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Compute FID between two multivariate Gaussians."""
    from scipy import linalg

    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            raise ValueError(f"Imaginary component {np.max(np.abs(covmean.imag))}")
        covmean = covmean.real
    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)


def compute_vfid(gt_features, pred_features):
    """Compute VFID from (N, 1024) feature arrays."""
    mu_gt = np.mean(gt_features, axis=0)
    sigma_gt = np.cov(gt_features, rowvar=False)
    mu_pred = np.mean(pred_features, axis=0)
    sigma_pred = np.cov(pred_features, rowvar=False)
    return calculate_frechet_distance(mu_gt, sigma_gt, mu_pred, sigma_pred)


# ──────────────────────────────────────────────────────────────────────
#  Helpers: discover video pairs
# ──────────────────────────────────────────────────────────────────────

def discover_video_pairs(gt_dir, pred_dir, method):
    """Find (gt_path, pred_path) pairs by matching filenames."""
    gt_dir, pred_dir = Path(gt_dir), Path(pred_dir)

    # Try method-suffixed names first, then plain names
    pred_videos = sorted(pred_dir.glob(f"*_{method}.mp4"))
    if not pred_videos:
        pred_videos = sorted(pred_dir.glob("*.mp4"))

    pairs = []
    for pred_path in pred_videos:
        name = pred_path.stem
        base_name = name.replace(f"_{method}", "") if f"_{method}" in name else name

        gt_path = None
        for pattern in [f"{base_name}.mp4", f"{base_name}_raw_video.mp4"]:
            candidate = gt_dir / pattern
            if candidate.exists():
                gt_path = candidate
                break

        if gt_path is not None:
            pairs.append((gt_path, pred_path))

    return pairs


# ──────────────────────────────────────────────────────────────────────
#  CLI: quality  (GT-based: PSNR / SSIM / LPIPS / VFID)
# ──────────────────────────────────────────────────────────────────────

def cmd_quality(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Single-pair mode ──
    if args.gt and args.pred:
        gt_frames = read_video_frames(args.gt)
        pred_frames = read_video_frames(args.pred)
        gt_frames, pred_frames = align_frame_counts(gt_frames, pred_frames)
        n = len(gt_frames)

        psnr_vals, ssim_vals, lpips_vals = [], [], []
        for i in tqdm(range(n), desc="Per-frame metrics"):
            psnr_vals.append(compute_psnr(gt_frames[i], pred_frames[i]))
            ssim_vals.append(compute_ssim_frame(gt_frames[i], pred_frames[i]))
            lpips_vals.append(compute_lpips_frame(gt_frames[i], pred_frames[i], device))

        results = {
            "gt": str(args.gt),
            "pred": str(args.pred),
            "num_frames": n,
            "psnr": float(np.mean(psnr_vals)),
            "ssim": float(np.mean(ssim_vals)),
            "lpips": float(np.mean(lpips_vals)),
        }

        print(f"\n{'='*60}")
        print(f"  PSNR  : {results['psnr']:.2f} dB")
        print(f"  SSIM  : {results['ssim']:.4f}")
        print(f"  LPIPS : {results['lpips']:.4f}")
        print(f"  Frames: {n}")
        print(f"{'='*60}")

        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            print(f"Saved to {args.output}")
        return results

    # ── Batch mode (directories) ──
    if not (args.gt_dir and args.pred_dir):
        print("Error: provide either --gt/--pred or --gt-dir/--pred-dir")
        sys.exit(1)

    pairs = discover_video_pairs(args.gt_dir, args.pred_dir, args.method)
    if not pairs:
        print(f"No video pairs found in {args.pred_dir}")
        sys.exit(1)

    print(f"Found {len(pairs)} video pairs")

    # Load I3D for VFID
    i3d_model = None
    if args.i3d_checkpoint and Path(args.i3d_checkpoint).exists():
        print(f"Loading I3D from {args.i3d_checkpoint} ...")
        i3d_model = load_i3d_model(args.i3d_checkpoint, device)
    else:
        print("I3D checkpoint not found — skipping VFID. "
              "Download i3d_rgb_imagenet.pt from https://github.com/piergiaj/pytorch-i3d")

    all_psnr, all_ssim, all_lpips = [], [], []
    gt_feats, pred_feats = [], []

    for gt_path, pred_path in tqdm(pairs, desc="Evaluating"):
        # Per-frame metrics
        gt_frames = read_video_frames(str(gt_path))
        pred_frames = read_video_frames(str(pred_path))
        gt_frames, pred_frames = align_frame_counts(gt_frames, pred_frames)

        psnr_v, ssim_v, lpips_v = [], [], []
        for i in range(len(gt_frames)):
            psnr_v.append(compute_psnr(gt_frames[i], pred_frames[i]))
            ssim_v.append(compute_ssim_frame(gt_frames[i], pred_frames[i]))
            lpips_v.append(compute_lpips_frame(gt_frames[i], pred_frames[i], device))

        all_psnr.append(np.mean(psnr_v))
        all_ssim.append(np.mean(ssim_v))
        all_lpips.append(np.mean(lpips_v))

        # I3D features for VFID (truncate GT to generated length)
        if i3d_model is not None:
            pred_n = len(read_video_frames(str(pred_path)))
            pred_feats.append(extract_i3d_features(
                i3d_model, str(pred_path), device,
                target_frames=args.target_frames,
            ))
            gt_feats.append(extract_i3d_features(
                i3d_model, str(gt_path), device,
                target_frames=args.target_frames,
                max_frames=pred_n,
            ))

    results = {
        "method": args.method,
        "num_videos": len(pairs),
        "psnr": float(np.mean(all_psnr)),
        "ssim": float(np.mean(all_ssim)),
        "lpips": float(np.mean(all_lpips)),
    }

    if gt_feats and len(gt_feats) >= 2:
        vfid = compute_vfid(np.stack(gt_feats), np.stack(pred_feats))
        results["vfid"] = float(vfid)
        results["vfid_target_frames"] = args.target_frames

    print(f"\n{'='*60}")
    print(f"  Method: {args.method}  ({len(pairs)} videos)")
    print(f"  PSNR  : {results['psnr']:.2f} dB")
    print(f"  SSIM  : {results['ssim']:.4f}")
    print(f"  LPIPS : {results['lpips']:.4f}")
    if "vfid" in results:
        print(f"  VFID  : {results['vfid']:.4f}")
    print(f"{'='*60}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved to {args.output}")

    return results


# ──────────────────────────────────────────────────────────────────────
#  CLI: vbench  (reference-free metrics via VBench)
# ──────────────────────────────────────────────────────────────────────

VBENCH_DIMENSIONS = [
    "subject_consistency",
    "background_consistency",
    "motion_smoothness",
    "dynamic_degree",
    "aesthetic_quality",
    "imaging_quality",
]


def cmd_vbench(args):
    pred_dir = Path(args.pred_dir)

    # Discover videos
    pred_videos = sorted(pred_dir.glob(f"*_{args.method}.mp4"))
    if not pred_videos:
        pred_videos = sorted(pred_dir.glob("*.mp4"))
    if not pred_videos:
        print(f"No videos found in {pred_dir}")
        sys.exit(1)
    print(f"Found {len(pred_videos)} videos for method '{args.method}'")

    # Organize into a temp directory (VBench expects a flat folder)
    vbench_dir = pred_dir / f"_vbench_{args.method}"
    vbench_dir.mkdir(parents=True, exist_ok=True)
    for v in pred_videos:
        link = vbench_dir / v.name
        if link.exists():
            link.unlink()
        link.symlink_to(v.absolute())

    # VBench evaluate.py path
    vbench_script = args.vbench_script
    if not Path(vbench_script).exists():
        print(f"VBench evaluate.py not found at {vbench_script}")
        print("Install VBench: pip install vbench  or  git clone https://github.com/Vchitect/VBench")
        sys.exit(1)

    results_dir = Path(vbench_script).parent / "evaluation_results"
    dimensions = args.dimensions or VBENCH_DIMENSIONS

    all_scores = {}
    for dim in dimensions:
        print(f"\n  [{dim}]")
        existing_files = set(results_dir.glob("results_*_eval_results.json")) if results_dir.exists() else set()

        cmd = [
            "python", vbench_script,
            "--dimension", dim,
            "--videos_path", str(vbench_dir),
            "--mode=custom_input",
        ]
        if args.conda_env:
            cmd = ["conda", "run", "-n", args.conda_env] + cmd

        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if proc.returncode != 0:
                print(f"    Error: {proc.stderr[-300:]}")
                all_scores[dim] = None
                continue

            new_files = set(results_dir.glob("results_*_eval_results.json")) - existing_files
            if not new_files:
                print("    Warning: no result file produced")
                all_scores[dim] = None
                continue

            result_file = sorted(new_files, key=lambda p: p.stat().st_mtime)[-1]
            with open(result_file) as f:
                data = json.load(f)

            if dim in data and isinstance(data[dim], list) and data[dim]:
                score = float(data[dim][0])
                all_scores[dim] = score
                print(f"    Score: {score:.4f}")
            else:
                all_scores[dim] = None
                print("    Warning: could not parse score")

        except subprocess.TimeoutExpired:
            print("    Timeout")
            all_scores[dim] = None

    # Summary
    print(f"\n{'='*60}")
    print(f"  VBench — {args.method}  ({len(pred_videos)} videos)")
    for dim in dimensions:
        v = all_scores.get(dim)
        print(f"  {dim:<28s}: {v:.4f}" if v is not None else f"  {dim:<28s}: N/A")
    print(f"{'='*60}")

    if args.output:
        out = {"method": args.method, "num_videos": len(pred_videos), "scores": all_scores}
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Saved to {args.output}")


# ──────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="HetCache Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    # ── quality (GT-based) ──
    p_q = sub.add_parser("quality",
                         help="GT-based metrics: PSNR / SSIM / LPIPS / VFID")
    # Single-pair mode
    p_q.add_argument("--gt", type=str, default=None, help="Single GT video path")
    p_q.add_argument("--pred", type=str, default=None, help="Single predicted video path")
    # Batch mode
    p_q.add_argument("--gt-dir", type=str, default=None, help="GT video directory")
    p_q.add_argument("--pred-dir", type=str, default=None, help="Predicted video directory")
    p_q.add_argument("--method", type=str, default="hetcache",
                     help="Method suffix in filenames (default: hetcache)")
    # VFID options
    p_q.add_argument("--i3d-checkpoint", type=str, default="i3d_rgb_imagenet.pt",
                     help="Path to I3D pretrained weights")
    p_q.add_argument("--target-frames", type=int, default=8,
                     help="Number of frames sampled for I3D (default: 8)")
    p_q.add_argument("--output", type=str, default=None, help="Output JSON")

    # ── vbench (reference-free) ──
    p_v = sub.add_parser("vbench",
                         help="Reference-free metrics via VBench")
    p_v.add_argument("--pred-dir", type=str, required=True,
                     help="Directory of predicted videos")
    p_v.add_argument("--method", type=str, default="hetcache",
                     help="Method suffix in filenames")
    p_v.add_argument("--vbench-script", type=str, default="VBench/evaluate.py",
                     help="Path to VBench evaluate.py")
    p_v.add_argument("--conda-env", type=str, default=None,
                     help="Conda env name to run VBench in (e.g., vbench)")
    p_v.add_argument("--dimensions", nargs="+", default=None,
                     help=f"VBench dimensions (default: {VBENCH_DIMENSIONS})")
    p_v.add_argument("--output", type=str, default=None, help="Output JSON")

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "quality":
        cmd_quality(args)
    elif args.command == "vbench":
        cmd_vbench(args)


if __name__ == "__main__":
    main()
