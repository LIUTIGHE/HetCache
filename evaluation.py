#!/usr/bin/env python3
"""
Evaluation script for HetCache.

Computes quality metrics between generated videos and ground-truth:
  - PSNR  (per-frame, higher is better)
  - SSIM  (per-frame, higher is better)
  - LPIPS (per-frame, lower is better)
  - VFID  (distribution-level via I3D, lower is better)

Usage:
    # Per-video metrics (PSNR / SSIM / LPIPS)
    python evaluation.py pairwise \
        --gt data/real.mp4 \
        --pred data/test_hetcache.mp4

    # VFID across a set of videos
    python evaluation.py vfid \
        --gt-dir /path/to/gt_videos \
        --pred-dir /path/to/pred_videos \
        --method hetcache

Note on VFID frame alignment:
    VFID requires I3D features from temporally aligned content.  When GT
    videos are longer than generated videos, this script **truncates GT to
    the first N frames** (matching the generated length) before I3D feature
    extraction.  Previous evaluation code did not perform this truncation,
    leading to inflated absolute VFID values.  See README for details.
"""

import argparse
import json
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
                         target_frames=16, max_frames=None):
    """
    Extract 1024-d I3D features from a video.

    Args:
        target_frames: Number of frames to feed I3D (temporal sampling).
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

    # Resize to 224×224
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
#  CLI: pairwise  (single GT-pred pair → PSNR / SSIM / LPIPS)
# ──────────────────────────────────────────────────────────────────────

def cmd_pairwise(args):
    gt_frames = read_video_frames(args.gt)
    pred_frames = read_video_frames(args.pred)
    gt_frames, pred_frames = align_frame_counts(gt_frames, pred_frames)
    n = len(gt_frames)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    psnr_vals, ssim_vals, lpips_vals = [], [], []

    for i in tqdm(range(n), desc="Computing metrics"):
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


# ──────────────────────────────────────────────────────────────────────
#  CLI: vfid  (directory of videos → VFID)
# ──────────────────────────────────────────────────────────────────────

def cmd_vfid(args):
    gt_dir = Path(args.gt_dir)
    pred_dir = Path(args.pred_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Discover pred videos matching the method suffix
    pred_videos = sorted(pred_dir.glob(f"*_{args.method}.mp4"))
    if not pred_videos:
        # Try without suffix (direct name matching)
        pred_videos = sorted(pred_dir.glob("*.mp4"))

    if not pred_videos:
        print(f"No videos found in {pred_dir}")
        sys.exit(1)

    print(f"Found {len(pred_videos)} predicted videos")

    # Load I3D
    model = load_i3d_model(args.i3d_checkpoint, device)

    gt_feats, pred_feats = [], []
    for pred_path in tqdm(pred_videos, desc="Extracting I3D features"):
        # Determine GT path
        name = pred_path.stem
        if f"_{args.method}" in name:
            base_name = name.replace(f"_{args.method}", "")
        else:
            base_name = name

        # Try common GT naming patterns
        gt_path = None
        for pattern in [f"{base_name}.mp4", f"{base_name}_raw_video.mp4"]:
            candidate = gt_dir / pattern
            if candidate.exists():
                gt_path = candidate
                break
        if gt_path is None:
            print(f"  Warning: GT not found for {name}, skipping")
            continue

        # Get generated frame count to truncate GT
        pred_n = len(read_video_frames(str(pred_path)))

        pred_feat = extract_i3d_features(
            model, str(pred_path), device,
            target_frames=args.target_frames,
        )
        gt_feat = extract_i3d_features(
            model, str(gt_path), device,
            target_frames=args.target_frames,
            max_frames=pred_n,  # truncate GT to match generated length
        )
        pred_feats.append(pred_feat)
        gt_feats.append(gt_feat)

    if len(gt_feats) < 2:
        print("Not enough videos for VFID (need >= 2)")
        sys.exit(1)

    gt_feats = np.stack(gt_feats)
    pred_feats = np.stack(pred_feats)

    vfid = compute_vfid(gt_feats, pred_feats)

    print(f"\n{'='*60}")
    print(f"  VFID  : {vfid:.4f}")
    print(f"  Videos: {len(gt_feats)}")
    print(f"  Frames: {args.target_frames} (GT truncated to generated length)")
    print(f"{'='*60}")

    if args.output:
        results = {
            "method": args.method,
            "vfid": float(vfid),
            "num_videos": len(gt_feats),
            "target_frames": args.target_frames,
        }
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved to {args.output}")


# ──────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="HetCache Evaluation — PSNR / SSIM / LPIPS / VFID",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    # -- pairwise --
    p_pair = sub.add_parser("pairwise", help="Per-video PSNR / SSIM / LPIPS")
    p_pair.add_argument("--gt", type=str, required=True, help="Ground-truth video")
    p_pair.add_argument("--pred", type=str, required=True, help="Predicted video")
    p_pair.add_argument("--output", type=str, default=None, help="Output JSON")

    # -- vfid --
    p_vfid = sub.add_parser("vfid", help="VFID across a directory of videos")
    p_vfid.add_argument("--gt-dir", type=str, required=True, help="GT video directory")
    p_vfid.add_argument("--pred-dir", type=str, required=True, help="Predicted video directory")
    p_vfid.add_argument("--method", type=str, default="hetcache",
                        help="Method suffix in filenames (e.g., 'hetcache', 'baseline-50')")
    p_vfid.add_argument("--i3d-checkpoint", type=str, default="i3d_rgb_imagenet.pt",
                        help="Path to I3D pretrained weights")
    p_vfid.add_argument("--target-frames", type=int, default=16,
                        help="Number of frames for I3D feature extraction")
    p_vfid.add_argument("--output", type=str, default=None, help="Output JSON")

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "pairwise":
        cmd_pairwise(args)
    elif args.command == "vfid":
        cmd_vfid(args)


if __name__ == "__main__":
    main()
