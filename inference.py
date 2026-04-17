"""
HetCache Inference Script — Unified Video Editing Acceleration

Supports multiple acceleration methods for Wan2.1-VACE video inpainting/outpainting:
  - baseline:    No acceleration (full compute at every timestep)
  - teacache:    TeaCache timestep-level caching
  - hetcache:    HetCache heterogeneous token caching (ours)
  - pab:         Pyramid Attention Broadcast
  - adacache:    AdaCache adaptive caching
  - fastcache:   FastCache statistical caching

Combine any method with TeaCache by appending '_teacache' (e.g., pab_teacache).

Usage:
    # HetCache (recommended)
    python inference.py --mode hetcache --video data/real.mp4 --mask data/masks.mp4 \\
        --frames 33 --steps 50 --cache-thresh 0.05 --context-ratio 0.05 \\
        --margin-ratio 0.7 --use-kmeans --kmeans-clusters 16 \\
        --use-attention-interaction --apply-mask-to-input

    # Baseline
    python inference.py --mode baseline --video data/real.mp4 --mask data/masks.mp4 \\
        --frames 33 --steps 50 --apply-mask-to-input

    # TeaCache
    python inference.py --mode teacache --video data/real.mp4 --mask data/masks.mp4 \\
        --frames 33 --steps 50 --cache-thresh 0.05 --apply-mask-to-input

Paper: https://arxiv.org/abs/2603.24260
"""

import torch
import numpy as np
import time
import os
import argparse
import psutil
from pathlib import Path
from PIL import Image

import cv2

from diffsynth import save_video, VideoData
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig


# ---------------------------------------------------------------------------
#  Video I/O and Metrics
# ---------------------------------------------------------------------------

def read_video(video_path):
    """Read video frames from an .mp4 file into a numpy array."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.array(frames)


def compute_psnr(video1, video2):
    """Compute average PSNR between two videos (frame-by-frame)."""
    from skimage.metrics import peak_signal_noise_ratio as psnr
    assert len(video1) == len(video2), "Videos must have the same number of frames"
    return np.mean([psnr(f1, f2) for f1, f2 in zip(video1, video2)])


def compute_lpips_score(video1, video2, device="cuda"):
    """Compute average LPIPS (perceptual distance) between two videos."""
    import lpips
    assert len(video1) == len(video2), "Videos must have the same number of frames"
    loss_fn = lpips.LPIPS(net="alex").to(device)
    values = []
    for f1, f2 in zip(video1, video2):
        t1 = torch.from_numpy(f1).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0
        t2 = torch.from_numpy(f2).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0
        with torch.no_grad():
            values.append(loss_fn(t1.to(device), t2.to(device)).item())
    return np.mean(values)


def get_gpu_memory():
    """Return (current, peak) GPU memory usage in MB."""
    if torch.cuda.is_available():
        return (
            torch.cuda.memory_allocated() / 1024 ** 2,
            torch.cuda.max_memory_allocated() / 1024 ** 2,
        )
    return 0.0, 0.0


def get_cpu_memory():
    """Return current process RSS in MB."""
    return psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2


# ---------------------------------------------------------------------------
#  Compositing helpers
# ---------------------------------------------------------------------------

def composite_inpainted_video(generated_video, original_frames, mask_frames):
    """Composite generated (mask region) onto original (background) per-frame."""
    composited = []
    for gen, orig, mask in zip(generated_video, original_frames, mask_frames):
        gen_np = np.array(gen).astype(np.float32) / 255.0
        orig_np = np.array(orig).astype(np.float32) / 255.0
        m = np.array(mask.convert("L")).astype(np.float32) / 255.0
        m = m[:, :, np.newaxis]
        out = orig_np * (1 - m) + gen_np * m
        composited.append(Image.fromarray((out * 255).astype(np.uint8)))
    return composited


def apply_mask_to_video(video_frames, mask_frames, fill_color=(0, 0, 0)):
    """Zero-out (or fill) mask regions in the input video before inference."""
    color = np.array(fill_color, dtype=np.float32) / 255.0
    masked = []
    for vf, mf in zip(video_frames, mask_frames):
        v = np.array(vf).astype(np.float32) / 255.0
        m = np.array(mf.convert("L")).astype(np.float32) / 255.0
        m = m[:, :, np.newaxis]
        out = v * (1 - m) + color.reshape(1, 1, 3) * m
        masked.append(Image.fromarray((out * 255).astype(np.uint8)))
    return masked


# ---------------------------------------------------------------------------
#  Argument parsing
# ---------------------------------------------------------------------------

MODE_ALIASES = {
    "hetcache": "masked_teacache",
    "hetcache_slow": "masked_teacache",
    "hetcache_fast": "masked_teacache",
}

ALL_MODES = [
    "baseline",
    "teacache",
    "hetcache",           # alias for masked_teacache
    "masked_teacache",    # internal name
    "masked_token",
    "pab",
    "pab_teacache",
    "adacache",
    "adacache_teacache",
    "fastcache",
    "fastcache_teacache",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="HetCache — Unified Video Editing Acceleration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # HetCache (ours)
  python inference.py --mode hetcache --video data/real.mp4 --mask data/masks.mp4 \\
      --frames 33 --steps 50 --cache-thresh 0.05 --context-ratio 0.05 \\
      --margin-ratio 0.7 --use-kmeans --kmeans-clusters 16 --use-attention-interaction

  # Baseline (no acceleration)
  python inference.py --mode baseline --video data/real.mp4 --mask data/masks.mp4 --frames 33 --steps 50

  # TeaCache
  python inference.py --mode teacache --video data/real.mp4 --mask data/masks.mp4 --frames 33 --steps 50 --cache-thresh 0.05
""",
    )

    # --- Required ---
    parser.add_argument("--mode", type=str, required=True, choices=ALL_MODES,
                        help="Acceleration method to use")

    # --- Input / Output ---
    parser.add_argument("--video", type=str, default="data/real.mp4",
                        help="Path to input video")
    parser.add_argument("--mask", type=str, default="data/masks.mp4",
                        help="Path to mask video")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path (auto-generated if omitted)")
    parser.add_argument("--data-dir", type=str, default=".",
                        help="Base data directory (default: current dir)")

    # --- Video parameters ---
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--frames", type=int, default=81, help="Number of frames")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--steps", type=int, default=None,
                        help="Inference steps (default: 50)")

    # --- Prompts ---
    parser.add_argument("--prompt", type=str,
                        default="Inpaint the occluded region to blend naturally with the surroundings, producing a smooth and coherent video",
                        help="Positive prompt")
    parser.add_argument("--negative-prompt", type=str,
                        default="oversaturated colors, overexposed, static, blurry details, subtitles, painting style, artwork, still image, gray overall, worst quality, low quality, JPEG artifacts, ugly, deformed, extra fingers, poorly drawn hands, poorly drawn face, disfigured, malformed limbs, fused fingers, static frame, cluttered background, three legs, many people in background, walking backwards",
                        help="Negative prompt")

    # --- TeaCache ---
    parser.add_argument("--cache-thresh", type=float, default=None,
                        help="TeaCache L1 distance threshold (e.g. 0.05=slow, 0.02=fast)")
    parser.add_argument("--cache-model-id", type=str, default="Wan2.1-T2V-1.3B",
                        help="TeaCache polynomial model identifier")

    # --- HetCache (MaskedTokenCache) ---
    parser.add_argument("--context-ratio", type=float, default=0.05,
                        help="Context token sampling ratio (default: 0.05)")
    parser.add_argument("--margin-ratio", type=float, default=0.7,
                        help="Margin token sampling ratio (default: 0.7)")
    parser.add_argument("--ema-alpha", type=float, default=0.99,
                        help="EMA decay for cache update (default: 0.99)")
    parser.add_argument("--use-kmeans", action="store_true",
                        help="Enable K-Means clustering for context sampling")
    parser.add_argument("--kmeans-clusters", type=int, default=16,
                        help="Number of K-Means clusters (default: 16)")
    parser.add_argument("--use-generative-ema", action="store_true",
                        help="Use direct replacement for generative tokens (V2 update)")
    parser.add_argument("--use-attention-interaction", action="store_true",
                        help="Enable two-stage context sampling (K-Means + Attention)")

    # --- PAB (Pyramid Attention Broadcast) ---
    parser.add_argument("--pab-spatial-range", type=int, default=2)
    parser.add_argument("--pab-temporal-range", type=int, default=3)
    parser.add_argument("--pab-cross-range", type=int, default=5)
    parser.add_argument("--pab-spatial-threshold", type=int, nargs=2, default=[100, 800])
    parser.add_argument("--pab-temporal-threshold", type=int, nargs=2, default=[100, 800])
    parser.add_argument("--pab-cross-threshold", type=int, nargs=2, default=[100, 800])
    parser.add_argument("--pab-enable-mlp", action="store_true")

    # --- AdaCache ---
    parser.add_argument("--adacache-module", type=str, choices=["spatial", "cross_mlp", "both"], default="spatial")
    parser.add_argument("--adacache-blocks", type=int, nargs="+", default=None)
    parser.add_argument("--adacache-enable-moreg", action="store_true")

    # --- FastCache ---
    parser.add_argument("--fastcache-cache-threshold", type=float, default=0.05)
    parser.add_argument("--fastcache-motion-threshold", type=float, default=0.1)
    parser.add_argument("--fastcache-blocks", type=int, nargs="+", default=None)
    parser.add_argument("--fastcache-significance-level", type=float, default=0.05)

    # --- Video preprocessing ---
    parser.add_argument("--apply-mask-to-input", action="store_true",
                        help="Mask input video before inference (recommended)")
    parser.add_argument("--mask-fill-color", type=int, nargs=3, default=[0, 0, 0],
                        metavar=("R", "G", "B"),
                        help="Fill color for masked regions (default: 0 0 0)")

    # --- Misc ---
    parser.add_argument("--no-fp8", action="store_true",
                        help="Disable FP8 quantization (use BF16 instead)")

    args = parser.parse_args()

    # Resolve mode aliases
    internal_mode = MODE_ALIASES.get(args.mode, args.mode)

    # Auto-configure based on mode
    enable_pab = internal_mode in ("pab", "pab_teacache")
    enable_adacache = internal_mode in ("adacache", "adacache_teacache")
    enable_fastcache = internal_mode in ("fastcache", "fastcache_teacache")

    uses_teacache = internal_mode in (
        "teacache", "masked_teacache", "pab_teacache",
        "adacache_teacache", "fastcache_teacache",
    )

    if uses_teacache and args.cache_thresh is None:
        args.cache_thresh = 0.05  # default slow threshold

    if args.steps is None:
        args.steps = 50  # default 50 steps for all modes

    # Only enable HetCache token caching for masked_token / masked_teacache modes
    uses_hetcache = internal_mode in ("masked_token", "masked_teacache")
    if not uses_hetcache:
        args.context_ratio = 0.0
        args.margin_ratio = 0.0

    # Auto-generate output name
    if args.output is None:
        stem = Path(args.video).stem
        args.output = f"{stem}_inp_{args.mode}.mp4"

    # Store derived flags
    args._internal_mode = internal_mode
    args._enable_pab = enable_pab
    args._enable_adacache = enable_adacache
    args._enable_fastcache = enable_fastcache
    args._uses_teacache = uses_teacache

    return args


# ---------------------------------------------------------------------------
#  Pipeline loading
# ---------------------------------------------------------------------------

def load_pipeline(args):
    """Load the Wan2.1-VACE pipeline with configured quantization."""
    print("=" * 72)
    print("  HetCache — Video Editing Acceleration")
    print("=" * 72)
    print(f"  Mode          : {args.mode}")
    print(f"  Video         : {args.video}")
    print(f"  Mask          : {args.mask}")
    print(f"  Steps         : {args.steps}")
    print(f"  Frames        : {args.frames}")
    print(f"  Resolution    : {args.width}x{args.height}")

    # Print acceleration config
    accel_parts = []
    if args._uses_teacache:
        accel_parts.append(f"TeaCache(Δ={args.cache_thresh})")
    if args._internal_mode in ("masked_teacache", "masked_token"):
        accel_parts.append(
            f"HetCache(ctx={args.context_ratio}, mar={args.margin_ratio}, "
            f"kmeans={'on' if args.use_kmeans else 'off'}, "
            f"attn_interact={'on' if args.use_attention_interaction else 'off'})"
        )
    if args._enable_pab:
        accel_parts.append(f"PAB(s={args.pab_spatial_range},t={args.pab_temporal_range},c={args.pab_cross_range})")
    if args._enable_adacache:
        accel_parts.append(f"AdaCache(module={args.adacache_module})")
    if args._enable_fastcache:
        accel_parts.append(f"FastCache(thresh={args.fastcache_cache_threshold})")
    print(f"  Acceleration  : {' + '.join(accel_parts) if accel_parts else 'None (baseline)'}")
    print("=" * 72)

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()

    offload_dtype = torch.bfloat16 if args.no_fp8 else torch.float8_e4m3fn
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(
                model_id="Wan-AI/Wan2.1-VACE-1.3B",
                origin_file_pattern="diffusion_pytorch_model*.safetensors",
                offload_device="cpu",
                offload_dtype=offload_dtype,
            ),
            ModelConfig(
                model_id="Wan-AI/Wan2.1-VACE-1.3B",
                origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth",
                offload_device="cpu",
                offload_dtype=offload_dtype,
            ),
            ModelConfig(
                model_id="Wan-AI/Wan2.1-VACE-1.3B",
                origin_file_pattern="Wan2.1_VAE.pth",
                offload_device="cpu",
            ),
        ],
    )
    pipe.enable_vram_management()

    t1 = time.time()
    print(f"\n  Model loaded in {t1 - t0:.1f}s")
    return pipe, t0, t1


# ---------------------------------------------------------------------------
#  Data loading
# ---------------------------------------------------------------------------

def load_video_data(args):
    """Load input video and mask; optionally apply mask to input."""
    video_path = os.path.join(args.data_dir, args.video)
    mask_path = os.path.join(args.data_dir, args.mask)

    control_video = VideoData(video_path, height=args.height, width=args.width)
    control_mask = VideoData(mask_path, height=args.height, width=args.width)

    video_frames = [control_video[i] for i in range(args.frames)]
    mask_frames = [control_mask[i] for i in range(args.frames)]
    original_frames = [f.copy() for f in video_frames]

    if args.apply_mask_to_input:
        print("  Applying mask to input video...")
        video_frames = apply_mask_to_video(
            video_frames, mask_frames, fill_color=tuple(args.mask_fill_color)
        )

    print(f"  Loaded {len(video_frames)} frames at {args.width}x{args.height}")
    return video_frames, mask_frames, original_frames


# ---------------------------------------------------------------------------
#  Generation
# ---------------------------------------------------------------------------

def run_generation(pipe, args, video_frames, mask_frames):
    """Run the denoising loop with the selected acceleration method."""
    torch.cuda.reset_peak_memory_stats()
    t_start = time.time()

    gen_kwargs = {
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "vace_video": video_frames,
        "vace_video_mask": mask_frames,
        "seed": args.seed,
        "tiled": True,
        "height": args.height,
        "width": args.width,
        "num_frames": args.frames,
        "num_inference_steps": args.steps,
        "tea_cache_model_id": args.cache_model_id,
        # PAB
        "enable_pab": args._enable_pab,
        "pab_spatial_range": args.pab_spatial_range,
        "pab_temporal_range": args.pab_temporal_range,
        "pab_cross_range": args.pab_cross_range,
        "pab_spatial_threshold": args.pab_spatial_threshold,
        "pab_temporal_threshold": args.pab_temporal_threshold,
        "pab_cross_threshold": args.pab_cross_threshold,
        "pab_enable_mlp": args.pab_enable_mlp,
        # AdaCache
        "enable_adacache": args._enable_adacache,
        "adacache_module": args.adacache_module,
        "adacache_blocks": args.adacache_blocks,
        "adacache_enable_moreg": args.adacache_enable_moreg,
        # FastCache
        "enable_fastcache": args._enable_fastcache,
        "fastcache_cache_ratio_threshold": args.fastcache_cache_threshold,
        "fastcache_motion_threshold": args.fastcache_motion_threshold,
        "fastcache_blocks": args.fastcache_blocks,
        "fastcache_significance_level": args.fastcache_significance_level,
        # HetCache (MaskedTokenCache)
        "masked_token_context_ratio": args.context_ratio,
        "masked_token_margin_ratio": args.margin_ratio,
        "masked_token_ema_alpha": args.ema_alpha,
        "masked_token_use_kmeans": args.use_kmeans,
        "masked_token_kmeans_clusters": args.kmeans_clusters,
        "masked_token_use_generative_ema": args.use_generative_ema,
        "masked_token_generative_ema_alpha": 0.99,
        "masked_token_use_attention_guidance": False,
        "masked_token_use_attention_interaction": args.use_attention_interaction,
    }

    # Add TeaCache threshold if enabled
    if args._uses_teacache:
        gen_kwargs["tea_cache_l1_thresh"] = args.cache_thresh

    video = pipe(**gen_kwargs)

    t_end = time.time()
    _, gpu_peak = get_gpu_memory()
    cpu_mem = get_cpu_memory()

    print(f"\n  Inference time : {t_end - t_start:.2f}s")
    print(f"  Peak GPU mem   : {gpu_peak:.0f} MB")
    print(f"  FPS            : {len(video_frames) / (t_end - t_start):.2f}")

    return video, t_start, t_end, gpu_peak, cpu_mem


# ---------------------------------------------------------------------------
#  Save and evaluate
# ---------------------------------------------------------------------------

def save_and_evaluate(args, video, original_frames, mask_frames,
                      t_model_start, t_model_end, t_inf_start, t_inf_end,
                      gpu_peak, cpu_mem):
    """Composite, save, and compute quality metrics."""
    # Composite
    composited = composite_inpainted_video(video, original_frames, mask_frames)

    # Save
    output_path = os.path.join(args.data_dir, args.output)
    save_video(composited, output_path, fps=15, quality=5)
    print(f"\n  Output saved to: {output_path}")

    # Metrics (compare composited vs. original)
    gt_path = os.path.join(args.data_dir, args.video)
    v1 = read_video(gt_path)
    v2 = read_video(output_path)
    min_len = min(len(v1), len(v2))
    v1, v2 = v1[:min_len], v2[:min_len]
    if v1.shape != v2.shape:
        v2 = np.array([cv2.resize(f, (v1.shape[2], v1.shape[1])) for f in v2])

    # Quality metrics (optional — requires lpips and scikit-image)
    psnr_val, lpips_val = None, None
    try:
        psnr_val = compute_psnr(v1, v2)
        lpips_val = compute_lpips_score(v1, v2)
    except ImportError:
        pass

    # Summary
    print("\n" + "=" * 72)
    print("  RESULTS")
    print("=" * 72)
    print(f"  Mode           : {args.mode}")
    print(f"  Inference time : {t_inf_end - t_inf_start:.2f}s")
    print(f"  Model load     : {t_model_end - t_model_start:.1f}s")
    print(f"  Peak GPU       : {gpu_peak:.0f} MB")
    if psnr_val is not None:
        print(f"  PSNR           : {psnr_val:.2f} dB")
        print(f"  LPIPS          : {lpips_val:.4f}")
    else:
        print("  PSNR / LPIPS   : (install lpips & scikit-image for quality metrics)")
    print("=" * 72)

    return psnr_val, lpips_val


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Load model
    pipe, t_model_start, t_model_end = load_pipeline(args)

    # Load data
    video_frames, mask_frames, original_frames = load_video_data(args)

    # Run generation
    video, t_inf_start, t_inf_end, gpu_peak, cpu_mem = run_generation(
        pipe, args, video_frames, mask_frames
    )

    # Save and evaluate
    save_and_evaluate(
        args, video, original_frames, mask_frames,
        t_model_start, t_model_end, t_inf_start, t_inf_end,
        gpu_peak, cpu_mem,
    )


if __name__ == "__main__":
    main()
