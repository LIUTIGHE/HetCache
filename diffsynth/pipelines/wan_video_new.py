import torch, warnings, glob, os, types
import numpy as np
from PIL import Image
from einops import repeat, reduce, rearrange
from typing import Optional, Union
from typing_extensions import Literal
from dataclasses import dataclass
from tqdm import tqdm

from ..utils import BasePipeline, ModelConfig, PipelineUnit, PipelineUnitRunner
from ..models import ModelManager, load_state_dict
from ..models.wan_video_dit import WanModel, RMSNorm, sinusoidal_embedding_1d
from ..models.wan_video_text_encoder import WanTextEncoder, T5RelativeEmbedding, T5LayerNorm
from ..models.wan_video_vae import WanVideoVAE, RMS_norm, CausalConv3d, Upsample
from ..models.wan_video_image_encoder import WanImageEncoder
from ..models.wan_video_vace import VaceWanModel
from ..models.wan_video_motion_controller import WanMotionControllerModel
from ..schedulers.flow_match import FlowMatchScheduler
from ..prompters import WanPrompter
from ..vram_management import enable_vram_management, AutoWrappedModule, AutoWrappedLinear, WanAutoCastLayerNorm
from ..lora import GeneralLoRALoader

# Import PAB (Pyramid Attention Broadcast)
try:
    from re_PAB_mgr import PABConfig, set_pab_manager, update_steps
    PAB_AVAILABLE = True
except ImportError:
    PAB_AVAILABLE = False
    print("Warning: PAB module not found. PAB acceleration will be disabled.")

# Import AdaCache
try:
    from ..models.adacache_wrapper import (
        initialize_adacache,
        get_adacache_wrapper,
        get_adacache_state,
        is_adacache_enabled,
        reset_adacache,
        finalize_adacache,
        block_forward_with_adacache,
        AdaCacheConfig,
    )
    ADACACHE_AVAILABLE = True
except ImportError:
    ADACACHE_AVAILABLE = False

# Import FastCache
try:
    from ..models.fastcache_wrapper import (
        initialize_fastcache,
        get_fastcache_wrapper,
        get_fastcache_state,
        is_fastcache_enabled,
        reset_fastcache,
        finalize_fastcache,
        block_forward_with_fastcache,
        FastCacheConfig,
    )
    FASTCACHE_AVAILABLE = True
except ImportError:
    FASTCACHE_AVAILABLE = False
    print("Warning: AdaCache module not found. AdaCache acceleration will be disabled.")

# from .masked_region_accel_integration import MaskedRegionAccelerator, WanVideoUnit_MaskedRegionAccel

class WanVideoPipeline(BasePipeline):

    def __init__(self, device="cuda", torch_dtype=torch.bfloat16, tokenizer_path=None):
        super().__init__(
            device=device, torch_dtype=torch_dtype,
            height_division_factor=16, width_division_factor=16, time_division_factor=4, time_division_remainder=1
        )
        self.scheduler = FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)
        self.prompter = WanPrompter(tokenizer_path=tokenizer_path)
        self.text_encoder: WanTextEncoder = None
        self.image_encoder: WanImageEncoder = None
        self.dit: WanModel = None
        self.dit2: WanModel = None
        self.vae: WanVideoVAE = None
        self.motion_controller: WanMotionControllerModel = None
        self.vace: VaceWanModel = None
        self.in_iteration_models = ("dit", "motion_controller", "vace")
        self.in_iteration_models_2 = ("dit2", "motion_controller", "vace")
        self.unit_runner = PipelineUnitRunner()
        self.units = [
            WanVideoUnit_ShapeChecker(),
            WanVideoUnit_NoiseInitializer(),
            WanVideoUnit_InputVideoEmbedder(),
            WanVideoUnit_PromptEmbedder(),
            WanVideoUnit_ImageEmbedderVAE(),
            WanVideoUnit_ImageEmbedderCLIP(),
            WanVideoUnit_ImageEmbedderFused(),
            WanVideoUnit_FunControl(),
            WanVideoUnit_FunReference(),
            WanVideoUnit_FunCameraControl(),
            WanVideoUnit_SpeedControl(),
            WanVideoUnit_VACE(),
            # WanVideoUnit_MaskedRegionAccel(),  # (optional: masked region accelerator)
            WanVideoUnit_UnifiedSequenceParallel(),
            WanVideoUnit_TeaCache(),
            WanVideoUnit_TokenTeaCache(),
            WanVideoUnit_MaskedTokenCache(),  # 🚀 Masked token cache
            WanVideoUnit_CfgMerger(),
        ]
        self.model_fn = model_fn_wan_video
        
    
    def load_lora(self, module, path, alpha=1):
        loader = GeneralLoRALoader(torch_dtype=self.torch_dtype, device=self.device)
        lora = load_state_dict(path, torch_dtype=self.torch_dtype, device=self.device)
        loader.load(module, lora, alpha=alpha)

        
    def training_loss(self, **inputs):
        max_timestep_boundary = int(inputs.get("max_timestep_boundary", 1) * self.scheduler.num_train_timesteps)
        min_timestep_boundary = int(inputs.get("min_timestep_boundary", 0) * self.scheduler.num_train_timesteps)
        timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
        timestep = self.scheduler.timesteps[timestep_id].to(dtype=self.torch_dtype, device=self.device)
        
        inputs["latents"] = self.scheduler.add_noise(inputs["input_latents"], inputs["noise"], timestep)
        training_target = self.scheduler.training_target(inputs["input_latents"], inputs["noise"], timestep)
        
        noise_pred = self.model_fn(**inputs, timestep=timestep)
        
        loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
        loss = loss * self.scheduler.training_weight(timestep)
        return loss


    def enable_vram_management(self, num_persistent_param_in_dit=None, vram_limit=None, vram_buffer=0.5):
        self.vram_management_enabled = True
        if num_persistent_param_in_dit is not None:
            vram_limit = None
        else:
            if vram_limit is None:
                vram_limit = self.get_vram()
            vram_limit = vram_limit - vram_buffer
        if self.text_encoder is not None:
            dtype = next(iter(self.text_encoder.parameters())).dtype
            enable_vram_management(
                self.text_encoder,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Embedding: AutoWrappedModule,
                    T5RelativeEmbedding: AutoWrappedModule,
                    T5LayerNorm: AutoWrappedModule,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                vram_limit=vram_limit,
            )
        if self.dit is not None:
            dtype = next(iter(self.dit.parameters())).dtype
            device = "cpu" if vram_limit is not None else self.device
            enable_vram_management(
                self.dit,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv3d: AutoWrappedModule,
                    torch.nn.LayerNorm: WanAutoCastLayerNorm,
                    RMSNorm: AutoWrappedModule,
                    torch.nn.Conv2d: AutoWrappedModule,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device=device,
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                max_num_param=num_persistent_param_in_dit,
                overflow_module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                vram_limit=vram_limit,
            )
        if self.dit2 is not None:
            dtype = next(iter(self.dit2.parameters())).dtype
            device = "cpu" if vram_limit is not None else self.device
            enable_vram_management(
                self.dit2,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv3d: AutoWrappedModule,
                    torch.nn.LayerNorm: WanAutoCastLayerNorm,
                    RMSNorm: AutoWrappedModule,
                    torch.nn.Conv2d: AutoWrappedModule,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device=device,
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                max_num_param=num_persistent_param_in_dit,
                overflow_module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                vram_limit=vram_limit,
            )
        if self.vae is not None:
            dtype = next(iter(self.vae.parameters())).dtype
            enable_vram_management(
                self.vae,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv2d: AutoWrappedModule,
                    RMS_norm: AutoWrappedModule,
                    CausalConv3d: AutoWrappedModule,
                    Upsample: AutoWrappedModule,
                    torch.nn.SiLU: AutoWrappedModule,
                    torch.nn.Dropout: AutoWrappedModule,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device=self.device,
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
            )
        if self.image_encoder is not None:
            dtype = next(iter(self.image_encoder.parameters())).dtype
            enable_vram_management(
                self.image_encoder,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv2d: AutoWrappedModule,
                    torch.nn.LayerNorm: AutoWrappedModule,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=dtype,
                    computation_device=self.device,
                ),
            )
        if self.motion_controller is not None:
            dtype = next(iter(self.motion_controller.parameters())).dtype
            enable_vram_management(
                self.motion_controller,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=dtype,
                    computation_device=self.device,
                ),
            )
        if self.vace is not None:
            device = "cpu" if vram_limit is not None else self.device
            enable_vram_management(
                self.vace,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv3d: AutoWrappedModule,
                    torch.nn.LayerNorm: AutoWrappedModule,
                    RMSNorm: AutoWrappedModule,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device=device,
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                vram_limit=vram_limit,
            )
            
            
    def initialize_usp(self):
        import torch.distributed as dist
        from xfuser.core.distributed import initialize_model_parallel, init_distributed_environment
        dist.init_process_group(backend="nccl", init_method="env://")
        init_distributed_environment(rank=dist.get_rank(), world_size=dist.get_world_size())
        initialize_model_parallel(
            sequence_parallel_degree=dist.get_world_size(),
            ring_degree=1,
            ulysses_degree=dist.get_world_size(),
        )
        torch.cuda.set_device(dist.get_rank())
            
            
    def enable_usp(self):
        from xfuser.core.distributed import get_sequence_parallel_world_size
        from ..distributed.xdit_context_parallel import usp_attn_forward, usp_dit_forward

        for block in self.dit.blocks:
            block.self_attn.forward = types.MethodType(usp_attn_forward, block.self_attn)
        self.dit.forward = types.MethodType(usp_dit_forward, self.dit)
        if self.dit2 is not None:
            for block in self.dit2.blocks:
                block.self_attn.forward = types.MethodType(usp_attn_forward, block.self_attn)
            self.dit2.forward = types.MethodType(usp_dit_forward, self.dit2)
        self.sp_size = get_sequence_parallel_world_size()
        self.use_unified_sequence_parallel = True


    @staticmethod
    def from_pretrained(
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = "cuda",
        model_configs: list[ModelConfig] = [],
        tokenizer_config: ModelConfig = ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/*"),
        redirect_common_files: bool = True,
        use_usp=False,
    ):
        # Redirect model path
        if redirect_common_files:
            redirect_dict = {
                "models_t5_umt5-xxl-enc-bf16.pth": "Wan-AI/Wan2.1-T2V-1.3B",
                "Wan2.1_VAE.pth": "Wan-AI/Wan2.1-T2V-1.3B",
                "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth": "Wan-AI/Wan2.1-I2V-14B-480P",
            }
            for model_config in model_configs:
                if model_config.origin_file_pattern is None or model_config.model_id is None:
                    continue
                if model_config.origin_file_pattern in redirect_dict and model_config.model_id != redirect_dict[model_config.origin_file_pattern]:
                    print(f"To avoid repeatedly downloading model files, ({model_config.model_id}, {model_config.origin_file_pattern}) is redirected to ({redirect_dict[model_config.origin_file_pattern]}, {model_config.origin_file_pattern}). You can use `redirect_common_files=False` to disable file redirection.")
                    model_config.model_id = redirect_dict[model_config.origin_file_pattern]
        
        # Initialize pipeline
        pipe = WanVideoPipeline(device=device, torch_dtype=torch_dtype)
        if use_usp: pipe.initialize_usp()
        
        # Download and load models
        model_manager = ModelManager()
        for model_config in model_configs:
            model_config.download_if_necessary(use_usp=use_usp)
            model_manager.load_model(
                model_config.path,
                device=model_config.offload_device or device,
                torch_dtype=model_config.offload_dtype or torch_dtype
            )
        
        # Load models
        pipe.text_encoder = model_manager.fetch_model("wan_video_text_encoder")
        dit = model_manager.fetch_model("wan_video_dit", index=2)
        if isinstance(dit, list):
            pipe.dit, pipe.dit2 = dit
        else:
            pipe.dit = dit
        pipe.vae = model_manager.fetch_model("wan_video_vae")
        pipe.image_encoder = model_manager.fetch_model("wan_video_image_encoder")
        pipe.motion_controller = model_manager.fetch_model("wan_video_motion_controller")
        pipe.vace = model_manager.fetch_model("wan_video_vace")
        
        # Size division factor
        if pipe.vae is not None:
            pipe.height_division_factor = pipe.vae.upsampling_factor * 2
            pipe.width_division_factor = pipe.vae.upsampling_factor * 2

        # Initialize tokenizer
        tokenizer_config.download_if_necessary(use_usp=use_usp)
        pipe.prompter.fetch_models(pipe.text_encoder)
        pipe.prompter.fetch_tokenizer(tokenizer_config.path)
        
        # Unified Sequence Parallel
        if use_usp: pipe.enable_usp()
        return pipe


    @torch.no_grad()
    def __call__(
        self,
        # Prompt
        prompt: str,
        negative_prompt: Optional[str] = "",
        # Image-to-video
        input_image: Optional[Image.Image] = None,
        # First-last-frame-to-video
        end_image: Optional[Image.Image] = None,
        # Video-to-video
        input_video: Optional[list[Image.Image]] = None,
        denoising_strength: Optional[float] = 1.0,
        # ControlNet
        control_video: Optional[list[Image.Image]] = None,
        reference_image: Optional[Image.Image] = None,
        # Camera control
        camera_control_direction: Optional[Literal["Left", "Right", "Up", "Down", "LeftUp", "LeftDown", "RightUp", "RightDown"]] = None,
        camera_control_speed: Optional[float] = 1/54,
        camera_control_origin: Optional[tuple] = (0, 0.532139961, 0.946026558, 0.5, 0.5, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0),
        # VACE
        vace_video: Optional[list[Image.Image]] = None,
        vace_video_mask: Optional[Image.Image] = None,
        vace_reference_image: Optional[Image.Image] = None,
        vace_scale: Optional[float] = 1.0,
        # Randomness
        seed: Optional[int] = None,
        rand_device: Optional[str] = "cpu",
        # Shape
        height: Optional[int] = 480,
        width: Optional[int] = 832,
        num_frames=81,
        # Classifier-free guidance
        cfg_scale: Optional[float] = 5.0,
        cfg_merge: Optional[bool] = False,
        # Boundary
        switch_DiT_boundary: Optional[float] = 0.875,
        # Scheduler
        num_inference_steps: Optional[int] = 50,
        sigma_shift: Optional[float] = 5.0,
        # Speed control
        motion_bucket_id: Optional[int] = None,
        # VAE tiling
        tiled: Optional[bool] = True,
        tile_size: Optional[tuple[int, int]] = (30, 52),
        tile_stride: Optional[tuple[int, int]] = (15, 26),
        # Sliding window
        sliding_window_size: Optional[int] = None,
        sliding_window_stride: Optional[int] = None,
        # Teacache
        tea_cache_l1_thresh: Optional[float] = None,
        tea_cache_model_id: Optional[str] = "",
        # Token-level TeaCache (extended)
        token_cache_ratio: Optional[float] = None,
        token_similarity_threshold: Optional[float] = 0.95,
        # Masked Token Cache (HetCache)
        masked_token_context_ratio: Optional[float] = 0.0,
        masked_token_margin_ratio: Optional[float] = 0.5,
        masked_token_ema_alpha: Optional[float] = 0.99,
        masked_token_use_kmeans: Optional[bool] = False,  # Use K-Means for context sampling
        masked_token_kmeans_clusters: Optional[int] = 100,  # Number of K-Means clusters (10-200)
        masked_token_use_generative_ema: Optional[bool] = False,  # Use generative EMA
        masked_token_generative_ema_alpha: Optional[float] = 0.99,  # Generative EMA alpha
        masked_token_use_attention_guidance: Optional[bool] = False,  # [PHASE 1] Use attention-guided sampling
        masked_token_use_attention_interaction: Optional[bool] = False,  # [NEW] Use attention-interaction guided sampling
        # PAB (Pyramid Attention Broadcast)
        enable_pab: Optional[bool] = False,
        pab_spatial_range: Optional[int] = 2,
        pab_temporal_range: Optional[int] = 3,
        pab_cross_range: Optional[int] = 6,
        pab_spatial_threshold: Optional[list] = None,
        pab_temporal_threshold: Optional[list] = None,
        pab_cross_threshold: Optional[list] = None,
        pab_enable_mlp: Optional[bool] = False,
        # AdaCache (Adaptive Caching)
        enable_adacache: Optional[bool] = False,
        adacache_module: Optional[str] = 'spatial',  # 'spatial', 'cross_mlp', 'both'
        adacache_blocks: Optional[list] = None,  # Which blocks to cache
        adacache_codebook: Optional[dict] = None,  # Custom codebook
        adacache_enable_moreg: Optional[bool] = False,  # Motion regularization
        # FastCache (Hidden-state-level caching)
        enable_fastcache: Optional[bool] = False,
        fastcache_cache_ratio_threshold: Optional[float] = 0.05,  # Base cache threshold
        fastcache_motion_threshold: Optional[float] = 0.1,  # Motion detection threshold
        fastcache_blocks: Optional[list] = None,  # Which blocks to cache
        fastcache_significance_level: Optional[float] = 0.05,  # Statistical significance
        # 🔥 NEW: Visualization callback
        step_callback: Optional[callable] = None,  # Callback function called after each denoising step
        # progress_bar
        progress_bar_cmd=tqdm,
    ):
        # Initialize PAB if enabled
        if enable_pab and PAB_AVAILABLE:
            if pab_spatial_threshold is None:
                pab_spatial_threshold = [100, 800]
            if pab_temporal_threshold is None:
                pab_temporal_threshold = [100, 800]
            if pab_cross_threshold is None:
                pab_cross_threshold = [100, 800]
            
            pab_config = PABConfig(
                spatial_broadcast=True,
                spatial_threshold=pab_spatial_threshold,
                spatial_range=pab_spatial_range,
                temporal_broadcast=False,  # Wan model doesn't have separate temporal blocks like Latte
                temporal_threshold=pab_temporal_threshold,
                temporal_range=pab_temporal_range,
                cross_broadcast=True,
                cross_threshold=pab_cross_threshold,
                cross_range=pab_cross_range,
                mlp_broadcast=pab_enable_mlp,
                mlp_spatial_broadcast_config={} if not pab_enable_mlp else {
                    720: {"block": [0, 1, 2, 3, 4], "skip_count": 2},
                    640: {"block": [0, 1, 2, 3, 4], "skip_count": 2},
                    560: {"block": [0, 1, 2, 3, 4], "skip_count": 2},
                    480: {"block": [0, 1, 2, 3, 4], "skip_count": 2},
                    400: {"block": [0, 1, 2, 3, 4], "skip_count": 2},
                },
                mlp_temporal_broadcast_config={},
            )
            set_pab_manager(pab_config)
            update_steps(num_inference_steps)
            print(f"✅ PAB enabled: spatial_range={pab_spatial_range}, cross_range={pab_cross_range}, mlp={pab_enable_mlp}")
        elif enable_pab and not PAB_AVAILABLE:
            print("⚠️  Warning: PAB requested but re_PAB_mgr module not found. Continuing without PAB.")
        
        # Initialize AdaCache if enabled
        if enable_adacache and ADACACHE_AVAILABLE:
            if adacache_blocks is None:
                # Default: cache middle layers
                num_blocks = len(self.dit.blocks) if hasattr(self.dit, 'blocks') else 28
                adacache_blocks = [num_blocks // 2]  # Middle block
            
            adacache_config = AdaCacheConfig(
                enabled=True,
                cache_module=adacache_module,
                cache_blocks=adacache_blocks,
                codebook=adacache_codebook,
                num_steps=num_inference_steps,
                enable_moreg=adacache_enable_moreg,
                moreg_steps=(max(1, num_inference_steps // 10), num_inference_steps - num_inference_steps // 10),
                moreg_strides=[1],
                moreg_hyp=(0.385, 8, 1, 2),
                mograd_mul=10.0,
            )
            
            num_blocks = len(self.dit.blocks) if hasattr(self.dit, 'blocks') else 28
            initialize_adacache(adacache_config, num_blocks)
            print(f"✅ AdaCache enabled: {adacache_module} caching on blocks {adacache_blocks}")
        elif enable_adacache and not ADACACHE_AVAILABLE:
            print("⚠️  Warning: AdaCache requested but adacache_wrapper module not found. Continuing without AdaCache.")
        
        # Initialize FastCache if enabled
        if enable_fastcache and FASTCACHE_AVAILABLE:
            if fastcache_blocks is None:
                # Default: cache middle third of blocks
                num_blocks = len(self.dit.blocks) if hasattr(self.dit, 'blocks') else 28
                start = num_blocks // 3
                end = 2 * num_blocks // 3
                fastcache_blocks = list(range(start, end))
            
            fastcache_config = FastCacheConfig(
                enabled=True,
                cache_ratio_threshold=fastcache_cache_ratio_threshold,
                motion_threshold=fastcache_motion_threshold,
                cache_blocks=fastcache_blocks,
                significance_level=fastcache_significance_level,
            )
            
            num_blocks = len(self.dit.blocks) if hasattr(self.dit, 'blocks') else 28
            initialize_fastcache(fastcache_config, num_blocks)
            print(f"✅ FastCache enabled: cache_thresh={fastcache_cache_ratio_threshold}, motion_thresh={fastcache_motion_threshold}, blocks={fastcache_blocks}")
        elif enable_fastcache and not FASTCACHE_AVAILABLE:
            print("⚠️  Warning: FastCache requested but fastcache_wrapper module not found. Continuing without FastCache.")

        
        # Scheduler
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength=denoising_strength, shift=sigma_shift)
        
        # Inputs
        inputs_posi = {
            "prompt": prompt,
            "tea_cache_l1_thresh": tea_cache_l1_thresh, "tea_cache_model_id": tea_cache_model_id, 
            "num_inference_steps": num_inference_steps,
            "token_cache_ratio": token_cache_ratio, "token_similarity_threshold": token_similarity_threshold,
            "masked_token_context_ratio": masked_token_context_ratio, "masked_token_margin_ratio": masked_token_margin_ratio, "masked_token_ema_alpha": masked_token_ema_alpha, "masked_token_use_kmeans": masked_token_use_kmeans, "masked_token_kmeans_clusters": masked_token_kmeans_clusters, "masked_token_use_generative_ema": masked_token_use_generative_ema, "masked_token_generative_ema_alpha": masked_token_generative_ema_alpha, "masked_token_use_attention_guidance": masked_token_use_attention_guidance, "masked_token_use_attention_interaction": masked_token_use_attention_interaction,
        }
        inputs_nega = {
            "negative_prompt": negative_prompt,
            "tea_cache_l1_thresh": tea_cache_l1_thresh, "tea_cache_model_id": tea_cache_model_id, 
            "num_inference_steps": num_inference_steps,
            "token_cache_ratio": token_cache_ratio, "token_similarity_threshold": token_similarity_threshold,
            "masked_token_context_ratio": masked_token_context_ratio, "masked_token_margin_ratio": masked_token_margin_ratio, "masked_token_ema_alpha": masked_token_ema_alpha, "masked_token_use_kmeans": masked_token_use_kmeans, "masked_token_kmeans_clusters": masked_token_kmeans_clusters, "masked_token_use_generative_ema": masked_token_use_generative_ema, "masked_token_generative_ema_alpha": masked_token_generative_ema_alpha, "masked_token_use_attention_guidance": masked_token_use_attention_guidance, "masked_token_use_attention_interaction": masked_token_use_attention_interaction,
        }
        inputs_shared = {
            "input_image": input_image,
            "end_image": end_image,
            "input_video": input_video, "denoising_strength": denoising_strength,
            "control_video": control_video, "reference_image": reference_image,
            "camera_control_direction": camera_control_direction, "camera_control_speed": camera_control_speed, "camera_control_origin": camera_control_origin,
            "vace_video": vace_video, "vace_video_mask": vace_video_mask, "vace_reference_image": vace_reference_image, "vace_scale": vace_scale,
            # 🔥 NEW: Pass callback to shared inputs
            "step_callback": step_callback,
            # # Additional parameters
            # "enable_masked_region_accel": enable_masked_region_accel,
            # "mask_cache_interval": mask_cache_interval,
            # "mask_boundary_cache_interval": mask_boundary_cache_interval,
            # "mask_boundary_width": mask_boundary_width,
            "seed": seed, "rand_device": rand_device,
            "height": height, "width": width, "num_frames": num_frames,
            "cfg_scale": cfg_scale, "cfg_merge": cfg_merge,
            "sigma_shift": sigma_shift,
            "motion_bucket_id": motion_bucket_id,
            "tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride,
            "sliding_window_size": sliding_window_size, "sliding_window_stride": sliding_window_stride,
        }
        for unit in self.units:
            inputs_shared, inputs_posi, inputs_nega = self.unit_runner(unit, self, inputs_shared, inputs_posi, inputs_nega)

        # Denoise
        self.load_models_to_device(self.in_iteration_models)
        models = {name: getattr(self, name) for name in self.in_iteration_models}
        
        # 🔥 NEW: Hook for intermediate output collection
        step_callback = inputs_shared.get("step_callback", None)
        
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            # Reset token cache at start of each timestep
            if "token_tea_cache" in inputs_posi and inputs_posi["token_tea_cache"] is not None:
                inputs_posi["token_tea_cache"].reset_step()
            if "token_tea_cache" in inputs_nega and inputs_nega["token_tea_cache"] is not None:
                inputs_nega["token_tea_cache"].reset_step()
                
            # Switch DiT if necessary
            if timestep.item() < switch_DiT_boundary * self.scheduler.num_train_timesteps and self.dit2 is not None and models["dit"] is not self.dit2:
                self.load_models_to_device(self.in_iteration_models_2)
                models["dit"] = self.dit2
                
            # Timestep
            timestep = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)
            
            # 🔥 Enable attention return if callback is provided
            return_attn = step_callback is not None
            
            # Inference
            model_output_posi = self.model_fn(**models, **inputs_shared, **inputs_posi, timestep=timestep, all_timesteps=self.scheduler.timesteps, use_adacache=enable_adacache, use_fastcache=enable_fastcache, return_attention_weights=return_attn)
            
            # Unpack output based on return_attention_weights
            if return_attn and isinstance(model_output_posi, tuple):
                noise_pred_posi, attention_weights_posi = model_output_posi
            else:
                noise_pred_posi = model_output_posi
                attention_weights_posi = None
                
            if cfg_scale != 1.0:
                if cfg_merge:
                    noise_pred_posi, noise_pred_nega = noise_pred_posi.chunk(2, dim=0)
                else:
                    model_output_nega = self.model_fn(**models, **inputs_shared, **inputs_nega, timestep=timestep, all_timesteps=self.scheduler.timesteps, use_adacache=enable_adacache, use_fastcache=enable_fastcache, return_attention_weights=False)
                    noise_pred_nega = model_output_nega
                noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
            else:
                noise_pred = noise_pred_posi

            # Scheduler
            inputs_shared["latents"] = self.scheduler.step(noise_pred, self.scheduler.timesteps[progress_id], inputs_shared["latents"])
            if "first_frame_latents" in inputs_shared:
                inputs_shared["latents"][:, :, 0:1] = inputs_shared["first_frame_latents"]
            
            # 🔥 NEW: Call step callback to collect intermediate outputs
            if step_callback is not None:
                # Get attention map from model output or masked_token_cache
                attention_map = None
                token_classification = None
                
                # Priority 1: Use directly returned attention weights
                if return_attn and attention_weights_posi is not None:
                    attention_map = attention_weights_posi
                    # print(f"[DEBUG] Got attention from model output: {attention_map.shape}")
                # Priority 2: Use cached attention from masked_token_cache (correct attribute: cached_attention)
                elif "masked_token_cache" in inputs_posi and inputs_posi["masked_token_cache"] is not None:
                    mtc = inputs_posi["masked_token_cache"]
                    if hasattr(mtc, 'cached_attention') and mtc.cached_attention is not None:
                        attention_map = mtc.cached_attention
                        # print(f"[DEBUG] Got attention from cache: {attention_map.shape}")
                    if hasattr(mtc, 'token_classification') and mtc.token_classification is not None:
                        token_classification = mtc.token_classification
                        # print(f"[DEBUG] Got token classification: {token_classification.keys() if isinstance(token_classification, dict) else type(token_classification)}")
                else:
                    # print(f"[DEBUG] No attention available (return_attn={return_attn}, attention_weights_posi={attention_weights_posi is not None}, has_cache={inputs_posi.get('masked_token_cache') is not None})")
                
                step_callback(
                    step_idx=progress_id,
                    timestep=timestep.item(),
                    latent=inputs_shared["latents"],
                    attention_map=attention_map,
                    token_classification=token_classification,
                    metadata={
                        'noise_pred_norm': noise_pred.norm().item() if noise_pred is not None else None,
                    }
                )
        
        # VACE (TODO: remove it)
        if vace_reference_image is not None:
            inputs_shared["latents"] = inputs_shared["latents"][:, :, 1:]

        # Decode
        self.load_models_to_device(['vae'])
        video = self.vae.decode(inputs_shared["latents"], device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        video = self.vae_output_to_video(video)
        self.load_models_to_device([])

        # 📊 Print TeaCache statistics if enabled
        if "tea_cache" in inputs_posi and inputs_posi["tea_cache"] is not None:
            inputs_posi["tea_cache"].print_statistics()
        
        # 📊 Print AdaCache statistics if enabled
        if enable_adacache and ADACACHE_AVAILABLE:
            finalize_adacache()
            reset_adacache()  # Reset for next generation
        
        # 📊 Print FastCache statistics if enabled
        if enable_fastcache and FASTCACHE_AVAILABLE:
            finalize_fastcache()
            reset_fastcache()  # Reset for next generation

        return video



class WanVideoUnit_ShapeChecker(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=("height", "width", "num_frames"))

    def process(self, pipe: WanVideoPipeline, height, width, num_frames):
        height, width, num_frames = pipe.check_resize_height_width(height, width, num_frames)
        return {"height": height, "width": width, "num_frames": num_frames}



class WanVideoUnit_NoiseInitializer(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=("height", "width", "num_frames", "seed", "rand_device", "vace_reference_image"))

    def process(self, pipe: "WanVideoPipeline", height, width, num_frames, seed, rand_device, vace_reference_image):
        length = (num_frames - 1) // 4 + 1
        if vace_reference_image is not None:
            length += 1
        shape = (1, pipe.vae.model.z_dim, length, height // pipe.vae.upsampling_factor, width // pipe.vae.upsampling_factor)
        noise = pipe.generate_noise(shape, seed=seed, rand_device=rand_device)
        if vace_reference_image is not None:
            noise = torch.concat((noise[:, :, -1:], noise[:, :, :-1]), dim=2)
        return {"noise": noise}
    


class WanVideoUnit_InputVideoEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_video", "noise", "tiled", "tile_size", "tile_stride", "vace_reference_image"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: "WanVideoPipeline", input_video, noise, tiled, tile_size, tile_stride, vace_reference_image):
        if input_video is None:
            return {"latents": noise}
        pipe.load_models_to_device(["vae"])
        input_video = pipe.preprocess_video(input_video)
        input_latents = pipe.vae.encode(input_video, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
        if vace_reference_image is not None:
            vace_reference_image = pipe.preprocess_video([vace_reference_image])
            vace_reference_latents = pipe.vae.encode(vace_reference_image, device=pipe.device).to(dtype=pipe.torch_dtype, device=pipe.device)
            input_latents = torch.concat([vace_reference_latents, input_latents], dim=2)
        if pipe.scheduler.training:
            return {"latents": noise, "input_latents": input_latents}
        else:
            latents = pipe.scheduler.add_noise(input_latents, noise, timestep=pipe.scheduler.timesteps[0])
            return {"latents": latents}



class WanVideoUnit_PromptEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params_posi={"prompt": "prompt", "positive": "positive"},
            input_params_nega={"prompt": "negative_prompt", "positive": "positive"},
            onload_model_names=("text_encoder",)
        )

    def process(self, pipe: "WanVideoPipeline", prompt, positive) -> dict:
        pipe.load_models_to_device(self.onload_model_names)
        prompt_emb = pipe.prompter.encode_prompt(prompt, positive=positive, device=pipe.device)
        return {"context": prompt_emb}



class WanVideoUnit_ImageEmbedder(PipelineUnit):
    """
    Deprecated
    """
    def __init__(self):
        super().__init__(
            input_params=("input_image", "end_image", "num_frames", "height", "width", "tiled", "tile_size", "tile_stride"),
            onload_model_names=("image_encoder", "vae")
        )

    def process(self, pipe: "WanVideoPipeline", input_image, end_image, num_frames, height, width, tiled, tile_size, tile_stride):
        if input_image is None or pipe.image_encoder is None:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        image = pipe.preprocess_image(input_image.resize((width, height))).to(pipe.device)
        clip_context = pipe.image_encoder.encode_image([image])
        msk = torch.ones(1, num_frames, height//8, width//8, device=pipe.device)
        msk[:, 1:] = 0
        if end_image is not None:
            end_image = pipe.preprocess_image(end_image.resize((width, height))).to(pipe.device)
            vae_input = torch.concat([image.transpose(0,1), torch.zeros(3, num_frames-2, height, width).to(image.device), end_image.transpose(0,1)],dim=1)
            if pipe.dit.has_image_pos_emb:
                clip_context = torch.concat([clip_context, pipe.image_encoder.encode_image([end_image])], dim=1)
            msk[:, -1:] = 1
        else:
            vae_input = torch.concat([image.transpose(0, 1), torch.zeros(3, num_frames-1, height, width).to(image.device)], dim=1)

        msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, height//8, width//8)
        msk = msk.transpose(1, 2)[0]
        
        y = pipe.vae.encode([vae_input.to(dtype=pipe.torch_dtype, device=pipe.device)], device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)[0]
        y = y.to(dtype=pipe.torch_dtype, device=pipe.device)
        y = torch.concat([msk, y])
        y = y.unsqueeze(0)
        clip_context = clip_context.to(dtype=pipe.torch_dtype, device=pipe.device)
        y = y.to(dtype=pipe.torch_dtype, device=pipe.device)
        return {"clip_feature": clip_context, "y": y}



class WanVideoUnit_ImageEmbedderCLIP(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_image", "end_image", "height", "width"),
            onload_model_names=("image_encoder",)
        )

    def process(self, pipe: "WanVideoPipeline", input_image, end_image, height, width):
        if input_image is None or pipe.image_encoder is None or not pipe.dit.require_clip_embedding:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        image = pipe.preprocess_image(input_image.resize((width, height))).to(pipe.device)
        clip_context = pipe.image_encoder.encode_image([image])
        if end_image is not None:
            end_image = pipe.preprocess_image(end_image.resize((width, height))).to(pipe.device)
            if pipe.dit.has_image_pos_emb:
                clip_context = torch.concat([clip_context, pipe.image_encoder.encode_image([end_image])], dim=1)
        clip_context = clip_context.to(dtype=pipe.torch_dtype, device=pipe.device)
        return {"clip_feature": clip_context}
    


class WanVideoUnit_ImageEmbedderVAE(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_image", "end_image", "num_frames", "height", "width", "tiled", "tile_size", "tile_stride"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: "WanVideoPipeline", input_image, end_image, num_frames, height, width, tiled, tile_size, tile_stride):
        if input_image is None or not pipe.dit.require_vae_embedding:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        image = pipe.preprocess_image(input_image.resize((width, height))).to(pipe.device)
        msk = torch.ones(1, num_frames, height//8, width//8, device=pipe.device)
        msk[:, 1:] = 0
        if end_image is not None:
            end_image = pipe.preprocess_image(end_image.resize((width, height))).to(pipe.device)
            vae_input = torch.concat([image.transpose(0,1), torch.zeros(3, num_frames-2, height, width).to(image.device), end_image.transpose(0,1)],dim=1)
            msk[:, -1:] = 1
        else:
            vae_input = torch.concat([image.transpose(0, 1), torch.zeros(3, num_frames-1, height, width).to(image.device)], dim=1)

        msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, height//8, width//8)
        msk = msk.transpose(1, 2)[0]
        
        y = pipe.vae.encode([vae_input.to(dtype=pipe.torch_dtype, device=pipe.device)], device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)[0]
        y = y.to(dtype=pipe.torch_dtype, device=pipe.device)
        y = torch.concat([msk, y])
        y = y.unsqueeze(0)
        y = y.to(dtype=pipe.torch_dtype, device=pipe.device)
        return {"y": y}



class WanVideoUnit_ImageEmbedderFused(PipelineUnit):
    """
    Encode input image to latents using VAE. This unit is for Wan-AI/Wan2.2-TI2V-5B.
    """
    def __init__(self):
        super().__init__(
            input_params=("input_image", "latents", "height", "width", "tiled", "tile_size", "tile_stride"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: "WanVideoPipeline", input_image, latents, height, width, tiled, tile_size, tile_stride):
        if input_image is None or not pipe.dit.fuse_vae_embedding_in_latents:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        image = pipe.preprocess_image(input_image.resize((width, height))).transpose(0, 1)
        z = pipe.vae.encode([image], device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        latents[:, :, 0: 1] = z
        return {"latents": latents, "fuse_vae_embedding_in_latents": True, "first_frame_latents": z}



class WanVideoUnit_FunControl(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("control_video", "num_frames", "height", "width", "tiled", "tile_size", "tile_stride", "clip_feature", "y"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: "WanVideoPipeline", control_video, num_frames, height, width, tiled, tile_size, tile_stride, clip_feature, y):
        if control_video is None:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        control_video = pipe.preprocess_video(control_video)
        control_latents = pipe.vae.encode(control_video, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
        control_latents = control_latents.to(dtype=pipe.torch_dtype, device=pipe.device)
        if clip_feature is None or y is None:
            clip_feature = torch.zeros((1, 257, 1280), dtype=pipe.torch_dtype, device=pipe.device)
            y = torch.zeros((1, 16, (num_frames - 1) // 4 + 1, height//8, width//8), dtype=pipe.torch_dtype, device=pipe.device)
        else:
            y = y[:, -16:]
        y = torch.concat([control_latents, y], dim=1)
        return {"clip_feature": clip_feature, "y": y}
    


class WanVideoUnit_FunReference(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("reference_image", "height", "width", "reference_image"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: "WanVideoPipeline", reference_image, height, width):
        if reference_image is None:
            return {}
        pipe.load_models_to_device(["vae"])
        reference_image = reference_image.resize((width, height))
        reference_latents = pipe.preprocess_video([reference_image])
        reference_latents = pipe.vae.encode(reference_latents, device=pipe.device)
        clip_feature = pipe.preprocess_image(reference_image)
        clip_feature = pipe.image_encoder.encode_image([clip_feature])
        return {"reference_latents": reference_latents, "clip_feature": clip_feature}



class WanVideoUnit_FunCameraControl(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("height", "width", "num_frames", "camera_control_direction", "camera_control_speed", "camera_control_origin", "latents", "input_image"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: "WanVideoPipeline", height, width, num_frames, camera_control_direction, camera_control_speed, camera_control_origin, latents, input_image):
        if camera_control_direction is None:
            return {}
        camera_control_plucker_embedding = pipe.dit.control_adapter.process_camera_coordinates(
            camera_control_direction, num_frames, height, width, camera_control_speed, camera_control_origin)
        
        control_camera_video = camera_control_plucker_embedding[:num_frames].permute([3, 0, 1, 2]).unsqueeze(0)
        control_camera_latents = torch.concat(
            [
                torch.repeat_interleave(control_camera_video[:, :, 0:1], repeats=4, dim=2),
                control_camera_video[:, :, 1:]
            ], dim=2
        ).transpose(1, 2)
        b, f, c, h, w = control_camera_latents.shape
        control_camera_latents = control_camera_latents.contiguous().view(b, f // 4, 4, c, h, w).transpose(2, 3)
        control_camera_latents = control_camera_latents.contiguous().view(b, f // 4, c * 4, h, w).transpose(1, 2)
        control_camera_latents_input = control_camera_latents.to(device=pipe.device, dtype=pipe.torch_dtype)

        input_image = input_image.resize((width, height))
        input_latents = pipe.preprocess_video([input_image])
        pipe.load_models_to_device(self.onload_model_names)
        input_latents = pipe.vae.encode(input_latents, device=pipe.device)
        y = torch.zeros_like(latents).to(pipe.device)
        y[:, :, :1] = input_latents
        y = y.to(dtype=pipe.torch_dtype, device=pipe.device)
        return {"control_camera_latents_input": control_camera_latents_input, "y": y}



class WanVideoUnit_SpeedControl(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=("motion_bucket_id",))

    def process(self, pipe: "WanVideoPipeline", motion_bucket_id):
        if motion_bucket_id is None:
            return {}
        motion_bucket_id = torch.Tensor((motion_bucket_id,)).to(dtype=pipe.torch_dtype, device=pipe.device)
        return {"motion_bucket_id": motion_bucket_id}



class WanVideoUnit_VACE(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("vace_video", "vace_video_mask", "vace_reference_image", "vace_scale", "height", "width", "num_frames", "tiled", "tile_size", "tile_stride"),
            onload_model_names=("vae",)
        )

    def process(
        self,
        pipe: "WanVideoPipeline",
        vace_video, vace_video_mask, vace_reference_image, vace_scale,
        height, width, num_frames,
        tiled, tile_size, tile_stride
    ):
        if vace_video is not None or vace_video_mask is not None or vace_reference_image is not None:
            pipe.load_models_to_device(["vae"])
            if vace_video is None:
                vace_video = torch.zeros((1, 3, num_frames, height, width), dtype=pipe.torch_dtype, device=pipe.device)
            else:
                vace_video = pipe.preprocess_video(vace_video)
            
            if vace_video_mask is None:
                vace_video_mask = torch.ones_like(vace_video)
            else:
                vace_video_mask = pipe.preprocess_video(vace_video_mask, min_value=0, max_value=1)
            
            # print(f"[DEBUG] vace_video shape before VAE: {vace_video.shape}")
            # print(f"[DEBUG] vace_video_mask shape: {vace_video_mask.shape}")
            
            inactive = vace_video * (1 - vace_video_mask) + 0 * vace_video_mask
            reactive = vace_video * vace_video_mask + 0 * (1 - vace_video_mask)
            inactive = pipe.vae.encode(inactive, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
            reactive = pipe.vae.encode(reactive, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
            
            # print(f"[DEBUG] inactive latent shape after VAE: {inactive.shape}")
            # print(f"[DEBUG] reactive latent shape after VAE: {reactive.shape}")
            
            vace_video_latents = torch.concat((inactive, reactive), dim=1)
            
            vace_mask_latents = rearrange(vace_video_mask[0,0], "T (H P) (W Q) -> 1 (P Q) T H W", P=8, Q=8)
            vace_mask_latents = torch.nn.functional.interpolate(vace_mask_latents, size=((vace_mask_latents.shape[2] + 3) // 4, vace_mask_latents.shape[3], vace_mask_latents.shape[4]), mode='nearest-exact')
            
            if vace_reference_image is None:
                pass
            else:
                vace_reference_image = pipe.preprocess_video([vace_reference_image])
                vace_reference_latents = pipe.vae.encode(vace_reference_image, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
                vace_reference_latents = torch.concat((vace_reference_latents, torch.zeros_like(vace_reference_latents)), dim=1)
                vace_video_latents = torch.concat((vace_reference_latents, vace_video_latents), dim=2)
                vace_mask_latents = torch.concat((torch.zeros_like(vace_mask_latents[:, :, :1]), vace_mask_latents), dim=2)
            
            vace_context = torch.concat((vace_video_latents, vace_mask_latents), dim=1)
            return {"vace_context": vace_context, "vace_scale": vace_scale, "vace_mask_latents": vace_mask_latents}
        else:
            return {"vace_context": None, "vace_scale": vace_scale, "vace_mask_latents": None}



class WanVideoUnit_UnifiedSequenceParallel(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=())

    def process(self, pipe: "WanVideoPipeline"):
        if hasattr(pipe, "use_unified_sequence_parallel"):
            if pipe.use_unified_sequence_parallel:
                return {"use_unified_sequence_parallel": True}
        return {}



class WanVideoUnit_TeaCache(PipelineUnit):
    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params_posi={"num_inference_steps": "num_inference_steps", "tea_cache_l1_thresh": "tea_cache_l1_thresh", "tea_cache_model_id": "tea_cache_model_id"},
            input_params_nega={"num_inference_steps": "num_inference_steps", "tea_cache_l1_thresh": "tea_cache_l1_thresh", "tea_cache_model_id": "tea_cache_model_id"},
        )

    def process(self, pipe: "WanVideoPipeline", num_inference_steps, tea_cache_l1_thresh, tea_cache_model_id):
        if tea_cache_l1_thresh is None:
            return {}
        return {"tea_cache": TeaCache(num_inference_steps, rel_l1_thresh=tea_cache_l1_thresh, model_id=tea_cache_model_id)}


class WanVideoUnit_TokenTeaCache(PipelineUnit):
    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params_posi={
                "num_inference_steps": "num_inference_steps",
                "token_cache_ratio": "token_cache_ratio",
                "token_similarity_threshold": "token_similarity_threshold",
            },
            input_params_nega={
                "num_inference_steps": "num_inference_steps",
                "token_cache_ratio": "token_cache_ratio",
                "token_similarity_threshold": "token_similarity_threshold",
            },
        )

    def process(self, pipe: "WanVideoPipeline", num_inference_steps, token_cache_ratio, token_similarity_threshold):
        # Only enable TokenTeaCache if ratio is in valid range (0, 1)
        if token_cache_ratio is None or token_cache_ratio <= 0.0 or token_cache_ratio >= 1.0:
            return {}
        
        num_layers = len(pipe.dit.blocks) if pipe.dit is not None else 0
        if num_layers == 0:
            return {}
        
        return {
            "token_tea_cache": TokenTeaCache(
                num_inference_steps=num_inference_steps,
                num_layers=num_layers,
                token_cache_ratio=token_cache_ratio,
                similarity_threshold=token_similarity_threshold or 0.95,
                offload_to_cpu=False,  # Disable CPU offload to maintain speed
            )
        }


class WanVideoUnit_MaskedTokenCache(PipelineUnit):
    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params_posi={
                "num_inference_steps": "num_inference_steps",
                "masked_token_context_ratio": "masked_token_context_ratio",
                "masked_token_margin_ratio": "masked_token_margin_ratio",
                "masked_token_ema_alpha": "masked_token_ema_alpha",
                "masked_token_use_kmeans": "masked_token_use_kmeans",
                "masked_token_kmeans_clusters": "masked_token_kmeans_clusters",
                "masked_token_use_generative_ema": "masked_token_use_generative_ema",
                "masked_token_generative_ema_alpha": "masked_token_generative_ema_alpha",
                "masked_token_use_attention_guidance": "masked_token_use_attention_guidance",  # [PHASE 1]
                "masked_token_use_attention_interaction": "masked_token_use_attention_interaction",  # [NEW]
            },
            input_params_nega={
                "num_inference_steps": "num_inference_steps",
                "masked_token_context_ratio": "masked_token_context_ratio",
                "masked_token_margin_ratio": "masked_token_margin_ratio",
                "masked_token_ema_alpha": "masked_token_ema_alpha",
                "masked_token_use_kmeans": "masked_token_use_kmeans",
                "masked_token_kmeans_clusters": "masked_token_kmeans_clusters",
                "masked_token_use_generative_ema": "masked_token_use_generative_ema",
                "masked_token_generative_ema_alpha": "masked_token_generative_ema_alpha",
                "masked_token_use_attention_guidance": "masked_token_use_attention_guidance",  # [PHASE 1]
                "masked_token_use_attention_interaction": "masked_token_use_attention_interaction",  # [NEW]
            },
        )

    def process(self, pipe: "WanVideoPipeline", num_inference_steps, 
                masked_token_context_ratio, masked_token_margin_ratio, masked_token_ema_alpha,
                masked_token_use_kmeans, masked_token_kmeans_clusters,
                masked_token_use_generative_ema, masked_token_generative_ema_alpha,
                masked_token_use_attention_guidance, masked_token_use_attention_interaction):
        # Only enable MaskedTokenCache if context_ratio is in valid range (0, 1)
        if (masked_token_context_ratio is None or masked_token_context_ratio <= 0.0 or 
            masked_token_context_ratio >= 1.0):
            return {}
        
        # Get actual number of layers from model
        num_layers = len(pipe.dit.blocks) if pipe.dit is not None else 20
        
        # Determine model_id for TeaCache coefficients
        model_id = "Wan2.1-VACE-1.3B"  # Default for VACE model
        if hasattr(pipe, 'dit') and hasattr(pipe.dit, 'config'):
            # Try to extract from config
            config = pipe.dit.config
            if 'model_name' in config:
                model_id = config['model_name']
        
        return {
            "masked_token_cache": MaskedTokenCache(
                margin_pixels=5,  # Latent space margin (3 tokens ≈ 48 pixels)
                context_sample_ratio=masked_token_context_ratio,
                margin_sample_ratio=masked_token_margin_ratio or 0.7,
                ema_alpha=masked_token_ema_alpha or 0.99,
                warmup_steps=2,
                max_cached_layers=20,
                num_inference_steps=num_inference_steps,
                timestep_cache_threshold=0.1,  # TeaCache rel_l1_thresh
                model_id=model_id,
                use_kmeans_sampling=masked_token_use_kmeans or False,
                kmeans_n_clusters=masked_token_kmeans_clusters or 100,
                use_generative_ema=masked_token_use_generative_ema or False,
                generative_ema_alpha=masked_token_generative_ema_alpha or 0.99,
                use_attention_guidance=masked_token_use_attention_guidance or False,  # [PHASE 1]
                # [NEW] Attention-interaction guided context sampling (default: False for backward compatibility)
                use_attention_interaction=masked_token_use_attention_interaction or False,
                num_layers=num_layers,  # [FIX] Pass actual number of layers
            )
        }



class WanVideoUnit_CfgMerger(PipelineUnit):
    def __init__(self):
        super().__init__(take_over=True)
        self.concat_tensor_names = ["context", "clip_feature", "y", "reference_latents"]

    def process(self, pipe: "WanVideoPipeline", inputs_shared, inputs_posi, inputs_nega):
        if not inputs_shared["cfg_merge"]:
            return inputs_shared, inputs_posi, inputs_nega
        for name in self.concat_tensor_names:
            tensor_posi = inputs_posi.get(name)
            tensor_nega = inputs_nega.get(name)
            tensor_shared = inputs_shared.get(name)
            if tensor_posi is not None and tensor_nega is not None:
                inputs_shared[name] = torch.concat((tensor_posi, tensor_nega), dim=0)
            elif tensor_shared is not None:
                inputs_shared[name] = torch.concat((tensor_shared, tensor_shared), dim=0)
        inputs_posi.clear()
        inputs_nega.clear()
        return inputs_shared, inputs_posi, inputs_nega



class TeaCache:
    def __init__(self, num_inference_steps, rel_l1_thresh, model_id):
        self.num_inference_steps = num_inference_steps
        self.step = 0
        self.accumulated_rel_l1_distance = 0
        self.previous_modulated_input = None
        self.rel_l1_thresh = rel_l1_thresh
        self.previous_residual = None
        self.previous_hidden_states = None
        
        # 📊 Statistics tracking
        self.computed_steps = []
        self.skipped_steps = []
        self.step_distances = []  # Store accumulated distance at each step
        
        self.coefficients_dict = {
            "Wan2.1-T2V-1.3B": [-5.21862437e+04, 9.23041404e+03, -5.28275948e+02, 1.36987616e+01, -4.99875664e-02],
            "Wan2.1-T2V-14B": [-3.03318725e+05, 4.90537029e+04, -2.65530556e+03, 5.87365115e+01, -3.15583525e-01],
            "Wan2.1-I2V-14B-480P": [2.57151496e+05, -3.54229917e+04,  1.40286849e+03, -1.35890334e+01, 1.32517977e-01],
            "Wan2.1-I2V-14B-720P": [ 8.10705460e+03,  2.13393892e+03, -3.72934672e+02,  1.66203073e+01, -4.17769401e-02],
        }
        if model_id not in self.coefficients_dict:
            supported_model_ids = ", ".join([i for i in self.coefficients_dict])
            raise ValueError(f"{model_id} is not a supported TeaCache model id. Please choose a valid model id in ({supported_model_ids}).")
        self.coefficients = self.coefficients_dict[model_id]

    def check(self, dit: WanModel, x, t_mod):
        modulated_inp = t_mod.clone()
        if self.step == 0 or self.step == self.num_inference_steps - 1:
            should_calc = True
            self.accumulated_rel_l1_distance = 0
        else:
            coefficients = self.coefficients
            rescale_func = np.poly1d(coefficients)
            self.accumulated_rel_l1_distance += rescale_func(((modulated_inp-self.previous_modulated_input).abs().mean() / self.previous_modulated_input.abs().mean()).cpu().item())
            if self.accumulated_rel_l1_distance < self.rel_l1_thresh:
                should_calc = False
            else:
                should_calc = True
                self.accumulated_rel_l1_distance = 0
        
        # 📊 Record statistics
        self.step_distances.append(self.accumulated_rel_l1_distance)
        if should_calc:
            self.computed_steps.append(self.step)
            # print(f"[TeaCache] Step {self.step}: COMPUTE (distance={self.accumulated_rel_l1_distance:.6f}, thresh={self.rel_l1_thresh})")
        else:
            self.skipped_steps.append(self.step)
            # print(f"[TeaCache] Step {self.step}: SKIP (distance={self.accumulated_rel_l1_distance:.6f} < thresh={self.rel_l1_thresh})")
        
        self.previous_modulated_input = modulated_inp
        self.step += 1
        if self.step == self.num_inference_steps:
            self.step = 0
        if should_calc:
            self.previous_hidden_states = x.clone()
        return not should_calc

    def store(self, hidden_states):
        self.previous_residual = hidden_states - self.previous_hidden_states
        self.previous_hidden_states = None

    def update(self, hidden_states):
        hidden_states = hidden_states + self.previous_residual
        return hidden_states
    
    def print_statistics(self):
        """Print TeaCache statistics at the end of generation"""
        total_steps = self.num_inference_steps
        num_computed = len(self.computed_steps)
        num_skipped = len(self.skipped_steps)
        skip_rate = num_skipped / total_steps * 100 if total_steps > 0 else 0
        speedup = total_steps / num_computed if num_computed > 0 else 1.0
        
        print("\n" + "=" * 80)
        print("📊 TeaCache Statistics")
        print("=" * 80)
        print(f"Total steps: {total_steps}")
        print(f"Computed steps: {num_computed} - {sorted(self.computed_steps)}")
        print(f"Skipped steps: {num_skipped} - {sorted(self.skipped_steps)}")
        print(f"Skip rate: {skip_rate:.1f}%")
        print(f"Theoretical speedup: {speedup:.2f}x")
        print(f"Threshold: {self.rel_l1_thresh}")
        print("=" * 80)


class TokenTeaCache:
    """
    Token-level cache for DiT blocks. Performs lightweight token clustering to identify
    which tokens need recomputation vs. which can be retrieved from cache.
    
    Key features:
    - Lightweight cosine similarity based clustering
    - Maintains K/V cache per layer
    - Supports partial token computation
    - Dynamically determines compute budget based on token changes
    """
    
    def __init__(
        self,
        num_inference_steps: int,
        num_layers: int,
        token_cache_ratio: float = 0.5,  # Compute only 50% of tokens per step
        similarity_threshold: float = 0.95,  # Cosine similarity threshold
        warmup_steps: int = 2,  # Full compute for first N steps to build cache
        offload_to_cpu: bool = False,  # Whether to offload cache to CPU (trades speed for memory)
    ):
        self.num_inference_steps = num_inference_steps
        self.num_layers = num_layers
        self.token_cache_ratio = token_cache_ratio
        self.similarity_threshold = similarity_threshold
        self.warmup_steps = warmup_steps
        self.current_step = 0
        self.offload_to_cpu = offload_to_cpu  # CPU offload toggle
        
        # Per-layer cache: {layer_id: {"K": tensor, "V": tensor, "Y": tensor, "X_prev": tensor}}
        self.layer_caches = [{} for _ in range(num_layers)]
        
    def reset_step(self):
        """Reset step counter at the start of each denoising iteration."""
        self.current_step = 0
        # Clear all caches for new denoising process
        self.layer_caches = [{} for _ in range(self.num_layers)]
    
    def pick_tokens_to_compute(self, x_current: torch.Tensor, layer_id: int) -> tuple:
        """
        Determine which tokens need computation based on similarity to previous step.
        
        Args:
            x_current: Current input tensor [B, N, D]
            layer_id: Current layer index
            
        Returns:
            indices_compute: Indices of tokens to compute
            indices_skip: Indices of tokens to skip (use cache)
            should_use_cache: Whether to use caching for this step
        """
        # Warmup phase or no previous cache
        if self.current_step < self.warmup_steps or "X_prev" not in self.layer_caches[layer_id]:
            return None, None, False
        
        B, N, D = x_current.shape
        x_prev = self.layer_caches[layer_id]["X_prev"]
        
        # X_prev device handling: only move if offload_to_cpu is enabled
        if self.offload_to_cpu and x_prev.device != x_current.device:
            x_prev = x_prev.to(x_current.device)
        
        # Compute cosine similarity per token (lightweight)
        x_current_norm = torch.nn.functional.normalize(x_current, dim=-1)
        x_prev_norm = torch.nn.functional.normalize(x_prev, dim=-1)
        similarities = (x_current_norm * x_prev_norm).sum(dim=-1)  # [B, N]
        
        # Strategy: select the most changed tokens for recomputation
        num_compute = max(1, int(N * self.token_cache_ratio))
        
        # Find tokens with lowest similarity (highest change)
        _, indices_sorted = torch.sort(similarities, dim=1, descending=False)
        indices_compute = indices_sorted[:, :num_compute]  # [B, num_compute]
        indices_skip = indices_sorted[:, num_compute:]  # [B, N - num_compute]
        
        return indices_compute, indices_skip, True
    
    def get_cached_kv(self, layer_id: int, indices_skip: torch.Tensor) -> tuple:
        """
        Retrieve cached K, V for skipped tokens.
        
        Args:
            layer_id: Layer index
            indices_skip: Indices of tokens to skip [B, N_skip]
            
        Returns:
            K_cached, V_cached for skipped tokens
        """
        cache = self.layer_caches[layer_id]
        if "K" not in cache or "V" not in cache:
            return None, None

        if self.offload_to_cpu:
            # CPU offload mode: index on CPU then move to device
            indices_cpu = indices_skip.cpu()
            B = indices_cpu.shape[0]
            K_cached_list = []
            V_cached_list = []

            for b in range(B):
                K_cached_list.append(cache["K"][b, indices_cpu[b]])
                V_cached_list.append(cache["V"][b, indices_cpu[b]])

            # Stack on CPU then move to the requesting device
            K_cached = torch.stack(K_cached_list, dim=0)  # [B, N_skip, D]
            V_cached = torch.stack(V_cached_list, dim=0)
            return K_cached.to(indices_skip.device), V_cached.to(indices_skip.device)
        else:
            # GPU mode: direct indexing (faster)
            B = indices_skip.shape[0]
            K_cached_list = []
            V_cached_list = []
            
            for b in range(B):
                K_cached_list.append(cache["K"][b, indices_skip[b]])
                V_cached_list.append(cache["V"][b, indices_skip[b]])
            
            K_cached = torch.stack(K_cached_list, dim=0)  # [B, N_skip, D]
            V_cached = torch.stack(V_cached_list, dim=0)
            
            return K_cached, V_cached
    
    def get_cached_output(self, layer_id: int, indices_skip: torch.Tensor) -> torch.Tensor:
        """
        Retrieve cached output Y for skipped tokens.
        
        Args:
            layer_id: Layer index
            indices_skip: Indices of tokens to skip [B, N_skip]
            
        Returns:
            Y_cached for skipped tokens
        """
        cache = self.layer_caches[layer_id]
        if "Y" not in cache:
            return None

        if self.offload_to_cpu:
            # CPU offload mode
            indices_cpu = indices_skip.cpu()
            B = indices_cpu.shape[0]
            Y_cached_list = []

            for b in range(B):
                Y_cached_list.append(cache["Y"][b, indices_cpu[b]])

            Y_cached = torch.stack(Y_cached_list, dim=0)
            return Y_cached.to(indices_skip.device)
        else:
            # GPU mode: direct indexing
            B = indices_skip.shape[0]
            Y_cached_list = []
            
            for b in range(B):
                Y_cached_list.append(cache["Y"][b, indices_skip[b]])
            
            Y_cached = torch.stack(Y_cached_list, dim=0)
            return Y_cached
    
    def update_cache(
        self,
        layer_id: int,
        x_current: torch.Tensor,
        K_new: torch.Tensor,
        V_new: torch.Tensor,
        Y_new: torch.Tensor,
        indices_compute: torch.Tensor = None,
    ):
        """
        Update cache with newly computed values.
        
        Args:
            layer_id: Layer index
            x_current: Current input [B, N, D]
            K_new: New K values (either full or partial)
            V_new: New V values (either full or partial)
            Y_new: New Y output values (either full or partial)
            indices_compute: If partial, indices that were computed
        """
        cache = self.layer_caches[layer_id]

        # Store current input for next step comparison
        if self.offload_to_cpu:
            cache["X_prev"] = x_current.detach().cpu().clone()
        else:
            cache["X_prev"] = x_current.detach().clone()

        if indices_compute is None:
            # Full computation, cache everything
            if self.offload_to_cpu:
                cache["K"] = K_new.detach().cpu().clone()
                cache["V"] = V_new.detach().cpu().clone()
                cache["Y"] = Y_new.detach().cpu().clone()
            else:
                cache["K"] = K_new.detach().clone()
                cache["V"] = V_new.detach().clone()
                cache["Y"] = Y_new.detach().clone()
        else:
            # Partial computation, update only computed indices
            B, N_compute = indices_compute.shape

            # Initialize cache if not exists
            if "K" not in cache:
                B_full, N_full, D = x_current.shape
                if self.offload_to_cpu:
                    cache["K"] = torch.zeros(B_full, N_full, K_new.shape[-1],
                                            dtype=K_new.dtype, device="cpu")
                    cache["V"] = torch.zeros(B_full, N_full, V_new.shape[-1],
                                            dtype=V_new.dtype, device="cpu")
                    cache["Y"] = torch.zeros(B_full, N_full, Y_new.shape[-1],
                                            dtype=Y_new.dtype, device="cpu")
                else:
                    cache["K"] = torch.zeros(B_full, N_full, K_new.shape[-1],
                                            dtype=K_new.dtype, device=K_new.device)
                    cache["V"] = torch.zeros(B_full, N_full, V_new.shape[-1],
                                            dtype=V_new.dtype, device=V_new.device)
                    cache["Y"] = torch.zeros(B_full, N_full, Y_new.shape[-1],
                                            dtype=Y_new.dtype, device=Y_new.device)

            # Update computed indices
            if self.offload_to_cpu:
                indices_cpu = indices_compute.cpu()
                for b in range(B):
                    cache["K"][b, indices_cpu[b]] = K_new[b].detach().cpu()
                    cache["V"][b, indices_cpu[b]] = V_new[b].detach().cpu()
                    cache["Y"][b, indices_cpu[b]] = Y_new[b].detach().cpu()
            else:
                for b in range(B):
                    cache["K"][b, indices_compute[b]] = K_new[b].detach()
                    cache["V"][b, indices_compute[b]] = V_new[b].detach()
                    cache["Y"][b, indices_compute[b]] = Y_new[b].detach()
    
    def advance_step(self):
        """Advance to next timestep."""
        self.current_step += 1


class MaskedTokenCache:
    """
    🚀 REDESIGNED: Mask-aware token-level cache for inpainting (similar to TokenTeaCache architecture).
    
    Key differences from old design:
    - Does NOT change sequence length
    - Marks which tokens to compute (generative 100%, margin 50%, context 20%)
    - Block performs selective computation on marked tokens
    - Uses EMA cache to fill non-computed tokens
    
    Architecture:
    - pick_tokens_to_compute(): Returns indices_compute and indices_skip
    - block_forward_with_masked_token_cache(): Performs selective block computation
    - Cache stores: K, V, Y (output), X_prev for each layer
    
    Expected speedup: ~2.4x on typical inpainting tasks (20% mask coverage)
    """
    
    def __init__(
        self,
        margin_pixels: int = 5,  # Changed: margin in LATENT space (tokens), not pixel space
        context_sample_ratio: float = 0.05,  # Reduced from 0.2 for better speedup
        margin_sample_ratio: float = 0.7,     # Increased from 0.5 for better edge quality
        ema_alpha: float = 0.99,
        warmup_steps: int = 2,
        max_cached_layers: int = 20,  # Limit cache to save memory for long videos
        num_inference_steps: int = 30,  # Total denoising steps
        timestep_cache_threshold: float = 0.1,  # TeaCache rel_l1_thresh (lower = more aggressive)
        model_id: str = "Wan2.1-VACE-1.3B",  # For TeaCache coefficients
        use_kmeans_sampling: bool = False,  # Use K-Means in hidden space for better context quality
        kmeans_n_clusters: int = 100,  # Number of clusters for K-Means (10-200, lower=faster, higher=better quality)
        use_attention_guidance: bool = False,  # [PHASE 1] Use attention maps to guide context token selection
        use_attention_interaction: bool = False,  # [NEW] Attention-interaction guided sampling (default: False)
        use_generative_ema: bool = False,  # Use EMA for generative tokens (default: False, direct replacement)
        generative_ema_alpha: float = 0.99,  # EMA alpha for generative tokens (if use_generative_ema=True)
        num_layers: int = None,  # [FIX] Actual number of transformer layers (will be set from dit.blocks)
    ):
        # Note: margin_pixels is now in LATENT token space, not pixel space
        # For reference: pixel_space_margin = margin_pixels * 16 (8x VAE + 2x patch)
        # margin_pixels=3 → ~48 pixels in image space
        
        # [DEBUG] Print initialization parameters
        # print(f"\n{'='*80}")
        # print(f"[MaskedTokenCache] Initializing with:")
        # print(f"  context_sample_ratio: {context_sample_ratio}")
        # print(f"  margin_sample_ratio: {margin_sample_ratio}")
        # print(f"  use_kmeans_sampling: {use_kmeans_sampling}")
        # print(f"  kmeans_n_clusters: {kmeans_n_clusters}")
        # print(f"  use_attention_guidance: {use_attention_guidance}")
        # print(f"  use_attention_interaction: {use_attention_interaction}")  # [NEW]
        # print(f"  use_generative_ema: {use_generative_ema}")
        # print(f"{'='*80}\n")
        
        self.margin_pixels = margin_pixels
        self.context_sample_ratio = context_sample_ratio
        self.margin_sample_ratio = margin_sample_ratio
        self.ema_alpha = ema_alpha
        self.warmup_steps = warmup_steps
        self.max_cached_layers = max_cached_layers
        self.num_inference_steps = num_inference_steps
        self.use_kmeans_sampling = use_kmeans_sampling
        self.kmeans_n_clusters_target = kmeans_n_clusters  # User-specified target
        self.use_generative_ema = use_generative_ema  # NEW: Use EMA for generative tokens
        self.generative_ema_alpha = generative_ema_alpha  # NEW: EMA alpha for generative tokens
        
        # K-Means cache for context sampling (warm start)
        self.kmeans_cluster_centers = None  # Cached centers for faster convergence
        self.kmeans_n_clusters = None
        
        # V2: K-Means results caching (computed once at last layer of full-compute step)
        self.kmeans_cluster_labels = None  # Cluster assignments: [N_context] int labels
        self.kmeans_cluster_centers = None  # Cluster centers: [n_clusters, D]
        self.kmeans_context_indices = None  # Global indices of context tokens: [N_context]
        self.kmeans_valid = False  # Whether cached result is valid for current step
        self.num_layers = num_layers if num_layers is not None else 20  # [FIX] Use actual layer count
        
        # [PHASE 1] Attention-guided sampling
        self.use_attention_guidance = use_attention_guidance
        self.cached_attention = None  # Attention map from layer 0: [B, N_context, N_generative]
        self.attention_timestep = -1  # Which timestep the attention was computed
        
        # [NEW] Attention-Interaction guided context selection (Layer 23)
        # Only enabled when use_attention_interaction=True (default: False for backward compatibility)
        self.use_attention_interaction = use_attention_interaction
        self.cached_attention_last_layer = None  # [B, N_context, N_generative] fp16 from layer 23
        self.attention_last_layer_timestep = -1  # Timestep when last_layer attention was cached
        self.attention_cache_device = 'cuda'  # Keep on GPU for faster access (can be 'cpu' for memory saving)
        
        # TeaCache timestep skipping parameters (exact copy from TeaCache)
        self.timestep_cache_threshold = timestep_cache_threshold
        self.accumulated_rel_l1_distance = 0
        self.previous_modulated_input = None
        
        # V3: Partial-compute vs Full-compute discrimination
        # [MASKED_TEACACHE] Three-level decision:
        # - < 1.0x threshold: skip step
        # - 1.0x ~ 1.5x: partial compute (sample context tokens with K-Means+Attention)
        # - >= 1.5x: full compute (all tokens, cache attention/K-Means)
        self.partial_compute_threshold = timestep_cache_threshold * 1.5  # 1.5x threshold for partial
        self.is_full_compute = True  # Track current step type
        
        # TeaCache polynomial coefficients for different models
        self.coefficients_dict = {
            "Wan2.1-T2V-1.3B": [-5.21862437e+04, 9.23041404e+03, -5.28275948e+02, 1.36987616e+01, -4.99875664e-02],
            "Wan2.1-T2V-14B": [-3.03318725e+05, 4.90537029e+04, -2.65530556e+03, 5.87365115e+01, -3.15583525e-01],
            "Wan2.1-I2V-14B-480P": [2.57151496e+05, -3.54229917e+04,  1.40286849e+03, -1.35890334e+01, 1.32517977e-01],
            "Wan2.1-I2V-14B-720P": [ 8.10705460e+03,  2.13393892e+03, -3.72934672e+02,  1.66203073e+01, -4.17769401e-02],
            "Wan2.1-VACE-1.3B": [-5.21862437e+04, 9.23041404e+03, -5.28275948e+02, 1.36987616e+01, -4.99875664e-02],  # Use T2V-1.3B coefficients
        }
        if model_id not in self.coefficients_dict:
            print(f"[Warning] Model {model_id} not in TeaCache coefficients, using Wan2.1-T2V-1.3B")
            model_id = "Wan2.1-T2V-1.3B"
        self.coefficients = self.coefficients_dict[model_id]
        
        self.current_step = 0
        
        # Per-layer cache (same structure as TokenTeaCache)
        self.layer_caches = []  # Will be initialized with num_layers
        
        # Token classification (computed once per step, reused across layers)
        self.token_classification = None
        self.indices_compute_cache = None  # [B, N_compute]
        self.indices_skip_cache = None  # [B, N_skip]
        self.cached_context_indices = None  # [FIX] Cache context_indices to avoid repeated torch.where
        
        # [OPTIMIZATION] Cache K-Means candidates for attention-interaction
        self.attention_interaction_candidates = None  # Local indices of K-Means candidates
        self.attention_interaction_candidates_step = -1  # Step when candidates were computed
        
        # Mask latent storage
        self.mask_latent = None
        self.h = None
        self.w = None
        
        # Timestep-level caching (TeaCache integration)
        self.previous_modulated_input = None
        self.previous_hidden_states = None
        self.previous_residual = None
    
    def reset_step(self):
        """Reset at start of new denoising iteration."""
        self.current_step = 0
        self.accumulated_rel_l1_distance = 0
        self.previous_modulated_input = None
        self.previous_hidden_states = None
        self.previous_residual = None
        self.layer_caches = []
        self.token_classification = None
        self.indices_compute_cache = None
        self.indices_skip_cache = None
    
    def advance_step(self):
        """Advance to next timestep."""
        self.current_step += 1
        # Clear token classification (will re-classify next step)
        self.token_classification = None
        self.indices_compute_cache = None
        self.indices_skip_cache = None
        self.cached_context_indices = None  # [FIX] Clear cached context indices
        
        # [IMPORTANT] DON'T invalidate K-Means cache when entering full-compute step!
        # The old cache remains valid until new cache is computed at last layer.
        # This allows full-compute steps to use old K-Means if needed during early layers.
        # The cache will be overwritten by compute_and_cache_kmeans at the last layer.
        
        # Note: K-Means cache is only invalidated when context indices change (see _sample_from_kmeans_clusters)
        
        # Clear layer caches periodically to save memory (every 5 steps)
        if self.current_step % 5 == 0:
            self.layer_caches.clear()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def initialize_mask(self, mask_latent: torch.Tensor, h: int, w: int):
        """Store mask information for this denoising step."""
        self.mask_latent = mask_latent
        self.h = h
        self.w = w
    
    def check_timestep_skip(self, x: torch.Tensor, t_mod: torch.Tensor) -> tuple:
        """
        Check timestep compute level using TeaCache's logic with 2-level discrimination.
        
        Returns:
            should_skip: True if timestep can be skipped entirely (use cached residual)
            is_full_compute: True if full computation needed, False for partial
        
        Logic:
            - Skip: accumulated_distance < threshold → use previous_residual
            - Partial: threshold < accumulated_distance < 2*threshold → prune context tokens
            - Full: accumulated_distance >= 2*threshold → compute all tokens
        """
        import numpy as np
        
        modulated_inp = t_mod.clone()
        
        # First and last steps must compute (TeaCache logic)
        # Also must compute if no previous_residual exists yet
        if (self.current_step == 0 or 
            self.current_step == self.num_inference_steps - 1 or
            self.previous_residual is None):
            should_calc = True
            self.accumulated_rel_l1_distance = 0
            is_full_compute = True
        else:
            coefficients = self.coefficients
            rescale_func = np.poly1d(coefficients)
            rel_l1 = ((modulated_inp - self.previous_modulated_input).abs().mean() / 
                      self.previous_modulated_input.abs().mean()).cpu().item()
            self.accumulated_rel_l1_distance += rescale_func(rel_l1)
            
            # Three-level decision
            if self.accumulated_rel_l1_distance < self.timestep_cache_threshold:
                # Skip entire step (only if we have valid cache)
                should_calc = False
                is_full_compute = False
            elif self.accumulated_rel_l1_distance < self.partial_compute_threshold:
                # Partial compute (prune context tokens)
                should_calc = True
                is_full_compute = False
                self.accumulated_rel_l1_distance = 0
            else:
                # Full compute (all tokens)
                should_calc = True
                is_full_compute = True
                self.accumulated_rel_l1_distance = 0
        
        self.previous_modulated_input = modulated_inp
        self.is_full_compute = is_full_compute
        
        # Store hidden states for residual calculation (if computing)
        if should_calc:
            self.previous_hidden_states = x.clone()
        
        return not should_calc, is_full_compute
    
    def store_timestep_residual(self, hidden_states: torch.Tensor):
        """Store residual for timestep skipping (like TeaCache)."""
        if self.previous_hidden_states is not None:
            self.previous_residual = hidden_states - self.previous_hidden_states
            self.previous_hidden_states = None
    
    def apply_timestep_residual(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply cached residual to skip timestep computation."""
        if self.previous_residual is not None:
            return hidden_states + self.previous_residual
        return hidden_states
    
    def classify_tokens(self, x: torch.Tensor = None) -> dict:
        """
        V2: Classify tokens AFTER DiT tokenization (in token space, not latent space).
        
        This ensures accurate classification by aligning with the actual token grid,
        avoiding the blurry boundaries in VAE latent space.
        
        Args:
            x: Tokenized features [B, N, D] (optional, used to infer N)
        
        Returns:
            dict with 'generative', 'margin', 'context' boolean masks [B, N]
        """
        import torch.nn.functional as F
        from scipy.ndimage import distance_transform_edt
        import time
        
        t_start = time.time()
        
        mask_latent = self.mask_latent
        h, w = self.h, self.w
        device = mask_latent.device
        
        # Infer batch size and sequence length
        if x is not None:
            B, N, D = x.shape
            T = N // (h * w)
        else:
            B = mask_latent.shape[0]
            N = None
            T = None
        
        # Handle different mask formats
        if mask_latent.ndim == 5:
            # VACE mask latents [B, C, T_mask, H_latent, W_latent]
            B_mask, C, T_mask, H_latent, W_latent = mask_latent.shape
            
            # Step 1: Average over channels to get binary mask [B, T, H, W]
            mask_binary = (mask_latent.mean(dim=1) > 0.5).float()  # [B, T_mask, H_latent, W_latent]
            
            # Step 2: Infer T from x if available, else use T_mask
            if T is None:
                T_actual = T_mask
            else:
                T_actual = T
            
            # Replicate mask if needed (single frame mask → all frames)
            if T_mask == 1 and T_actual > 1:
                mask_binary = mask_binary.repeat(1, T_actual, 1, 1)  # [B, T, H, W]
            
            # Step 3: Downsample to TOKEN grid [B, T, h, w]
            # This is the KEY IMPROVEMENT: align mask with actual token positions
            mask_token_space = F.interpolate(
                mask_binary.view(B * T_actual, 1, H_latent, W_latent),
                size=(h, w),
                mode='nearest'
            ).view(B, T_actual, h, w)
            
        elif mask_latent.ndim == 4:
            # Simple mask [B, C, H, W] - assume single frame
            B_mask = mask_latent.shape[0]
            H_latent, W_latent = mask_latent.shape[2], mask_latent.shape[3]
            
            mask_single = (mask_latent.mean(dim=1, keepdim=True) > 0.5).float()  # [B, 1, H, W]
            
            # Downsample to token space
            mask_token_space_single = F.interpolate(
                mask_single, size=(h, w), mode='nearest'
            )  # [B, 1, h, w]
            
            # Infer T
            if T is None:
                T_actual = 1
            else:
                T_actual = T
                mask_token_space_single = mask_token_space_single.repeat(1, T_actual, 1, 1)
            
            mask_token_space = mask_token_space_single  # [B, T, h, w]
        else:
            raise ValueError(f"Unexpected mask shape: {mask_latent.shape}")
        
        # Flatten: [B, T*h*w]
        B_actual, T_actual = mask_token_space.shape[0], mask_token_space.shape[1]
        mask_flat = mask_token_space.view(B_actual, -1)  # [B, N]
        
        # Generative: inside mask
        generative_mask = mask_flat > 0.5
        
        # Margin: distance transform IN TOKEN SPACE (not latent!)
        # This is KEY: we compute distance in the aligned token grid
        B_shape, N_flat = mask_flat.shape
        h_w = h * w
        margin_mask = torch.zeros_like(mask_flat, dtype=torch.bool)
        
        for b in range(B_shape):
            for t in range(T_actual):
                # Extract single frame mask in TOKEN space [h, w]
                mask_frame = mask_token_space[b, t].cpu().numpy()  # [h, w]
                
                # Distance transform on token grid
                dist_outside = distance_transform_edt(1 - mask_frame)
                margin_frame = (dist_outside > 0) & (dist_outside <= self.margin_pixels)
                
                # Place back into flattened tensor
                start_idx = t * h_w
                end_idx = (t + 1) * h_w
                margin_mask[b, start_idx:end_idx] = torch.from_numpy(margin_frame.flatten()).to(device)
        
        # Context: everything else
        context_mask = ~generative_mask & ~margin_mask
        
        # Debug: check distribution
        # t_classify = time.time() - t_start
        # print("\n[Token Classification V2]")
        # # The above code is a Python script that contains a print statement. However, the print
        # statement is empty, so it will not output anything when executed.
        # print(f"  Classification time: {t_classify:.4f}s")
        # print(f"  Mask shape: {mask_latent.shape} → Token space: {mask_token_space.shape}")
        # print(f"  Token grid: T={T_actual}, h={h}, w={w}, N={N_flat}")
        # print(f"  margin_pixels (token space): {self.margin_pixels} tokens (~{self.margin_pixels*16} pixels)")
        # print(f"  Generative tokens: {generative_mask.sum().item()} ({generative_mask.float().mean()*100:.1f}%)")
        # print(f"  Margin tokens: {margin_mask.sum().item()} ({margin_mask.float().mean()*100:.1f}%)")
        # print(f"  Context tokens: {context_mask.sum().item()} ({context_mask.float().mean()*100:.1f}%)")
        
        # Debug: check distance distribution
        if B_shape > 0 and T_actual > 0:
            mask_frame = mask_token_space[0, 0].cpu().numpy()
            dist_outside = distance_transform_edt(1 - mask_frame)
            # print(f"  Distance transform stats: min={dist_outside.min():.1f}, max={dist_outside.max():.1f}, mean={dist_outside.mean():.1f}")
            # print(f"  Tokens within margin (<={self.margin_pixels}): {(dist_outside <= self.margin_pixels).sum()} / {dist_outside.size}")
        
        return {
            'generative': generative_mask,
            'margin': margin_mask,
            'context': context_mask,
        }
    
    def pick_tokens_to_compute(self, x: torch.Tensor, layer_id: int) -> tuple:
        """
        Determine which tokens need computation (mirroring TokenTeaCache interface).
        
        Args:
            x: Current input [B, N, D]
            layer_id: Layer index
            
        Returns:
            indices_compute: [B, N_compute] - tokens to compute
            indices_skip: [B, N_skip] - tokens to skip (use cache)
            should_use_cache: bool
        """
        # Warmup: compute all tokens
        if self.current_step < self.warmup_steps:
            return None, None, False
        
        # Classify tokens on first layer of each step
        # V2: Pass x to classify_tokens() for accurate token grid alignment
        if self.token_classification is None:
            if self.mask_latent is None:
                # No mask provided - fall back to full computation
                return None, None, False
            
            self.token_classification = self.classify_tokens(x=x)  # V2: Pass x
        
        # Use cached indices if available
        if self.indices_compute_cache is not None:
            return self.indices_compute_cache, self.indices_skip_cache, True
        
        # Build compute/skip indices
        B, N, D = x.shape
        device = x.device
        
        classification = self.token_classification
        generative = classification['generative']  # [B, N]
        margin = classification['margin']
        context = classification['context']
        
        # For batch_size=1 (typical for video)
        assert B == 1, "MaskedTokenCache currently supports batch_size=1"
        
        generative_indices = torch.where(generative[0])[0]
        margin_indices = torch.where(margin[0])[0]
        context_indices = torch.where(context[0])[0]
        
        # [FIX] Cache context_indices for reuse in block_forward (cached at layer_id==0)
        if layer_id == 0:
            self.cached_context_indices = context_indices
        
        # Validate indices are within bounds
        assert generative_indices.max() < N if len(generative_indices) > 0 else True, "Generative indices out of bounds"
        assert margin_indices.max() < N if len(margin_indices) > 0 else True, "Margin indices out of bounds"
        assert context_indices.max() < N if len(context_indices) > 0 else True, "Context indices out of bounds"
        
        # Sample margin tokens with spatial coherence (uniform stride, not random)
        N_margin = len(margin_indices)
        K_margin = max(1, int(N_margin * self.margin_sample_ratio)) if N_margin > 0 else 0
        if K_margin < N_margin:
            # Use uniform stride for spatial continuity (better edge quality)
            stride = N_margin // K_margin
            margin_sampled = margin_indices[::stride][:K_margin]
        else:
            margin_sampled = margin_indices
        
        # Sample context tokens based on compute type
        N_context = len(context_indices)
        K_context = max(1, int(N_context * self.context_sample_ratio)) if N_context > 0 else 0
        
        # [MASKED_TEACACHE DEBUG] Log sampling strategy at layer 0
        if layer_id == 0:
            cache_available = self.cached_attention_last_layer is not None
            cache_age = self.get_attention_cache_age(self.current_step) if cache_available else 999
            kmeans_available = self.kmeans_valid and self.kmeans_cluster_labels is not None
            
            # print(f"\n{'='*70}")
            # print(f"[MASKED_TEACACHE] Step {self.current_step}, Layer {layer_id}")
            # print(f"{'='*70}")
            # print(f"TeaCache Distance: {self.accumulated_rel_l1_distance:.4f} (threshold={self.timestep_cache_threshold:.4f}, partial={self.partial_compute_threshold:.4f})")
            
            # if self.is_full_compute:
            #     print(f"Decision: FULL COMPUTE (distance >= {self.partial_compute_threshold:.4f})")
            #     print(f"  → Computing ALL tokens: generative + margin + context")
            #     print(f"  → Will cache at last layer (29):")
            #     if self.use_kmeans_sampling:
            #         print(f"     • K-Means clustering ({self.kmeans_n_clusters_target} clusters target)")
            #     if self.use_attention_interaction:
            #         print(f"     • Sparse attention map (context→generative)")
            #     print(f"  Context tokens: {N_context} (all)")
            # else:
            #     print(f"Decision: PARTIAL COMPUTE ({self.timestep_cache_threshold:.4f} <= distance < {self.partial_compute_threshold:.4f})")
            #     print(f"  → Using generative + margin + SAMPLED context")
            #     print(f"  Sampling Strategy:")
            #     if self.use_kmeans_sampling and kmeans_available and self.use_attention_interaction and cache_available:
            #         n_clusters = len(torch.unique(self.kmeans_cluster_labels)) if self.kmeans_cluster_labels is not None else 0
            #         print(f"     • K-Means + Attention (COMBINED)")
            #         print(f"       - {n_clusters} semantic clusters")
            #         print(f"       - Top-K by attention within each cluster")
            #     elif self.use_attention_interaction and cache_available:
            #         print(f"     • Attention-only (cache_age={cache_age})")
            #     elif self.use_kmeans_sampling and kmeans_available:
            #         n_clusters = len(torch.unique(self.kmeans_cluster_labels)) if self.kmeans_cluster_labels is not None else 0
            #         print(f"     • K-Means-only ({n_clusters} clusters)")
            #     else:
            #         print(f"     • Uniform stride (fallback)")
            #     print(f"  Context tokens: {N_context} → {K_context} ({K_context/N_context*100:.1f}%)")
            # print(f"{'='*70}\n")
        
        if self.is_full_compute:
            # Full-compute: use all context tokens (will do K-Means at last layer)
            context_sampled = context_indices
            if layer_id == 0 and (self.use_attention_interaction or self.use_kmeans_sampling):
                strategy = "K-Means+Attention" if (self.use_kmeans_sampling and self.use_attention_interaction) else \
                          ("Attention" if self.use_attention_interaction else "K-Means")
                # print(f"  → Using FULL context (all {len(context_indices)} tokens, will cache {strategy})")
                # print(f"  [DEBUG] use_kmeans_sampling={self.use_kmeans_sampling}, use_attention_interaction={self.use_attention_interaction}")
        elif self.use_attention_interaction and self.use_kmeans_sampling and \
             self.kmeans_valid and self.cached_attention_last_layer is not None:
            # [COMBINED] K-Means + Attention: use K-Means clusters + attention scores within clusters
            # This combines semantic diversity (K-Means) with relevance ranking (Attention)
            # if layer_id == 0:
                # print(f"  → Using K-MEANS+ATTENTION sampling ({len(context_indices)} → {K_context} tokens)")
                # print(f"  [DEBUG] kmeans_valid={self.kmeans_valid}, n_clusters={len(torch.unique(self.kmeans_cluster_labels)) if self.kmeans_cluster_labels is not None else 'None'}")
                # print(f"  [DEBUG] cached_attention_last_layer available: {self.cached_attention_last_layer is not None}")
            context_sampled = self._kmeans_attention_combined_sample(
                context_indices, K_context, layer_id
            )
        elif self.use_attention_interaction and self.cached_attention_last_layer is not None:
            # [ATTENTION-ONLY] Use attention-interaction guided sampling
            # Only active when use_attention_interaction=True and no K-Means
            if layer_id == 0:
                # print(f"  → Using ATTENTION-ONLY sampling ({len(context_indices)} → {K_context} tokens)")
                # print(f"  [DEBUG] use_kmeans_sampling={self.use_kmeans_sampling}, kmeans_valid={self.kmeans_valid}")
                # print(f"  [DEBUG] Why not combined? kmeans_sampling={self.use_kmeans_sampling}, kmeans_valid={self.kmeans_valid}, attn_cache={self.cached_attention_last_layer is not None}")
                pass
            context_sampled = self._attention_interaction_sample_context(
                x, context_indices, K_context, layer_id
            )
        elif self.kmeans_valid and self.kmeans_cluster_labels is not None:
            # Partial-compute with cached K-Means: sample from each cluster
            # This is used when use_attention_interaction=False or attention cache unavailable
            if layer_id == 0:
                # print(f"  → Using K-MEANS-ONLY sampling ({len(context_indices)} → {K_context} tokens)")
                # print(f"  [DEBUG] n_clusters={len(torch.unique(self.kmeans_cluster_labels))}")
                # print(f"  [DEBUG] use_kmeans_sampling={self.use_kmeans_sampling}, kmeans_valid={self.kmeans_valid}")
                pass
            context_sampled = self._sample_from_kmeans_clusters(
                context_indices, K_context, layer_id
            )
        elif self.use_kmeans_sampling and K_context < N_context and N_context > 0:
            # Fallback: old K-Means (shouldn't happen if full-compute runs first)
            context_sampled = self._kmeans_sample_context(
                x, context_indices, K_context, layer_id
            )
        elif K_context < N_context:
            # Fallback: uniform stride
            stride = N_context // K_context
            context_sampled = context_indices[::stride][:K_context]
        else:
            context_sampled = context_indices
        
        # Combine: generative (100%) + margin (50%) + context (20%)
        indices_compute = torch.cat([
            generative_indices,
            margin_sampled,
            context_sampled
        ])
        
        # Remove duplicates and sort
        indices_compute = torch.unique(indices_compute, sorted=True)
        
        # Skip: remaining tokens
        compute_set = set(indices_compute.tolist())
        skip_list = [i for i in range(N) if i not in compute_set]
        indices_skip = torch.tensor(skip_list, device=device, dtype=torch.long)
        
        # Profiling: log token distribution
        # if layer_id == 0:  # Only log on first layer
        #     # print(f"\n[MaskedTokenCache Profiling] Step {self.current_step}")
        #     # print(f"  Total tokens: {N}")
        #     # print(f"  Generative: {len(generative_indices)} ({len(generative_indices)/N*100:.1f}%)")
        #     # print(f"  Margin: {N_margin} → sampled {K_margin} ({K_margin/N*100:.1f}%)")
        #     # print(f"  Context: {N_context} → sampled {K_context} ({K_context/N*100:.1f}%)")
        #     # print(f"  → Compute: {len(indices_compute)} ({len(indices_compute)/N*100:.1f}%)")
        #     # print(f"  → Skip: {len(indices_skip)} ({len(indices_skip)/N*100:.1f}%)")
            
        #     # Memory profiling
        #     if torch.cuda.is_available():
        #         mem_allocated = torch.cuda.memory_allocated() / 1024**2
        #         mem_cached = torch.cuda.memory_reserved() / 1024**2
        #         print(f"  GPU Memory: {mem_allocated:.1f}MB allocated, {mem_cached:.1f}MB reserved")
        
        # Validate total coverage
        assert len(indices_compute) + len(indices_skip) == N, f"Token coverage error: {len(indices_compute)} + {len(indices_skip)} != {N}"
        
        # Reshape for batch
        indices_compute = indices_compute.unsqueeze(0)  # [1, N_compute]
        indices_skip = indices_skip.unsqueeze(0)  # [1, N_skip]
        
        # Cache for reuse across layers in same step
        self.indices_compute_cache = indices_compute
        self.indices_skip_cache = indices_skip
        
        return indices_compute, indices_skip, True
    
    def get_cached_kv(self, layer_id: int, indices_skip: torch.Tensor) -> tuple:
        """Retrieve cached K, V for skipped tokens."""
        if layer_id >= len(self.layer_caches):
            return None, None
        
        cache = self.layer_caches[layer_id]
        if "K" not in cache or "V" not in cache:
            return None, None
        
        B = indices_skip.shape[0]
        K_list = []
        V_list = []
        
        for b in range(B):
            K_list.append(cache["K"][b, indices_skip[b]])
            V_list.append(cache["V"][b, indices_skip[b]])
        
        K_cached = torch.stack(K_list, dim=0)
        V_cached = torch.stack(V_list, dim=0)
        
        return K_cached, V_cached
    
    def get_cached_output(self, layer_id: int, indices_skip: torch.Tensor) -> torch.Tensor:
        """Retrieve cached output Y for skipped tokens."""
        if layer_id >= len(self.layer_caches):
            return None
        
        cache = self.layer_caches[layer_id]
        if "Y" not in cache:
            return None
        
        B = indices_skip.shape[0]
        Y_list = []
        
        for b in range(B):
            Y_list.append(cache["Y"][b, indices_skip[b]])
        
        return torch.stack(Y_list, dim=0)
    
    def _attention_guided_sample_context(
        self,
        context_indices: torch.Tensor,
        K_context: int,
        layer_id: int,
    ) -> torch.Tensor:
        """
        [PHASE 1] Sample context tokens using cached attention map.
        Select tokens with highest attention scores to generative region.
        
        Args:
            context_indices: Indices of context tokens [N_context]
            K_context: Number of tokens to sample
            layer_id: Current layer ID
            
        Returns:
            sampled_indices: Selected context token indices [K_context]
        """
        import time
        
        t_start = time.time()
        
        # Get cached attention [B, N_context, N_generative]
        # Move back to GPU if needed
        attention = self.cached_attention
        if attention.device != context_indices.device:
            attention = attention.to(context_indices.device)
        
        B, N_attn_context, N_generative = attention.shape
        N_context = len(context_indices)
        
        # Compute relevance: sum of attention over all generative tokens
        # Higher score = more relevant to generative region
        relevance_scores = attention[0].sum(dim=1)  # [N_attn_context]
        
        # Handle case where context_indices might not match attention shape
        # (e.g., if mask changed between timesteps - shouldn't happen but be safe)
        if N_attn_context != N_context:
            print(f"  [Warning] Attention shape mismatch: {N_attn_context} vs {N_context}, "
                  f"falling back to uniform sampling")
            # Fallback: uniform stride sampling
            stride = max(1, N_context // K_context)
            selected_local = list(range(0, N_context, stride))[:K_context]
            return context_indices[selected_local]
        
        # Select top-K context tokens by relevance
        K_actual = min(K_context, N_context)
        _, top_indices = torch.topk(relevance_scores, k=K_actual, largest=True)
        
        # Convert local indices to global token indices
        selected_indices = context_indices[top_indices.cpu().numpy()]
        
        # t_elapsed = time.time() - t_start
        
        # if layer_id == 0:
        #     print(f"  [Attention-Guided] Sampling time: {t_elapsed*1000:.1f}ms, "
        #           f"Selected {len(selected_indices)}/{N_context} context tokens, "
        #           f"Attention age: {self.current_step - self.attention_timestep} steps")
        
        return selected_indices
    
    def _kmeans_sample_context(
        self,
        x: torch.Tensor,
        context_indices: torch.Tensor,
        K_context: int,
        layer_id: int,
    ) -> torch.Tensor:
        """
        Sample context tokens using K-Means clustering in hidden space.
        Use fixed number of clusters (e.g., 100) and sample multiple tokens per cluster.
        
        [PHASE 1] If attention guidance is enabled and attention is available,
        delegate to attention-guided sampling instead.
        
        Args:
            x: Hidden features [B, N, D]
            context_indices: Indices of context tokens [N_context]
            K_context: Number of tokens to sample
            layer_id: Current layer ID
            
        Returns:
            sampled_indices: Selected context token indices [K_context]
        """
        # [PHASE 1] Check if we should use attention-guided sampling
        if self.use_attention_guidance and self.cached_attention is not None:
            return self._attention_guided_sample_context(context_indices, K_context, layer_id)
        
        # Otherwise, proceed with K-Means distance-based sampling
        import time
        import numpy as np
        from sklearn.cluster import MiniBatchKMeans
        
        t_start = time.time()
        
        B, N, D = x.shape
        N_context = len(context_indices)
        
        # Extract context token features [N_context, D]
        # Convert to float32 if needed (sklearn doesn't support bfloat16)
        context_features = x[0, context_indices].detach()
        if context_features.dtype == torch.bfloat16:
            context_features = context_features.to(torch.float32)
        context_features = context_features.cpu().numpy()
        
        # Use fixed number of clusters (much smaller than K_context)
        # Rule: min(user_target, K_context, N_context // 2)
        # Relaxed from N_context // 10 to allow higher cluster counts (e.g., 100)
        n_clusters = min(self.kmeans_n_clusters_target, K_context, max(10, N_context // 2))
        tokens_per_cluster = max(1, K_context // n_clusters)
        
        # Initialize K-Means with cached centers (warm start)
        init_method = 'k-means++'
        use_warm_start = False
        if (self.kmeans_cluster_centers is not None and 
            self.kmeans_n_clusters == n_clusters and
            self.kmeans_cluster_centers.shape[1] == D):
            # Use cached centers for warm start
            init_method = self.kmeans_cluster_centers
            use_warm_start = True
        
        # Use MiniBatchKMeans for speed (much faster than standard KMeans)
        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            init=init_method,
            n_init=1 if use_warm_start else 3,
            max_iter=100,
            batch_size=min(1000, N_context),
            random_state=42,
        )
        kmeans.fit(context_features)
        
        # Cache cluster centers for next timestep
        self.kmeans_cluster_centers = kmeans.cluster_centers_
        self.kmeans_n_clusters = n_clusters
        
        # Sample tokens from each cluster
        labels = kmeans.labels_
        selected_local_indices = []
        
        for cluster_id in range(n_clusters):
            cluster_mask = (labels == cluster_id)
            cluster_size = cluster_mask.sum()
            if cluster_size == 0:
                continue
            
            # Determine how many tokens to sample from this cluster
            n_sample = min(tokens_per_cluster, cluster_size)
            
            # Get cluster members
            cluster_indices_local = np.where(cluster_mask)[0]
            
            if n_sample == 1 or cluster_size == 1:
                # Select closest to center
                cluster_features = context_features[cluster_mask]
                cluster_center = kmeans.cluster_centers_[cluster_id]
                distances = np.linalg.norm(cluster_features - cluster_center, axis=1)
                closest_idx = np.argmin(distances)
                selected_local_indices.append(cluster_indices_local[closest_idx])
            else:
                # Sample multiple: closest + uniformly sampled
                cluster_features = context_features[cluster_mask]
                cluster_center = kmeans.cluster_centers_[cluster_id]
                distances = np.linalg.norm(cluster_features - cluster_center, axis=1)
                
                # Always include closest
                closest_idx = np.argmin(distances)
                selected = [cluster_indices_local[closest_idx]]
                
                # Uniform sampling for the rest
                remaining = n_sample - 1
                if remaining > 0:
                    stride = max(1, cluster_size // remaining)
                    for i in range(0, cluster_size, stride):
                        if len(selected) >= n_sample:
                            break
                        if i != closest_idx:  # Avoid duplicate
                            selected.append(cluster_indices_local[i])
                
                selected_local_indices.extend(selected[:n_sample])
        
        # Ensure we have exactly K_context tokens (or close to it)
        selected_local_indices = selected_local_indices[:K_context]
        
        # Convert to global token indices
        selected_indices = context_indices[selected_local_indices]
        
        # t_elapsed = time.time() - t_start
        
        # if layer_id == 0:  # Only log on first layer
        #     print(f"  [K-Means] Clustering time: {t_elapsed*1000:.1f}ms, "
        #           f"Clusters: {n_clusters}, Tokens/cluster: ~{tokens_per_cluster}, "
        #           f"Total: {len(context_indices)}→{len(selected_indices)}")
        
        return selected_indices
    
    def cache_attention_map(self, attention_weights: torch.Tensor, layer_id: int):
        """
        Cache attention map from layer 0 for attention-guided sampling.
        
        Args:
            attention_weights: Attention weights [B, N_context, N_generative]
                               Context tokens attending to generative tokens
            layer_id: Layer index (we only cache from layer 0)
        """
        if not self.use_attention_guidance:
            return
        
        # Only cache from layer 0
        if layer_id != 0:
            return
        
        # Store sparse attention (context → generative only)
        # attention_weights is already in the correct format
        self.cached_attention = attention_weights.detach().cpu()  # Move to CPU to save GPU memory
        self.attention_timestep = self.current_step
        
        # if layer_id == 0:
        #     print(f"  [Attention Cache] Stored attention map: {attention_weights.shape}, "
        #           f"timestep={self.current_step}, memory~{attention_weights.numel()*4/1024/1024:.1f}MB")
    
    def cache_attention_last_layer(self, attention_weights: torch.Tensor, layer_id: int, timestep: int, 
                                   total_layers: int = None, x: torch.Tensor = None, context_indices: torch.Tensor = None):
        """
        [NEW] Cache attention map from LAST LAYER for attention-interaction guided sampling.
        Also computes and caches K-Means candidates for the next partial-compute steps.
        
        Only enabled when use_attention_interaction=True (default: False for backward compatibility).
        
        Args:
            attention_weights: Sparse attention [B, num_heads, N_context, N_generative] or 
                              averaged [B, N_context, N_generative]
            layer_id: Layer index (we only cache from last layer)
            timestep: Current timestep
            total_layers: Total number of layers in the model (to update self.num_layers)
            x: Hidden features [B, N, D] for K-Means computation (optional)
            context_indices: Indices of context tokens (optional)
        """
        if not self.use_attention_interaction:
            return
        
        # Update num_layers if provided (first time only)
        if total_layers is not None and self.num_layers != total_layers:
            self.num_layers = total_layers
        
        # Only cache from last layer
        if layer_id != (self.num_layers - 1):
            return
        
        # Average over heads if needed
        if attention_weights.ndim == 4:  # [B, num_heads, N_context, N_generative]
            attention_weights = attention_weights.mean(dim=1)  # [B, N_context, N_generative]
        
        # Store in fp16 to save memory (64 MB → 32 MB for typical case)
        attention_fp16 = attention_weights.half().detach()
        
        # Keep on GPU or move to CPU based on config
        if self.attention_cache_device == 'cpu':
            self.cached_attention_last_layer = attention_fp16.cpu()
        else:
            self.cached_attention_last_layer = attention_fp16
        
        self.attention_last_layer_timestep = timestep
        
        # [OPTIMIZATION] Compute K-Means candidates here for next partial-compute steps
        if x is not None and context_indices is not None:
            self._compute_kmeans_candidates_for_attention_interaction(x, context_indices)
        
        # [DEBUG] Profiling
        memory_mb = attention_fp16.numel() * 2 / 1024 / 1024  # fp16 = 2 bytes
        # print(f"\n[cache_attention_last_layer] Layer {layer_id}, Step {timestep}")
        # print(f"  Cached attention: {attention_fp16.shape}, memory={memory_mb:.1f}MB ({self.attention_cache_device})")
        if x is not None and context_indices is not None:
            # print(f"  K-Means candidates: {len(self.attention_interaction_candidates)} tokens")
        # print(f"  This will be used in NEXT partial-compute step\n")
    
    def get_attention_cache_age(self, current_timestep: int) -> int:
        """
        [NEW] Get the age (in steps) of the cached attention map.
        Returns large number if no cache available.
        """
        if self.cached_attention_last_layer is None:
            return 999  # No cache
        return abs(current_timestep - self.attention_last_layer_timestep)
    
    def _compute_kmeans_candidates_for_attention_interaction(self, x: torch.Tensor, context_indices: torch.Tensor):
        """
        [OPTIMIZATION] Compute K-Means candidates at full-compute last layer.
        Cache results for use in subsequent partial-compute steps.
        
        This runs once per full-compute step, avoiding redundant K-Means computation
        in every partial-compute step.
        
        Args:
            x: Hidden features [B, N, D]
            context_indices: Indices of context tokens [N_context]
        """
        import numpy as np
        from sklearn.cluster import MiniBatchKMeans
        
        N_context = len(context_indices)
        # Compute 2x candidates (will select top-K using attention in partial steps)
        K_context_typical = int(N_context * self.context_sample_ratio)
        K_candidates = min(2 * K_context_typical, N_context)
        
        # Extract context features
        context_features = x[0, context_indices].detach()
        if context_features.dtype == torch.bfloat16:
            context_features = context_features.to(torch.float32)
        context_features_np = context_features.cpu().numpy()
        
        # K-Means with more clusters for better diversity
        # Use user-specified kmeans_n_clusters_target, relaxed limit to N_context // 2
        # [FIX] Remove K_candidates limit to match compute_and_cache_kmeans behavior
        n_clusters = min(self.kmeans_n_clusters_target, max(10, N_context // 2))
        tokens_per_cluster = max(1, K_candidates // n_clusters)
        
        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            init='k-means++',
            n_init=3,
            max_iter=100,
            batch_size=min(1000, N_context),
            random_state=42,
        )
        kmeans.fit(context_features_np)
        labels = kmeans.labels_
        
        # Sample from each cluster
        candidate_local_indices = []
        for cluster_id in range(n_clusters):
            cluster_mask = (labels == cluster_id)
            cluster_indices = np.where(cluster_mask)[0]
            cluster_size = len(cluster_indices)
            
            if cluster_size == 0:
                continue
            
            n_sample = min(tokens_per_cluster, cluster_size)
            
            if n_sample == 1:
                # Sample closest to center
                cluster_features = context_features_np[cluster_mask]
                center = kmeans.cluster_centers_[cluster_id]
                distances = np.linalg.norm(cluster_features - center, axis=1)
                closest = np.argmin(distances)
                candidate_local_indices.append(cluster_indices[closest])
            else:
                # Uniform sampling
                stride = max(1, cluster_size // n_sample)
                sampled = cluster_indices[::stride][:n_sample]
                candidate_local_indices.extend(sampled)
        
        # Ensure we have K_candidates
        candidate_local_indices = candidate_local_indices[:K_candidates]
        
        # Cache for use in next partial-compute steps
        self.attention_interaction_candidates = candidate_local_indices
        self.attention_interaction_candidates_step = self.current_step
        
        # [FIX] DON'T cache K-Means here if use_kmeans_sampling=True
        # Because compute_and_cache_kmeans will run later and handle K-Means caching properly
        # Only cache K-Means here if ONLY attention-interaction is enabled (not kmeans sampling)
        if not self.use_kmeans_sampling:
            # Pure attention-interaction mode: cache K-Means for potential combined sampling
            self.kmeans_cluster_labels = torch.from_numpy(labels).to(torch.long)  # [N_context]
            self.kmeans_cluster_centers = kmeans.cluster_centers_  # [n_clusters, D] numpy
            self.kmeans_context_indices = context_indices.clone()  # [N_context]
            self.kmeans_valid = True
            # print(f"  [DEBUG] K-Means cached for attention-only mode: {n_clusters} clusters")
        else:
            # K-Means sampling enabled: compute_and_cache_kmeans will handle caching later
            # print(f"  [DEBUG] K-Means caching deferred to compute_and_cache_kmeans (use_kmeans_sampling=True)")
    
    def _attention_interaction_sample_context(
        self,
        x: torch.Tensor,
        context_indices: torch.Tensor,
        K_context: int,
        layer_id: int,
    ) -> torch.Tensor:
        """
        [NEW] Two-stage context sampling: K-Means (semantic diversity) → Attention (interaction strength).
        
        Stage 1: K-Means clustering to get 2*K_context candidates (ensure semantic diversity)
                 - Computed in full-compute last layer, cached for partial-compute steps
        Stage 2: Use cached attention from last layer to select K_context tokens with highest 
                 interaction with generative region
        
        Only enabled when use_attention_interaction=True (default: False).
        Falls back to pure K-Means if attention cache is unavailable or stale.
        
        Args:
            x: Hidden features [B, N, D]
            context_indices: Indices of context tokens [N_context]
            K_context: Number of tokens to sample
            layer_id: Current layer ID
            
        Returns:
            sampled_indices: Selected context token indices [K_context]
        """
        import numpy as np
        
        # Check if attention cache is available and fresh
        cache_age = self.get_attention_cache_age(self.current_step)
        if self.cached_attention_last_layer is None or cache_age > 15:
            # Fallback to pure K-Means if no cache or cache too old (>15 steps)
            return self._kmeans_sample_context(x, context_indices, K_context, layer_id)
        
        # === Stage 1: Get K-Means candidates (from cache or compute) ===
        N_context = len(context_indices)
        K_candidates = min(2 * K_context, N_context)  # 2x oversampling
        
        # Check if we have valid cached candidates
        candidates_valid = (
            self.attention_interaction_candidates is not None and
            self.attention_interaction_candidates_step == self.current_step
        )
        
        if candidates_valid:
            # Use cached candidates from full-compute last layer
            candidate_local_indices = self.attention_interaction_candidates
            candidate_global_indices = context_indices[candidate_local_indices]
        else:
            # This should only happen in edge cases (fallback)
            # Use uniform sampling as emergency fallback
            stride = max(1, N_context // K_candidates)
            candidate_local_indices = list(range(0, N_context, stride))[:K_candidates]
            candidate_global_indices = context_indices[candidate_local_indices]
        
        # === Stage 2: Attention-based refinement ===
        # Load cached attention from last layer
        attention = self.cached_attention_last_layer  # [B, N_context, N_generative] fp16
        
        # Move to GPU if needed (use x's device)
        if attention.device != x.device:
            attention = attention.to(x.device)
        
        B, N_attn_context, N_generative = attention.shape
        
        # Sanity check: attention should match context size
        if N_attn_context != N_context:
            # print(f"  [Warning] Attention shape mismatch: {N_attn_context} vs {N_context}, "
            #       f"falling back to K-Means only")
            return candidate_global_indices[:K_context]
        
        # Compute interaction strength for candidates
        # attention[0, candidate_local_indices, :] → [K_candidates, N_generative]
        candidate_attention = attention[0, candidate_local_indices, :]
        
        # Sum over generative tokens to get total interaction strength
        interaction_scores = candidate_attention.sum(dim=1)  # [K_candidates]
        
        # Select top-K by interaction strength
        K_final = min(K_context, len(candidate_global_indices))
        _, top_indices = torch.topk(interaction_scores, k=K_final, largest=True)
        
        # Map back to global indices
        selected_indices = candidate_global_indices[top_indices.cpu().numpy()]
        
        # Profiling (disabled for production)
        # if layer_id == 0:
        #     print(f"  [Attention-Interaction] Stage1: {N_context}→{len(candidate_global_indices)} (K-Means), "
        #           f"Stage2: {len(candidate_global_indices)}→{len(selected_indices)} (Attention), "
        #           f"Interaction range: [{interaction_scores.min():.4f}, {interaction_scores.max():.4f}]")
        
        return selected_indices
    
    def compute_and_cache_kmeans(self, x: torch.Tensor, context_indices: torch.Tensor, layer_id: int):
        """
        Compute K-Means clustering on context tokens at the LAST layer of full-compute step.
        Cache the results for use in subsequent partial-compute steps.
        
        Args:
            x: Hidden states from last DiT layer [B, N, D]
            context_indices: Global indices of context tokens [N_context]
            layer_id: Current layer ID
        """
        import time
        import numpy as np
        from sklearn.cluster import MiniBatchKMeans
        
        # [DEBUG] Always log when called
        # print(f"\n[compute_and_cache_kmeans] Called at Layer {layer_id}, Step {self.current_step}")
        # print(f"  is_full_compute={self.is_full_compute}, num_layers={self.num_layers}, layer_id={layer_id}")
        
        # Only run at last layer of full-compute steps
        if not self.is_full_compute or layer_id != (self.num_layers - 1):
            # print(f"  → Skipping: not (full_compute and last_layer)")
            return
        
        t_start = time.time()
        
        B, N, D = x.shape
        N_context = len(context_indices)
        
        if N_context == 0:
            return
        
        # Extract context token features from DEEP layer [N_context, D]
        context_features = x[0, context_indices].detach()
        if context_features.dtype == torch.bfloat16:
            context_features = context_features.to(torch.float32)
        context_features = context_features.cpu().numpy()
        
        # Determine number of clusters
        # Relaxed from N_context // 10 to N_context // 2 to allow higher cluster counts
        n_clusters = min(self.kmeans_n_clusters_target, max(10, N_context // 2))
        
        # Run K-Means
        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            init='k-means++',
            n_init=3,
            max_iter=100,
            batch_size=min(1000, N_context),
            random_state=42,
        )
        kmeans.fit(context_features)
        
        # Cache results
        self.kmeans_cluster_labels = torch.from_numpy(kmeans.labels_).to(torch.long)  # [N_context]
        self.kmeans_cluster_centers = kmeans.cluster_centers_  # [n_clusters, D] numpy
        self.kmeans_context_indices = context_indices.clone()  # [N_context]
        self.kmeans_valid = True
        
        t_elapsed = time.time() - t_start
        # print(f"\n[K-Means Cache] Layer {layer_id}, Step {self.current_step}")
        # print(f"  Computed {n_clusters} clusters from {N_context} context tokens in {t_elapsed*1000:.1f}ms")
        # print(f"  ✓ kmeans_valid set to True")
        # print(f"  This will be used in NEXT partial-compute steps\n")
    
    def _kmeans_attention_combined_sample(
        self,
        context_indices: torch.Tensor,
        K_context: int,
        layer_id: int,
    ) -> torch.Tensor:
        """
        Two-stage sampling: K-Means clustering + Attention-based selection within each cluster.
        
        Strategy:
        1. Use cached K-Means clusters to ensure semantic diversity
        2. Within each cluster, use attention scores to select most relevant tokens
        
        This combines:
        - K-Means: semantic diversity across different visual/feature clusters
        - Attention: relevance ranking within each cluster
        
        Args:
            context_indices: Current context token indices [N_context]
            K_context: Number of tokens to sample
            layer_id: Current layer ID
            
        Returns:
            sampled_indices: Selected context token indices [K_context]
        """
        import numpy as np
        
        # Fallback if no K-Means cache or no attention cache
        if not self.kmeans_valid or self.kmeans_cluster_labels is None:
            if layer_id == 0:
                print("  [Warning] No K-Means cache, falling back to pure attention sampling")
            return self._attention_interaction_sample_context(None, context_indices, K_context, layer_id)
        
        # [FIX] Use attention from LAST layer (not layer 0)
        if self.cached_attention_last_layer is None:
            if layer_id == 0:
                print("  [Warning] No attention cache, falling back to pure K-Means sampling")
            return self._sample_from_kmeans_clusters(context_indices, K_context, layer_id)
        
        # Verify context indices match cached indices
        if not torch.equal(context_indices, self.kmeans_context_indices):
            if layer_id == 0:
                print("  [Warning] Context indices changed, falling back to pure attention sampling")
            return self._attention_interaction_sample_context(None, context_indices, K_context, layer_id)
        
        # Get attention scores from LAST layer [B, N_context, N_generative]
        attention = self.cached_attention_last_layer
        if attention.device != context_indices.device:
            attention = attention.to(context_indices.device)
        
        B, N_attn_context, N_generative = attention.shape
        N_context = len(context_indices)
        
        if N_attn_context != N_context:
            if layer_id == 0:
                print("  [Warning] Attention shape mismatch, falling back to pure K-Means sampling")
            return self._sample_from_kmeans_clusters(context_indices, K_context, layer_id)
        
        # Compute attention-based relevance scores [N_context]
        relevance_scores = attention[0].sum(dim=1)  # Sum over all generative tokens
        relevance_scores = relevance_scores.cpu().numpy()
        
        # Get cluster labels [N_context]
        labels = self.kmeans_cluster_labels.numpy()
        n_clusters = len(np.unique(labels))
        tokens_per_cluster = max(1, K_context // n_clusters)
        
        selected_local_indices = []
        
        # For each cluster, select top-K tokens by attention score
        for cluster_id in range(n_clusters):
            cluster_mask = (labels == cluster_id)
            cluster_size = cluster_mask.sum()
            if cluster_size == 0:
                continue
            
            # Get tokens in this cluster
            cluster_indices_local = np.where(cluster_mask)[0]
            
            # Get attention scores for tokens in this cluster
            cluster_scores = relevance_scores[cluster_indices_local]
            
            # Determine how many tokens to sample from this cluster
            n_sample = min(tokens_per_cluster, cluster_size)
            
            # Select top-K tokens by attention score within this cluster
            if n_sample >= cluster_size:
                # Take all tokens in cluster
                selected = cluster_indices_local.tolist()
            else:
                # Select top-n_sample by attention score
                top_k_in_cluster = np.argsort(cluster_scores)[-n_sample:]  # Indices within cluster
                selected = cluster_indices_local[top_k_in_cluster].tolist()
            
            selected_local_indices.extend(selected)
        
        # Ensure exactly K_context tokens (distribute remaining tokens to largest clusters)
        if len(selected_local_indices) < K_context:
            remaining = K_context - len(selected_local_indices)
            # Get all tokens not yet selected
            all_indices = set(range(N_context))
            selected_set = set(selected_local_indices)
            remaining_indices = list(all_indices - selected_set)
            
            # Select remaining tokens by highest attention scores
            remaining_scores = relevance_scores[remaining_indices]
            top_remaining = np.argsort(remaining_scores)[-remaining:]
            selected_local_indices.extend([remaining_indices[i] for i in top_remaining])
        
        # Trim to exactly K_context if we have too many
        selected_local_indices = selected_local_indices[:K_context]
        
        # Convert to tensor and get global indices
        selected_local_tensor = torch.tensor(selected_local_indices, dtype=torch.long, device=context_indices.device)
        selected_indices = context_indices[selected_local_tensor]
        
        if layer_id == 0:
            # print(f"  [K-Means+Attention] {n_clusters} clusters × attention scores → {len(selected_indices)} tokens")
        
        return selected_indices
    
    def _sample_from_kmeans_clusters(
        self,
        context_indices: torch.Tensor,
        K_context: int,
        layer_id: int,
    ) -> torch.Tensor:
        """
        Sample context tokens from cached K-Means clusters (partial-compute steps).
        
        Strategy:
        - Each cluster contributes K_context / n_clusters tokens
        - Select cluster center (closest to centroid) for semantic coverage
        - Select spatially diverse tokens for spatial coverage
        - Propagate updates from sampled tokens to entire cluster via EMA
        
        Args:
            context_indices: Current context token indices [N_context]
            K_context: Number of tokens to sample
            layer_id: Current layer ID
            
        Returns:
            sampled_indices: Selected context token indices [K_context]
        """
        import numpy as np
        
        if not self.kmeans_valid or self.kmeans_cluster_labels is None:
            # Fallback: uniform sampling
            if K_context >= len(context_indices):
                return context_indices
            stride = len(context_indices) // K_context
            return context_indices[::stride][:K_context]
        
        # Verify context indices match cached indices
        if not torch.equal(context_indices, self.kmeans_context_indices):
            # print(f"  [Warning] Context indices changed, invalidating K-Means cache")
            self.kmeans_valid = False
            stride = len(context_indices) // K_context if K_context < len(context_indices) else 1
            return context_indices[::stride][:K_context]
        
        labels = self.kmeans_cluster_labels.numpy()  # [N_context]
        n_clusters = len(np.unique(labels))
        tokens_per_cluster = max(1, K_context // n_clusters)
        
        selected_local_indices = []
        
        for cluster_id in range(n_clusters):
            cluster_mask = (labels == cluster_id)
            cluster_size = cluster_mask.sum()
            if cluster_size == 0:
                continue
            
            n_sample = min(tokens_per_cluster, cluster_size)
            cluster_indices_local = np.where(cluster_mask)[0]
            
            # Always include one representative (we don't have distances here, use first)
            selected = [cluster_indices_local[0]]
            
            # Add spatially diverse tokens (uniform stride)
            if n_sample > 1:
                stride = max(1, cluster_size // n_sample)
                for i in range(stride, cluster_size, stride):
                    if len(selected) >= n_sample:
                        break
                    selected.append(cluster_indices_local[i])
            
            selected_local_indices.extend(selected[:n_sample])
        
        # Ensure exactly K_context tokens
        selected_local_indices = selected_local_indices[:K_context]
        
        # Convert to tensor for proper indexing
        selected_local_tensor = torch.tensor(selected_local_indices, dtype=torch.long, device=context_indices.device)
        selected_indices = context_indices[selected_local_tensor]
        
        if layer_id == 0:
            # print(f"  [K-Means Sample] Using cached clusters: {n_clusters} clusters → {len(selected_indices)} tokens")
        
        return selected_indices
    
    def update_cache(
        self,
        layer_id: int,
        x_current: torch.Tensor,
        K_new: torch.Tensor,
        V_new: torch.Tensor,
        Y_new: torch.Tensor,
        indices_compute: torch.Tensor = None,
    ):
        """
        Update cache with newly computed values.
        
        V2: Generative tokens use direct replacement (NO EMA)
            Margin/Context tokens use EMA update
        """
        # Skip caching for deep layers to save memory
        if layer_id >= self.max_cached_layers:
            return
        
        # Ensure cache list is long enough
        while len(self.layer_caches) <= layer_id:
            self.layer_caches.append({})
        
        cache = self.layer_caches[layer_id]
        
        if indices_compute is None:
            # Full computation - store everything
            cache["K"] = K_new.detach().clone()
            cache["V"] = V_new.detach().clone()
            cache["Y"] = Y_new.detach().clone()
        else:
            # Partial computation - update only computed tokens
            B, N_compute = indices_compute.shape
            B_full, N_full, D = x_current.shape
            
            # Initialize cache if needed
            if "K" not in cache:
                cache["K"] = torch.zeros(B_full, N_full, K_new.shape[-1],
                                        dtype=K_new.dtype, device=K_new.device)
                cache["V"] = torch.zeros(B_full, N_full, V_new.shape[-1],
                                        dtype=V_new.dtype, device=V_new.device)
                cache["Y"] = torch.zeros(B_full, N_full, Y_new.shape[-1],
                                        dtype=Y_new.dtype, device=Y_new.device)
            
            # Extract computed tokens from new values
            # K_new and V_new are already [B, N_compute, D]
            # But Y_new is full sequence [B, N_full, D], need to extract
            indices_expanded = indices_compute.unsqueeze(-1).expand(-1, -1, Y_new.shape[-1])
            Y_new_computed = torch.gather(Y_new, 1, indices_expanded)  # [B, N_compute, D]
            
            # V2: Separate generative from margin/context
            if self.token_classification is not None:
                classification = self.token_classification
                generative_mask = classification['generative']  # [B, N_full]
                
                alpha = self.ema_alpha
                for b in range(B):
                    computed_indices_b = indices_compute[b]
                    
                    # Validate indices are within bounds
                    max_idx = computed_indices_b.max().item() if len(computed_indices_b) > 0 else 0
                    if max_idx >= B_full * N_full:
                        print(f"[ERROR] Index out of bounds: max_idx={max_idx}, B_full={B_full}, N_full={N_full}")
                        print(f"  computed_indices_b shape: {computed_indices_b.shape}")
                        print(f"  computed_indices_b range: [{computed_indices_b.min().item()}, {computed_indices_b.max().item()}]")
                        # Skip this batch to avoid crash
                        continue
                    
                    # Check which computed indices are generative
                    is_generative = generative_mask[b, computed_indices_b]
                    
                    generative_local = torch.where(is_generative)[0]
                    margin_context_local = torch.where(~is_generative)[0]
                    
                    # Generative: Direct replacement OR EMA based on use_generative_ema flag
                    if len(generative_local) > 0:
                        gen_global = computed_indices_b[generative_local]
                        
                        # Additional bounds check
                        if gen_global.max().item() >= N_full:
                            print(f"[ERROR] Generative index out of bounds: {gen_global.max().item()} >= {N_full}")
                            continue
                        
                        if self.use_generative_ema:
                            # Use EMA for generative tokens (experimental)
                            gen_alpha = self.generative_ema_alpha
                            cache["K"][b, gen_global] = (
                                gen_alpha * cache["K"][b, gen_global] + 
                                (1 - gen_alpha) * K_new[b, generative_local].detach()
                            )
                            cache["V"][b, gen_global] = (
                                gen_alpha * cache["V"][b, gen_global] + 
                                (1 - gen_alpha) * V_new[b, generative_local].detach()
                            )
                            cache["Y"][b, gen_global] = (
                                gen_alpha * cache["Y"][b, gen_global] + 
                                (1 - gen_alpha) * Y_new_computed[b, generative_local].detach()
                            )
                        else:
                            # Direct replacement (default) - 100% accurate
                            cache["K"][b, gen_global] = K_new[b, generative_local].detach()
                            cache["V"][b, gen_global] = V_new[b, generative_local].detach()
                            cache["Y"][b, gen_global] = Y_new_computed[b, generative_local].detach()
                    
                    # Margin/Context: EMA update - smooth temporal continuity
                    if len(margin_context_local) > 0:
                        mc_global = computed_indices_b[margin_context_local]
                        
                        # Additional bounds check
                        if mc_global.max().item() >= N_full:
                            print(f"[ERROR] Margin/Context index out of bounds: {mc_global.max().item()} >= {N_full}")
                            continue
                        
                        # TODO V3: Cluster propagation (currently disabled for debugging)
                        # if not self.is_full_compute and self.kmeans_valid and self.kmeans_cluster_labels is not None:
                        #     self._propagate_to_clusters(...)
                        
                        # Normal EMA update
                        cache["K"][b, mc_global] = (
                            alpha * cache["K"][b, mc_global] + 
                            (1 - alpha) * K_new[b, margin_context_local].detach()
                        )
                        cache["V"][b, mc_global] = (
                            alpha * cache["V"][b, mc_global] + 
                            (1 - alpha) * V_new[b, margin_context_local].detach()
                        )
                        cache["Y"][b, mc_global] = (
                            alpha * cache["Y"][b, mc_global] + 
                            (1 - alpha) * Y_new_computed[b, margin_context_local].detach()
                        )
            else:
                # Fallback: Use EMA for all (old behavior)
                alpha = self.ema_alpha
                for b in range(B):
                    cache["K"][b, indices_compute[b]] = (
                        alpha * cache["K"][b, indices_compute[b]] + (1 - alpha) * K_new[b].detach()
                    )
                    cache["V"][b, indices_compute[b]] = (
                        alpha * cache["V"][b, indices_compute[b]] + (1 - alpha) * V_new[b].detach()
                    )
                    cache["Y"][b, indices_compute[b]] = (
                        alpha * cache["Y"][b, indices_compute[b]] + (1 - alpha) * Y_new_computed[b].detach()
                    )
    
    def _propagate_to_clusters(
        self,
        cache: dict,
        batch_idx: int,
        global_indices: torch.Tensor,
        local_indices: torch.Tensor,
        K_new: torch.Tensor,
        V_new: torch.Tensor,
        Y_new: torch.Tensor,
        alpha: float,
    ):
        """
        Propagate updated context tokens to their entire clusters (partial-compute only).
        
        For each computed context token:
        1. Find its cluster
        2. Update the token itself (EMA)
        3. Propagate update to all tokens in the same cluster (weighted EMA)
        
        Args:
            cache: Layer cache dict
            batch_idx: Batch index
            global_indices: Global token indices [N_mc]
            local_indices: Local indices in K_new/V_new [N_mc]
            K_new, V_new, Y_new: Newly computed values
            alpha: EMA weight
        """
        import numpy as np
        
        # Get context token classification
        classification = self.token_classification
        context_mask = classification['context'][batch_idx]  # [N_full]
        context_global_indices = torch.where(context_mask)[0]
        
        if not torch.equal(context_global_indices, self.kmeans_context_indices):
            # Context changed, fall back to normal EMA
            cache["K"][batch_idx, global_indices] = (
                alpha * cache["K"][batch_idx, global_indices] + 
                (1 - alpha) * K_new[batch_idx, local_indices].detach()
            )
            cache["V"][batch_idx, global_indices] = (
                alpha * cache["V"][batch_idx, global_indices] + 
                (1 - alpha) * V_new[batch_idx, local_indices].detach()
            )
            cache["Y"][batch_idx, global_indices] = (
                alpha * cache["Y"][batch_idx, global_indices] + 
                (1 - alpha) * Y_new[batch_idx, local_indices].detach()
            )
            return
        
        # Map global indices to context-local indices
        context_to_local = {idx.item(): i for i, idx in enumerate(context_global_indices)}
        labels = self.kmeans_cluster_labels.numpy()
        
        # For each computed context token
        for glob_idx, loc_idx in zip(global_indices, local_indices):
            if glob_idx.item() not in context_to_local:
                # Not a context token (shouldn't happen), skip
                continue
            
            context_local_idx = context_to_local[glob_idx.item()]
            cluster_id = labels[context_local_idx]
            
            # Find all tokens in this cluster
            cluster_mask = (labels == cluster_id)
            cluster_context_local = np.where(cluster_mask)[0]
            cluster_global = context_global_indices[cluster_context_local]
            
            # Update the computed token (normal EMA)
            cache["K"][batch_idx, glob_idx] = (
                alpha * cache["K"][batch_idx, glob_idx] + 
                (1 - alpha) * K_new[batch_idx, loc_idx].detach()
            )
            cache["V"][batch_idx, glob_idx] = (
                alpha * cache["V"][batch_idx, glob_idx] + 
                (1 - alpha) * V_new[batch_idx, loc_idx].detach()
            )
            cache["Y"][batch_idx, glob_idx] = (
                alpha * cache["Y"][batch_idx, glob_idx] + 
                (1 - alpha) * Y_new[batch_idx, loc_idx].detach()
            )
            
            # Propagate to other cluster members (higher alpha for smoothness)
            cluster_alpha = 0.95  # More conservative propagation
            other_members = cluster_global[cluster_global != glob_idx]
            
            if len(other_members) > 0:
                # Use the updated value (not the raw new value)
                updated_k = cache["K"][batch_idx, glob_idx]
                updated_v = cache["V"][batch_idx, glob_idx]
                updated_y = cache["Y"][batch_idx, glob_idx]
                
                cache["K"][batch_idx, other_members] = (
                    cluster_alpha * cache["K"][batch_idx, other_members] + 
                    (1 - cluster_alpha) * updated_k.detach()
                )
                cache["V"][batch_idx, other_members] = (
                    cluster_alpha * cache["V"][batch_idx, other_members] + 
                    (1 - cluster_alpha) * updated_v.detach()
                )
                cache["Y"][batch_idx, other_members] = (
                    cluster_alpha * cache["Y"][batch_idx, other_members] + 
                    (1 - cluster_alpha) * updated_y.detach()
                )
    
    def clear_cache(self):
        """Clear all cached data to free memory."""
        self.layer_caches.clear()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None


def block_forward_with_token_cache(
    block,
    x: torch.Tensor,
    context: torch.Tensor,
    t_mod: torch.Tensor,
    freqs: torch.Tensor,
    token_cache: TokenTeaCache,
    layer_id: int,
) -> torch.Tensor:
    """
    Forward pass through a DiT block with token-level caching.
    
    This implements the token caching strategy where:
    1. Pick which tokens need recomputation based on similarity
    2. Only compute Q/K/V for those tokens
    3. Reuse cached K/V for unchanged tokens
    4. Update cache with new values
    """
    # Determine which tokens to compute
    indices_compute, indices_skip, should_use_cache = token_cache.pick_tokens_to_compute(x, layer_id)
    
    if not should_use_cache or indices_compute is None:
        # Full computation (warmup or first step)
        # Pass empty dict to force tuple return from DiTBlock
        output, kv_cache = block(x, context, t_mod, freqs, indices_compute=None, cached_kv={})
        
        # Cache everything for next step
        token_cache.update_cache(
            layer_id=layer_id,
            x_current=x,
            K_new=kv_cache["k"],
            V_new=kv_cache["v"],
            Y_new=output,
            indices_compute=None
        )
        return output
    
    # Partial computation with caching
    # Build cached_kv dict from token cache
    cached_kv = {
        "k": token_cache.get_cached_kv(layer_id, indices_skip)[0],  # K_cached
        "v": token_cache.get_cached_kv(layer_id, indices_skip)[1],  # V_cached
        # Provide sliced cached outputs for skipped tokens (moved to device)
        "output": token_cache.get_cached_output(layer_id, indices_skip),
        "ffn_output": None,
    }
    # Try to build sliced FFN cache if present (move only needed slices to device)
    ffn_cache_full = token_cache.layer_caches[layer_id].get("ffn_output")
    if ffn_cache_full is not None:
        if token_cache.offload_to_cpu:
            # CPU offload mode: index on CPU and move to device
            indices_cpu = indices_skip.cpu()
            B = indices_cpu.shape[0]
            ffn_list = []
            for b in range(B):
                ffn_list.append(ffn_cache_full[b, indices_cpu[b]])
            if len(ffn_list) > 0:
                cached_kv["ffn_output"] = torch.stack(ffn_list, dim=0).to(x.device)
        else:
            # GPU mode: direct indexing
            B = indices_skip.shape[0]
            ffn_list = []
            for b in range(B):
                ffn_list.append(ffn_cache_full[b, indices_skip[b]])
            if len(ffn_list) > 0:
                cached_kv["ffn_output"] = torch.stack(ffn_list, dim=0)
    
    # Forward with partial computation
    output, kv_cache = block(x, context, t_mod, freqs, indices_compute, cached_kv)
    
    # Update cache with new computed values
    token_cache.update_cache(
        layer_id=layer_id,
        x_current=x,
        K_new=kv_cache["k"],
        V_new=kv_cache["v"],
        Y_new=output,
        indices_compute=indices_compute
    )
    
    # Store FFN output in cache for next iteration
    if "ffn_output" in kv_cache:
        cache = token_cache.layer_caches[layer_id]
        if indices_compute is None:
            # Full computation
            if token_cache.offload_to_cpu:
                cache["ffn_output"] = kv_cache["ffn_output"].detach().cpu().clone()
            else:
                cache["ffn_output"] = kv_cache["ffn_output"].detach().clone()
        else:
            # Partial update of FFN cache
            B, N_compute = indices_compute.shape
            if "ffn_output" not in cache:
                B_full, N_full, D = x.shape
                if token_cache.offload_to_cpu:
                    cache["ffn_output"] = torch.zeros(B_full, N_full, kv_cache["ffn_output"].shape[-1],
                                                      dtype=kv_cache["ffn_output"].dtype,
                                                      device="cpu")
                else:
                    cache["ffn_output"] = torch.zeros(B_full, N_full, kv_cache["ffn_output"].shape[-1],
                                                      dtype=kv_cache["ffn_output"].dtype,
                                                      device=kv_cache["ffn_output"].device)
            if token_cache.offload_to_cpu:
                indices_cpu = indices_compute.cpu()
                for b in range(B):
                    cache["ffn_output"][b, indices_cpu[b]] = kv_cache["ffn_output"][b].detach().cpu()
            else:
                for b in range(B):
                    cache["ffn_output"][b, indices_compute[b]] = kv_cache["ffn_output"][b].detach()
    
    return output


def block_forward_with_masked_token_cache(
    block,
    x: torch.Tensor,
    context: torch.Tensor,
    t_mod: torch.Tensor,
    freqs: torch.Tensor,
    masked_token_cache: MaskedTokenCache,
    layer_id: int,
    return_attention_weights: bool = False,  # [PHASE 1] For attention-guided sampling
    attention_mask_indices: dict = None,  # [PHASE 1] For sparse attention computation
) -> torch.Tensor:
    """
    🚀 Forward pass through DiT block with mask-aware selective computation.
    
    Similar to block_forward_with_token_cache but uses mask-based token selection.
    
    Flow:
    1. Pick tokens to compute based on mask classification (generative/margin/context)
    2. Compute only selected tokens
    3. Fill skipped tokens from EMA cache
    4. Update cache with new values
    
    [PHASE 1] If return_attention_weights=True (only for layer 0), also return attention map.
    """
    # Determine which tokens to compute
    indices_compute, indices_skip, should_use_cache = masked_token_cache.pick_tokens_to_compute(x, layer_id)
    
    if not should_use_cache or indices_compute is None:
        # Full computation (warmup or no mask)
        result = block(x, context, t_mod, freqs, 
                      indices_compute=None, indices_skip=None, cached_kv={},
                      return_attention_weights=return_attention_weights,
                      attention_mask_indices=attention_mask_indices)
        
        # Unpack result based on return_attention_weights
        if return_attention_weights:
            output, kv_cache, attn_weights = result
        else:
            output, kv_cache = result
        
        # Cache everything (including FFN output for partial steps)
        masked_token_cache.update_cache(
            layer_id=layer_id,
            x_current=x,
            K_new=kv_cache["k"],
            V_new=kv_cache["v"],
            Y_new=output,
            indices_compute=None
        )
        
        # TODO: Re-enable FFN caching after fixing index issues
        # Also cache FFN output if available
        # if "ffn_output" in kv_cache and layer_id < masked_token_cache.max_cached_layers:
        #     cache = masked_token_cache.layer_caches[layer_id]
        #     cache["ffn_output"] = kv_cache["ffn_output"].detach().clone()
        
        if return_attention_weights:
            return output, attn_weights
        return output
    
    #Partial computation with caching
    K_cached, V_cached = masked_token_cache.get_cached_kv(layer_id, indices_skip)
    output_cached = masked_token_cache.get_cached_output(layer_id, indices_skip)
    
    # ✅ FIX: Also get cached FFN output! (TEMPORARILY DISABLED for debugging)
    ffn_cached = None
    # TODO: Re-enable FFN caching after fixing index out of bounds issue
    # if layer_id < len(masked_token_cache.layer_caches):
    #     cache = masked_token_cache.layer_caches[layer_id]
    #     if "ffn_output" in cache:
    #         ...
    
    # If no cache available (e.g., deep layers), do full computation
    if K_cached is None or V_cached is None or output_cached is None:
        result = block(x, context, t_mod, freqs, 
                      indices_compute=None, indices_skip=None, cached_kv={},
                      return_attention_weights=return_attention_weights,
                      attention_mask_indices=attention_mask_indices)
        
        # Unpack result
        if return_attention_weights:
            output, kv_cache, attn_weights = result
        else:
            output, kv_cache = result
        
        # Cache if layer is within limit
        masked_token_cache.update_cache(
            layer_id=layer_id,
            x_current=x,
            K_new=kv_cache["k"],
            V_new=kv_cache["v"],
            Y_new=output,
            indices_compute=None
        )
        
        # [IMPORTANT] Cache K-Means before returning (when return_attention_weights=True)
        if masked_token_cache.use_kmeans_sampling and masked_token_cache.token_classification is not None:
            if masked_token_cache.cached_context_indices is not None:
                context_indices = masked_token_cache.cached_context_indices
            else:
                classification = masked_token_cache.token_classification
                context_mask = classification['context'][0]
                context_indices = torch.where(context_mask)[0]
            
            masked_token_cache.compute_and_cache_kmeans(output, context_indices, layer_id)
        
        if return_attention_weights:
            return output, attn_weights
        return output
    
    cached_kv = {
        "k": K_cached,
        "v": V_cached,
        "output": output_cached,
        "ffn_output": ffn_cached,  # ✅ FIX: Provide cached FFN output!
    }
    
    # Forward with partial computation (pass indices_skip explicitly)
    # Note: When using cached tokens, we don't compute full attention, so can't return weights
    output, kv_cache = block(x, context, t_mod, freqs, 
                            indices_compute, indices_skip, cached_kv,
                            return_attention_weights=False)  # Can't return weights with partial computation
    
    # Update cache
    masked_token_cache.update_cache(
        layer_id=layer_id,
        x_current=x,
        K_new=kv_cache["k"],
        V_new=kv_cache["v"],
        Y_new=output,
        indices_compute=indices_compute
    )
    
    # ✅ FIX: Cache FFN output for next iteration!
    if "ffn_output" in kv_cache and layer_id < masked_token_cache.max_cached_layers:
        cache = masked_token_cache.layer_caches[layer_id]
        B, N_compute = indices_compute.shape
        
        if "ffn_output" not in cache:
            # Initialize FFN cache
            B_full, N_full, D = x.shape
            cache["ffn_output"] = torch.zeros(B_full, N_full, kv_cache["ffn_output"].shape[-1],
                                              dtype=kv_cache["ffn_output"].dtype,
                                              device=kv_cache["ffn_output"].device)
        
        # Update FFN cache with newly computed values
        # Apply same EMA strategy as K/V/Y
        if masked_token_cache.token_classification is not None:
            classification = masked_token_cache.token_classification
            generative_mask = classification['generative']
            alpha = masked_token_cache.ema_alpha
            
            for b in range(B):
                computed_indices_b = indices_compute[b]
                is_generative = generative_mask[b, computed_indices_b]
                
                generative_local = torch.where(is_generative)[0]
                margin_context_local = torch.where(~is_generative)[0]
                
                # Generative: Direct replacement
                if len(generative_local) > 0:
                    gen_global = computed_indices_b[generative_local]
                    cache["ffn_output"][b, gen_global] = kv_cache["ffn_output"][b, generative_local].detach()
                
                # Margin/Context: EMA
                if len(margin_context_local) > 0:
                    mc_global = computed_indices_b[margin_context_local]
                    cache["ffn_output"][b, mc_global] = (
                        alpha * cache["ffn_output"][b, mc_global] +
                        (1 - alpha) * kv_cache["ffn_output"][b, margin_context_local].detach()
                    )
        else:
            # Fallback: EMA for all
            alpha = masked_token_cache.ema_alpha
            for b in range(B):
                cache["ffn_output"][b, indices_compute[b]] = (
                    alpha * cache["ffn_output"][b, indices_compute[b]] +
                    (1 - alpha) * kv_cache["ffn_output"][b].detach()
                )
    
    # V3: Compute and cache K-Means at last layer of full-compute steps
    # [IMPORTANT] Do this BEFORE returning, even when return_attention_weights=True!
    if masked_token_cache.use_kmeans_sampling and masked_token_cache.token_classification is not None:
        # [FIX] Use cached context_indices to avoid redundant torch.where
        if masked_token_cache.cached_context_indices is not None:
            context_indices = masked_token_cache.cached_context_indices
        else:
            classification = masked_token_cache.token_classification
            context_mask = classification['context'][0]  # [N]
            context_indices = torch.where(context_mask)[0]
        
        # Call K-Means caching (only runs at last layer of full-compute)
        masked_token_cache.compute_and_cache_kmeans(output, context_indices, layer_id)
    else:
        # [DEBUG] Why not calling K-Means cache?
        if layer_id >= 28:  # Only log for last few layers
            # print(f"\n[block_forward] Layer {layer_id}: NOT calling compute_and_cache_kmeans")
            # print(f"  use_kmeans_sampling={masked_token_cache.use_kmeans_sampling}")
            # print(f"  token_classification={'Available' if masked_token_cache.token_classification is not None else 'None'}\n")
    
    # If attention weights were requested, return them
    if return_attention_weights:
        return output, attn_weights
    return output


class TemporalTiler_BCTHW:
    def __init__(self):
        pass

    def build_1d_mask(self, length, left_bound, right_bound, border_width):
        x = torch.ones((length,))
        if border_width == 0:
            return x
        
        shift = 0.5
        if not left_bound:
            x[:border_width] = (torch.arange(border_width) + shift) / border_width
        if not right_bound:
            x[-border_width:] = torch.flip((torch.arange(border_width) + shift) / border_width, dims=(0,))
        return x

    def build_mask(self, data, is_bound, border_width):
        _, _, T, _, _ = data.shape
        t = self.build_1d_mask(T, is_bound[0], is_bound[1], border_width[0])
        mask = repeat(t, "T -> 1 1 T 1 1")
        return mask
    
    def run(self, model_fn, sliding_window_size, sliding_window_stride, computation_device, computation_dtype, model_kwargs, tensor_names, batch_size=None):
        tensor_names = [tensor_name for tensor_name in tensor_names if model_kwargs.get(tensor_name) is not None]
        tensor_dict = {tensor_name: model_kwargs[tensor_name] for tensor_name in tensor_names}
        B, C, T, H, W = tensor_dict[tensor_names[0]].shape
        if batch_size is not None:
            B *= batch_size
        data_device, data_dtype = tensor_dict[tensor_names[0]].device, tensor_dict[tensor_names[0]].dtype
        value = torch.zeros((B, C, T, H, W), device=data_device, dtype=data_dtype)
        weight = torch.zeros((1, 1, T, 1, 1), device=data_device, dtype=data_dtype)
        for t in range(0, T, sliding_window_stride):
            if t - sliding_window_stride >= 0 and t - sliding_window_stride + sliding_window_size >= T:
                continue
            t_ = min(t + sliding_window_size, T)
            model_kwargs.update({
                tensor_name: tensor_dict[tensor_name][:, :, t: t_:, :].to(device=computation_device, dtype=computation_dtype) \
                    for tensor_name in tensor_names
            })
            model_output = model_fn(**model_kwargs).to(device=data_device, dtype=data_dtype)
            mask = self.build_mask(
                model_output,
                is_bound=(t == 0, t_ == T),
                border_width=(sliding_window_size - sliding_window_stride,)
            ).to(device=data_device, dtype=data_dtype)
            value[:, :, t: t_, :, :] += model_output * mask
            weight[:, :, t: t_, :, :] += mask
        value /= weight
        model_kwargs.update(tensor_dict)
        return value



def model_fn_wan_video(
    dit: WanModel,
    motion_controller: WanMotionControllerModel = None,
    vace: VaceWanModel = None,
    latents: torch.Tensor = None,
    timestep: torch.Tensor = None,
    context: torch.Tensor = None,
    clip_feature: Optional[torch.Tensor] = None,
    y: Optional[torch.Tensor] = None,
    reference_latents = None,
    vace_context = None,
    vace_scale = 1.0,
    tea_cache: TeaCache = None,
    token_tea_cache: TokenTeaCache = None,
    masked_token_cache: MaskedTokenCache = None,  # 🚀 Masked token cache
    use_unified_sequence_parallel: bool = False,
    motion_bucket_id: Optional[torch.Tensor] = None,
    sliding_window_size: Optional[int] = None,
    sliding_window_stride: Optional[int] = None,
    cfg_merge: bool = False,
    use_gradient_checkpointing: bool = False,
    use_gradient_checkpointing_offload: bool = False,
    control_camera_latents_input = None,
    fuse_vae_embedding_in_latents: bool = False,
    vace_mask_latents = None,  # 🚀 Mask latents for masked token cache
    all_timesteps: Optional[torch.Tensor] = None,  # 🚀 For PAB
    use_adacache: bool = False,  # 🚀 AdaCache
    use_fastcache: bool = False,  # 🚀 FastCache
    **kwargs,
):
    if sliding_window_size is not None and sliding_window_stride is not None:
        model_kwargs = dict(
            dit=dit,
            motion_controller=motion_controller,
            vace=vace,
            latents=latents,
            timestep=timestep,
            context=context,
            clip_feature=clip_feature,
            y=y,
            reference_latents=reference_latents,
            vace_context=vace_context,
            vace_scale=vace_scale,
            tea_cache=tea_cache,
            use_unified_sequence_parallel=use_unified_sequence_parallel,
            motion_bucket_id=motion_bucket_id,
        )
        return TemporalTiler_BCTHW().run(
            model_fn_wan_video,
            sliding_window_size, sliding_window_stride,
            latents.device, latents.dtype,
            model_kwargs=model_kwargs,
            tensor_names=["latents", "y"],
            batch_size=2 if cfg_merge else 1
        )
    
    if use_unified_sequence_parallel:
        import torch.distributed as dist
        from xfuser.core.distributed import (get_sequence_parallel_rank,
                                            get_sequence_parallel_world_size,
                                            get_sp_group)

    # Timestep
    if dit.seperated_timestep and fuse_vae_embedding_in_latents:
        timestep = torch.concat([
            torch.zeros((1, latents.shape[3] * latents.shape[4] // 4), dtype=latents.dtype, device=latents.device),
            torch.ones((latents.shape[2] - 1, latents.shape[3] * latents.shape[4] // 4), dtype=latents.dtype, device=latents.device) * timestep
        ]).flatten()
        t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep).unsqueeze(0))
        if use_unified_sequence_parallel and dist.is_initialized() and dist.get_world_size() > 1:
            t_chunks = torch.chunk(t, get_sequence_parallel_world_size(), dim=1)
            t_chunks = [torch.nn.functional.pad(chunk, (0, 0, 0, t_chunks[0].shape[1]-chunk.shape[1]), value=0) for chunk in t_chunks]
            t = t_chunks[get_sequence_parallel_rank()]
        t_mod = dit.time_projection(t).unflatten(2, (6, dit.dim))
    else:
        t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep))
        t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))
    
    # Motion Controller
    if motion_bucket_id is not None and motion_controller is not None:
        t_mod = t_mod + motion_controller(motion_bucket_id).unflatten(1, (6, dit.dim))
    context = dit.text_embedding(context)

    x = latents
    # Merged cfg
    if x.shape[0] != context.shape[0]:
        x = torch.concat([x] * context.shape[0], dim=0)
    if timestep.shape[0] != context.shape[0]:
        timestep = torch.concat([timestep] * context.shape[0], dim=0)

    # Image Embedding
    if y is not None and dit.require_vae_embedding:
        x = torch.cat([x, y], dim=1)
    if clip_feature is not None and dit.require_clip_embedding:
        clip_embdding = dit.img_emb(clip_feature)
        context = torch.cat([clip_embdding, context], dim=1)
    
    # Add camera control
    x, (f, h, w) = dit.patchify(x, control_camera_latents_input)
    
    # Reference image
    if reference_latents is not None:
        if len(reference_latents.shape) == 5:
            reference_latents = reference_latents[:, :, 0]
        reference_latents = dit.ref_conv(reference_latents).flatten(2).transpose(1, 2)
        x = torch.concat([reference_latents, x], dim=1)
        f += 1
    
    freqs = torch.cat([
        dit.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
        dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
        dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
    ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)
    
    # TeaCache
    if tea_cache is not None:
        tea_cache_update = tea_cache.check(dit, x, t_mod)
    else:
        tea_cache_update = False
    
    # MaskedTokenCache timestep-level checking
    masked_token_timestep_skip = False
    if masked_token_cache is not None and vace_mask_latents is not None:
        masked_token_timestep_skip, is_full_compute = masked_token_cache.check_timestep_skip(x, t_mod)
        
        if masked_token_timestep_skip:
            # print(f"[MaskedTokenCache] Step {masked_token_cache.current_step}: SKIP timestep (accumulated_distance={masked_token_cache.accumulated_rel_l1_distance:.4f})")
        else:
            compute_type = "FULL" if is_full_compute else "PARTIAL"
            # print(f"[MaskedTokenCache] Step {masked_token_cache.current_step}: {compute_type} compute (accumulated_distance={masked_token_cache.accumulated_rel_l1_distance:.4f})")

        
    if vace_context is not None:
        vace_hints = vace(x, vace_context, context, t_mod, freqs)
    
    # 🚀 Masked Token Cache: Initialize mask for selective token computation
    if masked_token_cache is not None and vace_mask_latents is not None and not tea_cache_update and not masked_token_timestep_skip:
        masked_token_cache.initialize_mask(vace_mask_latents, h, w)
    
    # blocks
    if use_unified_sequence_parallel:
        if dist.is_initialized() and dist.get_world_size() > 1:
            chunks = torch.chunk(x, get_sequence_parallel_world_size(), dim=1)
            pad_shape = chunks[0].shape[1] - chunks[-1].shape[1]
            chunks = [torch.nn.functional.pad(chunk, (0, 0, 0, chunks[0].shape[1]-chunk.shape[1]), value=0) for chunk in chunks]
            x = chunks[get_sequence_parallel_rank()]
    if tea_cache_update:
        x = tea_cache.update(x)
    elif masked_token_timestep_skip:
        # Use MaskedTokenCache timestep skipping
        x = masked_token_cache.apply_timestep_residual(x)
    else:
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)
            return custom_forward
        
        for block_id, block in enumerate(dit.blocks):
            # 🚀 Priority 1: Masked Token Cache (mask-aware selective computation)
            if masked_token_cache is not None and vace_mask_latents is not None and not use_gradient_checkpointing:
                # [PHASE 1] Request attention weights from layer 0 for attention-guided sampling
                # Only request when token_classification is available (i.e., not first step)
                request_attention = (block_id == 0 and 
                                   masked_token_cache.use_attention_guidance and
                                   masked_token_cache.token_classification is not None)
                
                if request_attention:
                    # Prepare sparse attention mask indices to avoid OOM
                    token_cls = masked_token_cache.token_classification
                    attention_mask_indices = None
                    if token_cls is not None:
                        generative_mask = token_cls["generative"][0]  # [N]
                        context_mask = token_cls["context"][0]  # [N]
                        
                        generative_indices = torch.where(generative_mask)[0]
                        context_indices = torch.where(context_mask)[0]
                        
                        # Pass indices to compute only sparse attention (context→generative)
                        # This avoids computing full N×N attention (OOM issue)
                        attention_mask_indices = {
                            "context": context_indices,
                            "generative": generative_indices
                        }
                    
                    result = block_forward_with_masked_token_cache(
                        block, x, context, t_mod, freqs,
                        masked_token_cache=masked_token_cache,
                        layer_id=block_id,
                        return_attention_weights=True,
                        attention_mask_indices=attention_mask_indices
                    )
                    # Unpack result
                    if isinstance(result, tuple) and len(result) == 2:
                        x, attn_weights = result
                        
                        # Cache sparse attention (already in correct shape: [B, num_heads, N_context, N_generative])
                        if attn_weights is not None:
                            # Average over heads: [B, num_heads, N_context, N_generative] → [B, N_context, N_generative]
                            attn_context_to_gen = attn_weights[0].mean(dim=0)  # [N_context, N_generative]
                            
                            # Cache it
                            masked_token_cache.cache_attention_map(
                                attn_context_to_gen.unsqueeze(0),  # [1, N_context, N_generative]
                                layer_id=block_id
                            )
                    else:
                        x = result
                else:
                    # [OPTIMIZED] Check if we need attention from last layer for attention-interaction
                    need_last_layer_attention = (
                        masked_token_cache.use_attention_interaction and 
                        masked_token_cache.is_full_compute and 
                        block_id == (len(dit.blocks) - 1)  # Last layer
                    )
                    
                    if need_last_layer_attention:
                        # Extract attention during the FIRST forward pass (no redundant computation)
                        # [FIX] Need to ensure token_classification exists before extracting attention
                        # During warmup (step 0), token_classification might be None
                        # So we need to create it here if needed
                        token_cls = masked_token_cache.token_classification
                        if token_cls is None:
                            # Create classification for attention extraction
                            # This is needed for warmup step (step 0) when pick_tokens_to_compute returns early
                            token_cls = masked_token_cache.classify_tokens(x=x)
                            masked_token_cache.token_classification = token_cls
                        
                        generative_mask = token_cls["generative"][0]
                        context_mask = token_cls["context"][0]
                        
                        generative_indices = torch.where(generative_mask)[0]
                        context_indices = torch.where(context_mask)[0]
                        
                        attention_mask_indices = {
                            "context": context_indices,
                            "generative": generative_indices
                        }
                        
                        # Get attention weights during normal forward pass
                        result = block_forward_with_masked_token_cache(
                            block, x, context, t_mod, freqs,
                            masked_token_cache=masked_token_cache,
                            layer_id=block_id,
                            return_attention_weights=True,
                            attention_mask_indices=attention_mask_indices
                        )
                        
                        if isinstance(result, tuple) and len(result) == 2:
                            x, attn_weights = result
                            # Cache for attention-interaction sampling
                            if attn_weights is not None:
                                # Also pass x and context_indices for K-Means computation
                                masked_token_cache.cache_attention_last_layer(
                                    attn_weights,
                                    layer_id=block_id,
                                    timestep=masked_token_cache.current_step,
                                    total_layers=len(dit.blocks),
                                    x=x,
                                    context_indices=context_indices
                                )
                        else:
                            x = result
                    else:
                        # Normal forward without attention extraction
                        x = block_forward_with_masked_token_cache(
                            block, x, context, t_mod, freqs,
                            masked_token_cache=masked_token_cache,
                            layer_id=block_id
                        )
                
                # [NEW] Extract attention from LAST LAYER for attention-interaction guidance
                # [OPTIMIZED] Extract attention during the FIRST forward pass to avoid redundant computation
                # This is SEPARATE from layer-0 attention used in use_attention_guidance
            # 🚀 Priority 2: Token-level TeaCache (similarity-based selective computation)
            elif token_tea_cache is not None and not use_gradient_checkpointing:
                x = block_forward_with_token_cache(
                    block, x, context, t_mod, freqs,
                    token_cache=token_tea_cache,
                    layer_id=block_id
                )
            # 🚀 Priority 3: AdaCache (adaptive caching based on feature difference)
            elif use_adacache and is_adacache_enabled() and not use_gradient_checkpointing:
                ada_wrapper = get_adacache_wrapper(block_id)
                ada_state = get_adacache_state()
                
                if ada_wrapper is not None and ada_state is not None:
                    # Get current step
                    current_step = int(timestep.item() if timestep.numel() == 1 else timestep[0].item())
                    
                    # Calculate spatial size (S = h * w)
                    spatial_size = x.shape[1] // f if f > 0 else x.shape[1]
                    
                    # Forward with AdaCache wrapper
                    result = block_forward_with_adacache(
                        block, x, context, t_mod, freqs,
                        wrapper=ada_wrapper,
                        ada_state=ada_state,
                        current_step=current_step,
                        spatial_size=spatial_size,
                        timestep=current_step,
                        all_timesteps=all_timesteps,
                    )
                    
                    # Handle return format
                    if isinstance(result, tuple):
                        x = result[0]
                    else:
                        x = result
                else:
                    # Fallback to normal computation
                    x = block(x, context, t_mod, freqs,
                             timestep=timestep.item() if timestep.numel() == 1 else timestep[0].item(),
                             all_timesteps=all_timesteps)
            # 🚀 Priority 4: FastCache (statistical + motion-aware caching)
            elif use_fastcache and is_fastcache_enabled() and not use_gradient_checkpointing:
                fastcache_wrapper = get_fastcache_wrapper(block_id)
                
                if fastcache_wrapper is not None:
                    # Get current timestep
                    current_step = int(timestep.item() if timestep.numel() == 1 else timestep[0].item())
                    
                    # Get max timesteps from all_timesteps
                    max_timesteps = len(all_timesteps) if all_timesteps is not None else 1000
                    
                    # Get hidden size
                    hidden_size = x.shape[-1]
                    
                    # Forward with FastCache wrapper
                    x = block_forward_with_fastcache(
                        block, x, context, t_mod, freqs,
                        timestep=current_step,
                        max_timesteps=max_timesteps,
                        block_id=block_id,
                        hidden_size=hidden_size,
                    )
                else:
                    # Fallback to normal computation
                    x = block(x, context, t_mod, freqs,
                             timestep=timestep.item() if timestep.numel() == 1 else timestep[0].item(),
                             all_timesteps=all_timesteps)
            # Regular computation
            elif use_gradient_checkpointing_offload:
                with torch.autograd.graph.save_on_cpu():
                    x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        x, context, t_mod, freqs,
                        use_reentrant=False,
                    )
            elif use_gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, context, t_mod, freqs,
                    use_reentrant=False,
                )
            else:
                x = block(x, context, t_mod, freqs, 
                         timestep=timestep.item() if timestep.numel() == 1 else timestep[0].item(),
                         all_timesteps=all_timesteps)
            
            # Apply VACE hints
            if vace_context is not None and block_id in vace.vace_layers_mapping:
                current_vace_hint = vace_hints[vace.vace_layers_mapping[block_id]]
                if use_unified_sequence_parallel and dist.is_initialized() and dist.get_world_size() > 1:
                    current_vace_hint = torch.chunk(current_vace_hint, get_sequence_parallel_world_size(), dim=1)[get_sequence_parallel_rank()]
                    current_vace_hint = torch.nn.functional.pad(current_vace_hint, (0, 0, 0, chunks[0].shape[1] - current_vace_hint.shape[1]), value=0)
                x = x + current_vace_hint * vace_scale
        
        if tea_cache is not None:
            tea_cache.store(x)
        
        # Store MaskedTokenCache residual for timestep skipping
        if masked_token_cache is not None and not masked_token_timestep_skip:
            masked_token_cache.store_timestep_residual(x)
        
        # Advance token cache step counter
        if token_tea_cache is not None:
            token_tea_cache.advance_step()
        
        # Advance masked token cache step counter
        if masked_token_cache is not None:
            masked_token_cache.advance_step()
            
    x = dit.head(x, t)
    if use_unified_sequence_parallel:
        if dist.is_initialized() and dist.get_world_size() > 1:
            x = get_sp_group().all_gather(x, dim=1)
            x = x[:, :-pad_shape] if pad_shape > 0 else x
    # Remove reference latents
    if reference_latents is not None:
        x = x[:, reference_latents.shape[1]:]
        f -= 1
    x = dit.unpatchify(x, (f, h, w))
    return x