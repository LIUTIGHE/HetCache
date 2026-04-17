import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional
from einops import rearrange
from .utils import hash_state_dict_keys
from .wan_video_camera_controller import SimpleAdapter

# Import PAB (Pyramid Attention Broadcast) manager
try:
    from re_PAB_mgr import (
        enable_pab,
        get_mlp_output,
        if_broadcast_cross,
        if_broadcast_mlp,
        if_broadcast_spatial,
        save_mlp_output,
    )
    PAB_AVAILABLE = True
except ImportError:
    PAB_AVAILABLE = False
    # Define dummy functions
    def enable_pab(): return False
    def if_broadcast_spatial(t, c): return False, c
    def if_broadcast_cross(t, c): return False, c
    def if_broadcast_mlp(t, c, b, a, is_t): return False, c, False, None
    def save_mlp_output(t, b, o, is_t): pass
    def get_mlp_output(r, t, b, is_t): return None

# Import AdaCache manager
try:
    from adacache_mgr import (
        is_adacache_enabled,
        get_adacache_state,
        compute_cache_diff,
        select_cache_rate_from_codebook,
        compute_motion_score,
        apply_motion_regularization,
    )
    ADACACHE_AVAILABLE = True
except ImportError:
    ADACACHE_AVAILABLE = False
    def is_adacache_enabled(): return False
    def get_adacache_state(): return None

try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except (ModuleNotFoundError, ImportError, OSError):
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except (ModuleNotFoundError, ImportError, OSError):
    FLASH_ATTN_2_AVAILABLE = False

try:
    from sageattention import sageattn
    SAGE_ATTN_AVAILABLE = True
except ModuleNotFoundError:
    SAGE_ATTN_AVAILABLE = False
    
    
def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_heads: int, compatibility_mode=False):
    if compatibility_mode:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    elif FLASH_ATTN_3_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
        x = flash_attn_interface.flash_attn_func(q, k, v)
        if isinstance(x,tuple):
            x = x[0]
        x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    elif FLASH_ATTN_2_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
        x = flash_attn.flash_attn_func(q, k, v)
        x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    elif SAGE_ATTN_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = sageattn(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    else:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    return x


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    return (x * (1 + scale) + shift)


def sinusoidal_embedding_1d(dim, position):
    sinusoid = torch.outer(position.type(torch.float64), torch.pow(
        10000, -torch.arange(dim//2, dtype=torch.float64, device=position.device).div(dim//2)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype)


def precompute_freqs_cis_3d(dim: int, end: int = 1024, theta: float = 10000.0):
    # 3d rope precompute
    f_freqs_cis = precompute_freqs_cis(dim - 2 * (dim // 3), end, theta)
    h_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    w_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    return f_freqs_cis, h_freqs_cis, w_freqs_cis


def precompute_freqs_cis(dim: int, end: int = 1024, theta: float = 10000.0):
    # 1d rope precompute
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)
                   [: (dim // 2)].double() / dim))
    freqs = torch.outer(torch.arange(end, device=freqs.device), freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def rope_apply(x, freqs, num_heads):
    x = rearrange(x, "b s (n d) -> b s n d", n=num_heads)
    x_out = torch.view_as_complex(x.to(torch.float64).reshape(
        x.shape[0], x.shape[1], x.shape[2], -1, 2))
    x_out = torch.view_as_real(x_out * freqs).flatten(2)
    return x_out.to(x.dtype)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        dtype = x.dtype
        return self.norm(x.float()).to(dtype) * self.weight


class AttentionModule(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads
        
    def forward(self, q, k, v, return_attention_weights=False, attention_mask_indices=None):
        """
        Forward with optional attention weights return.
        
        Args:
            q, k, v: Query, Key, Value tensors
            return_attention_weights: If True, compute and return attention weights
                                     (slower, only use for layer 0 with attention guidance)
            attention_mask_indices: Optional dict with {"context": [indices], "generative": [indices]}
                                   If provided, only compute sparse attention (context→generative)
        
        Returns:
            x: Attention output
            attention_weights (optional): [B, num_heads, N_context, N_generative] if sparse,
                                         or [B, num_heads, N_q, N_k] if full
        """
        if return_attention_weights:
            # Check if we can use sparse attention computation
            if attention_mask_indices is not None and "context" in attention_mask_indices and "generative" in attention_mask_indices:
                # Sparse attention: only compute context→generative
                # This saves massive memory: N_context×N_generative instead of N×N
                context_indices = attention_mask_indices["context"]  # [N_context]
                generative_indices = attention_mask_indices["generative"]  # [N_generative]
                
                # Full computation for output (use flash attention)
                x = flash_attention(q=q, k=k, v=v, num_heads=self.num_heads)
                
                # Sparse attention weights computation for guidance
                q = rearrange(q, "b s (n d) -> b n s d", n=self.num_heads)
                k = rearrange(k, "b s (n d) -> b n s d", n=self.num_heads)
                
                # Extract only context Q and generative K
                q_context = q[:, :, context_indices, :]  # [B, num_heads, N_context, d]
                k_generative = k[:, :, generative_indices, :]  # [B, num_heads, N_generative, d]
                
                # Compute sparse attention: context→generative only
                scale = q_context.size(-1) ** -0.5
                attn_weights = torch.matmul(q_context, k_generative.transpose(-2, -1)) * scale  # [B, num_heads, N_context, N_generative]
                attn_weights = torch.softmax(attn_weights, dim=-1)
                
                return x, attn_weights
            else:
                # Full attention computation (original method, memory-intensive)
                # This is only used for layer 0 when attention guidance is enabled
                q = rearrange(q, "b s (n d) -> b n s d", n=self.num_heads)
                k = rearrange(k, "b s (n d) -> b n s d", n=self.num_heads)
                v = rearrange(v, "b s (n d) -> b n s d", n=self.num_heads)
                
                # Compute attention weights: [B, num_heads, N_q, N_k]
                scale = q.size(-1) ** -0.5
                attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
                attn_weights = torch.softmax(attn_weights, dim=-1)
                
                # Apply attention
                x = torch.matmul(attn_weights, v)
                x = rearrange(x, "b n s d -> b s (n d)", n=self.num_heads)
                
                return x, attn_weights
        else:
            # Use fast flash attention (no weights returned)
            x = flash_attention(q=q, k=k, v=v, num_heads=self.num_heads)
            return x


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
        
        self.attn = AttentionModule(self.num_heads)

    def forward(self, x, freqs, indices_compute=None, indices_skip=None, cached_kv=None, return_attention_weights=False, attention_mask_indices=None):
        """
        Forward with optional token-level caching support.
        
        Args:
            x: Input tensor [B, N, D]
            freqs: RoPE frequencies
            indices_compute: Optional indices of tokens to compute [B, N_compute]
            indices_skip: Optional indices of tokens to skip [B, N_skip]
            cached_kv: Optional dict with cached {"k": [B, N_skip, D], "v": [B, N_skip, D]}
            return_attention_weights: If True, return attention weights (only for layer 0, slows down)
            attention_mask_indices: Optional dict with {"context": [indices], "generative": [indices]}
                                   for sparse attention computation
        
        Returns:
            output: Attention output [B, N, D] (if no caching)
            OR (output, kv_cache): tuple if caching is enabled
            OR (output, kv_cache, attention_weights): if return_attention_weights=True
        """
        B, N, D = x.shape
        needs_caching = indices_compute is not None or cached_kv is not None
        
        # Case 1: Partial computation with caching
        if indices_compute is not None and cached_kv is not None:
            # Extract tokens to compute
            indices_expanded = indices_compute.unsqueeze(-1).expand(-1, -1, D)
            x_compute = torch.gather(x, 1, indices_expanded)  # [B, N_compute, D]
            
            # Compute Q/K/V only for selected tokens
            q_compute = self.norm_q(self.q(x_compute))
            k_new = self.norm_k(self.k(x_compute))
            v_new = self.v(x_compute)
            
            # Get frequencies for compute tokens
            # freqs is [N, 1, freq_dim], indices_compute is [B, N_compute]
            # We need [N_compute, 1, freq_dim] for rope_apply
            freqs_compute = freqs[indices_compute[0]]  # [N_compute, 1, freq_dim]
            
            # Apply RoPE
            q_compute = rope_apply(q_compute, freqs_compute, self.num_heads)
            k_new_rope = rope_apply(k_new, freqs_compute, self.num_heads)
            
            # Concatenate with cached K/V
            k_full = torch.cat([k_new_rope, cached_kv["k"]], dim=1)  # [B, N, D]
            v_full = torch.cat([v_new, cached_kv["v"]], dim=1)  # [B, N, D]
            
            # Attention with full context
            attn_out = self.attn(q_compute, k_full, v_full)  # [B, N_compute, D]
            
            # Prepare output: place computed values, reuse cached for skipped
            output = torch.zeros(B, N, D, dtype=x.dtype, device=x.device)
            output.scatter_(1, indices_expanded, self.o(attn_out))
            
            # Add cached outputs for skipped tokens
            # Note: cached_kv["output"] is already gathered [B, N_skip, D], no need to gather again
            if "output" in cached_kv and cached_kv["output"] is not None and indices_skip is not None:
                indices_skip_expanded = indices_skip.unsqueeze(-1).expand(-1, -1, D)
                output.scatter_(1, indices_skip_expanded, cached_kv["output"])
            
            # Return new K/V for caching (before RoPE for K)
            return output, {"k": k_new_rope, "v": v_new}
        
        # Case 2: Full computation
        else:
            q = self.norm_q(self.q(x))
            k = self.norm_k(self.k(x))
            v = self.v(x)
            q = rope_apply(q, freqs, self.num_heads)
            k = rope_apply(k, freqs, self.num_heads)
            
            # Compute attention (with or without weights)
            if return_attention_weights:
                x, attn_weights = self.attn(q, k, v, return_attention_weights=True, attention_mask_indices=attention_mask_indices)
            else:
                x = self.attn(q, k, v, return_attention_weights=False)
            
            output = self.o(x)
            
            # Return with or without cache depending on needs_caching
            if needs_caching:
                if return_attention_weights:
                    return output, {"k": k, "v": v}, attn_weights
                else:
                    return output, {"k": k, "v": v}
            else:
                if return_attention_weights:
                    return output, attn_weights
                else:
                    return output


class CrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6, has_image_input: bool = False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
        self.has_image_input = has_image_input
        if has_image_input:
            self.k_img = nn.Linear(dim, dim)
            self.v_img = nn.Linear(dim, dim)
            self.norm_k_img = RMSNorm(dim, eps=eps)
            
        self.attn = AttentionModule(self.num_heads)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        if self.has_image_input:
            img = y[:, :257]
            ctx = y[:, 257:]
        else:
            ctx = y
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(ctx))
        v = self.v(ctx)
        x = self.attn(q, k, v)
        if self.has_image_input:
            k_img = self.norm_k_img(self.k_img(img))
            v_img = self.v_img(img)
            y = flash_attention(q, k_img, v_img, num_heads=self.num_heads)
            x = x + y
        return self.o(x)


class GateModule(nn.Module):
    def __init__(self,):
        super().__init__()

    def forward(self, x, gate, residual):
        return x + gate * residual

class DiTBlock(nn.Module):
    def __init__(self, has_image_input: bool, dim: int, num_heads: int, ffn_dim: int, eps: float = 1e-6, block_idx: Optional[int] = None):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim

        self.self_attn = SelfAttention(dim, num_heads, eps)
        self.cross_attn = CrossAttention(
            dim, num_heads, eps, has_image_input=has_image_input)
        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(dim, eps=eps)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(
            approximate='tanh'), nn.Linear(ffn_dim, dim))
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        self.gate = GateModule()
        
        # PAB (Pyramid Attention Broadcast) caching
        self.block_idx = block_idx
        self.spatial_last = None
        self.spatial_count = 0
        self.cross_last = None
        self.cross_count = 0
        self.mlp_count = 0
        
        # AdaCache caching
        self.ada_spatial_cache = None  # Cache for spatial attention output
        self.ada_cross_mlp_cache = None  # Cache for cross-attn + MLP combined
        self.ada_prev_rate = 1  # Previous cache rate
        self.ada_next_step = 2  # Next step to recompute
        self.ada_prev_motion = 1.0  # Previous motion score (for MoReg)
        self.ada_recomputed_steps = []  # Track which steps recomputed
    
    def set_spatial_last(self, last_out: torch.Tensor):
        self.spatial_last = last_out
    
    def set_cross_last(self, last_out: torch.Tensor):
        self.cross_last = last_out

    def forward(self, x, context, t_mod, freqs, indices_compute=None, indices_skip=None, cached_kv=None, return_attention_weights=False, attention_mask_indices=None, timestep=None, all_timesteps=None):
        """
        Forward with optional token-level caching and PAB support.
        
        Args:
            x: Input [B, N, D]
            context: Cross-attention context
            t_mod: Time modulation
            freqs: RoPE frequencies
            indices_compute: Optional indices for partial computation
            indices_skip: Optional indices for tokens to skip (cached)
            cached_kv: Optional cached K/V/output dict
            return_attention_weights: If True, return attention weights from self_attn (only layer 0)
            timestep: Current timestep for PAB (integer)
            all_timesteps: All timesteps list for PAB MLP
        
        Returns:
            output: [B, N, D] (if no caching)
            OR (output, kv_cache): tuple if caching is enabled
            OR (output, kv_cache, attention_weights): if return_attention_weights=True
        """
        # Determine if we need caching
        needs_caching = indices_compute is not None or cached_kv is not None
        
        has_seq = len(t_mod.shape) == 4
        chunk_dim = 2 if has_seq else 1
        # msa: multi-head self-attention  mlp: multi-layer perceptron
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(6, dim=chunk_dim)
        if has_seq:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                shift_msa.squeeze(2), scale_msa.squeeze(2), gate_msa.squeeze(2),
                shift_mlp.squeeze(2), scale_mlp.squeeze(2), gate_mlp.squeeze(2),
            )
        
        # PAB: Check if should broadcast spatial/self-attention
        broadcast_spatial = False
        if enable_pab() and timestep is not None:
            broadcast_spatial, self.spatial_count = if_broadcast_spatial(int(timestep), self.spatial_count)
        
        # Self-attention with PAB broadcasting or token caching
        if broadcast_spatial and self.spatial_last is not None:
            # PAB: Reuse cached spatial attention result
            attn_out = self.spatial_last
            kv_cache = None
            attn_weights = None
        else:
            # Normal computation or token-level caching
            input_x = modulate(self.norm1(x), shift_msa, scale_msa)
            attn_result = self.self_attn(input_x, freqs, indices_compute, indices_skip, cached_kv, return_attention_weights, attention_mask_indices)
        
        # Unpack attention result
        attn_weights = None
        if not broadcast_spatial:
            # Only unpack if we computed (not broadcasted)
            if return_attention_weights:
                if needs_caching:
                    attn_out, kv_cache, attn_weights = attn_result
                else:
                    attn_out, attn_weights = attn_result
                    kv_cache = None
            else:
                if needs_caching:
                    attn_out, kv_cache = attn_result
                else:
                    attn_out = attn_result
                    kv_cache = None
            
            # PAB: Cache the result for future broadcasting
            if enable_pab():
                self.set_spatial_last(attn_out)
        
        x = self.gate(x, gate_msa, attn_out)
        
        # PAB: Check if should broadcast cross-attention
        broadcast_cross = False
        if enable_pab() and timestep is not None:
            broadcast_cross, self.cross_count = if_broadcast_cross(int(timestep), self.cross_count)
        
        # Cross-attention with PAB broadcasting
        if broadcast_cross and self.cross_last is not None:
            # PAB: Reuse cached cross attention result
            x = x + self.cross_last
        else:
            # Normal cross-attention computation
            cross_out = self.cross_attn(self.norm3(x), context)
            
            # PAB: Cache the result
            if enable_pab():
                self.set_cross_last(cross_out)
            
            x = x + cross_out
        
        # PAB: Check if should broadcast MLP/FFN
        broadcast_mlp = False
        broadcast_next_mlp = False
        broadcast_range = None
        if enable_pab() and timestep is not None and all_timesteps is not None and self.block_idx is not None:
            broadcast_mlp, self.mlp_count, broadcast_next_mlp, broadcast_range = if_broadcast_mlp(
                int(timestep),
                self.mlp_count,
                self.block_idx,
                all_timesteps.tolist() if isinstance(all_timesteps, torch.Tensor) else all_timesteps,
                is_temporal=False,  # This is spatial block
            )
        
        # FFN with PAB broadcasting or token caching
        if broadcast_mlp and broadcast_range is not None:
            # PAB: Reuse cached MLP output
            ff_output = get_mlp_output(
                broadcast_range,
                timestep=int(timestep),
                block_idx=self.block_idx,
                is_temporal=False,
            )
            x = self.gate(x, gate_mlp, ff_output)
        else:
            # Normal FFN computation or token-level caching
            input_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
            
            if (indices_compute is not None and indices_skip is not None and 
                cached_kv is not None and cached_kv.get("ffn_output") is not None):
                # Partial FFN computation
                B, N, D = x.shape
                indices_expanded = indices_compute.unsqueeze(-1).expand(-1, -1, D)
                x_compute = torch.gather(input_x, 1, indices_expanded)
                ffn_compute = self.ffn(x_compute)
                
                # Combine with cached FFN output
                ffn_output = torch.zeros(B, N, D, dtype=x.dtype, device=x.device)
                ffn_output.scatter_(1, indices_expanded, ffn_compute)
                
                # Add cached FFN for skipped tokens (use provided indices_skip)
                indices_skip_expanded = indices_skip.unsqueeze(-1).expand(-1, -1, D)
                cached_ffn = torch.gather(cached_kv["ffn_output"], 1, indices_skip_expanded)
                ffn_output.scatter_(1, indices_skip_expanded, cached_ffn)
                
                x = self.gate(x, gate_mlp, ffn_output)
                kv_cache["ffn_output"] = ffn_compute  # Store new FFN output
            else:
                # Full FFN computation
                ffn_output = self.ffn(input_x)
                x = self.gate(x, gate_mlp, ffn_output)
                if needs_caching and kv_cache is not None:
                    kv_cache["ffn_output"] = ffn_output
                
                # PAB: Cache MLP output if needed
                if enable_pab() and broadcast_next_mlp:
                    save_mlp_output(
                        timestep=int(timestep),
                        block_idx=self.block_idx,
                        ff_output=ffn_output,
                        is_temporal=False,
                    )
        
        # Store final output for next iteration (only if caching)
        if needs_caching and kv_cache is not None:
            kv_cache["output"] = x
        
        # Return format depends on whether caching is needed and if attention weights requested
        if return_attention_weights:
            if needs_caching:
                return x, kv_cache, attn_weights
            else:
                return x, attn_weights
        else:
            if needs_caching:
                return x, kv_cache
            else:
                return x


class MLP(torch.nn.Module):
    def __init__(self, in_dim, out_dim, has_pos_emb=False):
        super().__init__()
        self.proj = torch.nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim)
        )
        self.has_pos_emb = has_pos_emb
        if has_pos_emb:
            self.emb_pos = torch.nn.Parameter(torch.zeros((1, 514, 1280)))

    def forward(self, x):
        if self.has_pos_emb:
            x = x + self.emb_pos.to(dtype=x.dtype, device=x.device)
        return self.proj(x)


class Head(nn.Module):
    def __init__(self, dim: int, out_dim: int, patch_size: Tuple[int, int, int], eps: float):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_dim * math.prod(patch_size))
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, t_mod):
        if len(t_mod.shape) == 3:
            shift, scale = (self.modulation.unsqueeze(0).to(dtype=t_mod.dtype, device=t_mod.device) + t_mod.unsqueeze(2)).chunk(2, dim=2)
            x = (self.head(self.norm(x) * (1 + scale.squeeze(2)) + shift.squeeze(2)))
        else:
            shift, scale = (self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(2, dim=1)
            x = (self.head(self.norm(x) * (1 + scale) + shift))
        return x


class WanModel(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        in_dim: int,
        ffn_dim: int,
        out_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        patch_size: Tuple[int, int, int],
        num_heads: int,
        num_layers: int,
        has_image_input: bool,
        has_image_pos_emb: bool = False,
        has_ref_conv: bool = False,
        add_control_adapter: bool = False,
        in_dim_control_adapter: int = 24,
        seperated_timestep: bool = False,
        require_vae_embedding: bool = True,
        require_clip_embedding: bool = True,
        fuse_vae_embedding_in_latents: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.freq_dim = freq_dim
        self.has_image_input = has_image_input
        self.patch_size = patch_size
        self.seperated_timestep = seperated_timestep
        self.require_vae_embedding = require_vae_embedding
        self.require_clip_embedding = require_clip_embedding
        self.fuse_vae_embedding_in_latents = fuse_vae_embedding_in_latents

        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim)
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6))
        self.blocks = nn.ModuleList([
            DiTBlock(has_image_input, dim, num_heads, ffn_dim, eps, block_idx=i)
            for i in range(num_layers)
        ])
        self.head = Head(dim, out_dim, patch_size, eps)
        head_dim = dim // num_heads
        self.freqs = precompute_freqs_cis_3d(head_dim)

        if has_image_input:
            self.img_emb = MLP(1280, dim, has_pos_emb=has_image_pos_emb)  # clip_feature_dim = 1280
        if has_ref_conv:
            self.ref_conv = nn.Conv2d(16, dim, kernel_size=(2, 2), stride=(2, 2))
        self.has_image_pos_emb = has_image_pos_emb
        self.has_ref_conv = has_ref_conv
        if add_control_adapter:
            self.control_adapter = SimpleAdapter(in_dim_control_adapter, dim, kernel_size=patch_size[1:], stride=patch_size[1:])
        else:
            self.control_adapter = None

    def patchify(self, x: torch.Tensor, control_camera_latents_input: Optional[torch.Tensor] = None):
        x = self.patch_embedding(x)
        if self.control_adapter is not None and control_camera_latents_input is not None:
            y_camera = self.control_adapter(control_camera_latents_input)
            x = [u + v for u, v in zip(x, y_camera)]
            x = x[0].unsqueeze(0)
        grid_size = x.shape[2:]
        x = rearrange(x, 'b c f h w -> b (f h w) c').contiguous()
        return x, grid_size  # x, grid_size: (f, h, w)

    def unpatchify(self, x: torch.Tensor, grid_size: torch.Tensor):
        return rearrange(
            x, 'b (f h w) (x y z c) -> b c (f x) (h y) (w z)',
            f=grid_size[0], h=grid_size[1], w=grid_size[2], 
            x=self.patch_size[0], y=self.patch_size[1], z=self.patch_size[2]
        )

    def forward(self,
                x: torch.Tensor,
                timestep: torch.Tensor,
                context: torch.Tensor,
                clip_feature: Optional[torch.Tensor] = None,
                y: Optional[torch.Tensor] = None,
                use_gradient_checkpointing: bool = False,
                use_gradient_checkpointing_offload: bool = False,
                all_timesteps: Optional[torch.Tensor] = None,  # For PAB
                **kwargs,
                ):
        t = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, timestep))
        t_mod = self.time_projection(t).unflatten(1, (6, self.dim))
        context = self.text_embedding(context)
        
        if self.has_image_input:
            x = torch.cat([x, y], dim=1)  # (b, c_x + c_y, f, h, w)
            clip_embdding = self.img_emb(clip_feature)
            context = torch.cat([clip_embdding, context], dim=1)
        
        x, (f, h, w) = self.patchify(x)
        
        freqs = torch.cat([
            self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)
        
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)
            return custom_forward

        for block in self.blocks:
            if self.training and use_gradient_checkpointing:
                if use_gradient_checkpointing_offload:
                    with torch.autograd.graph.save_on_cpu():
                        result = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block),
                            x, context, t_mod, freqs,
                            use_reentrant=False,
                        )
                else:
                    result = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        x, context, t_mod, freqs,
                        use_reentrant=False,
                    )
                # Handle tuple or single tensor return
                x = result[0] if isinstance(result, tuple) else result
            else:
                # Pass timestep and all_timesteps for PAB support
                result = block(x, context, t_mod, freqs, 
                             timestep=timestep.item() if timestep.numel() == 1 else timestep[0].item(),
                             all_timesteps=all_timesteps)
                # Handle tuple or single tensor return
                x = result[0] if isinstance(result, tuple) else result

        x = self.head(x, t)
        x = self.unpatchify(x, (f, h, w))
        return x

    @staticmethod
    def state_dict_converter():
        return WanModelStateDictConverter()
    
    
class WanModelStateDictConverter:
    def __init__(self):
        pass

    def from_diffusers(self, state_dict):
        rename_dict = {
            "blocks.0.attn1.norm_k.weight": "blocks.0.self_attn.norm_k.weight",
            "blocks.0.attn1.norm_q.weight": "blocks.0.self_attn.norm_q.weight",
            "blocks.0.attn1.to_k.bias": "blocks.0.self_attn.k.bias",
            "blocks.0.attn1.to_k.weight": "blocks.0.self_attn.k.weight",
            "blocks.0.attn1.to_out.0.bias": "blocks.0.self_attn.o.bias",
            "blocks.0.attn1.to_out.0.weight": "blocks.0.self_attn.o.weight",
            "blocks.0.attn1.to_q.bias": "blocks.0.self_attn.q.bias",
            "blocks.0.attn1.to_q.weight": "blocks.0.self_attn.q.weight",
            "blocks.0.attn1.to_v.bias": "blocks.0.self_attn.v.bias",
            "blocks.0.attn1.to_v.weight": "blocks.0.self_attn.v.weight",
            "blocks.0.attn2.norm_k.weight": "blocks.0.cross_attn.norm_k.weight",
            "blocks.0.attn2.norm_q.weight": "blocks.0.cross_attn.norm_q.weight",
            "blocks.0.attn2.to_k.bias": "blocks.0.cross_attn.k.bias",
            "blocks.0.attn2.to_k.weight": "blocks.0.cross_attn.k.weight",
            "blocks.0.attn2.to_out.0.bias": "blocks.0.cross_attn.o.bias",
            "blocks.0.attn2.to_out.0.weight": "blocks.0.cross_attn.o.weight",
            "blocks.0.attn2.to_q.bias": "blocks.0.cross_attn.q.bias",
            "blocks.0.attn2.to_q.weight": "blocks.0.cross_attn.q.weight",
            "blocks.0.attn2.to_v.bias": "blocks.0.cross_attn.v.bias",
            "blocks.0.attn2.to_v.weight": "blocks.0.cross_attn.v.weight",
            "blocks.0.ffn.net.0.proj.bias": "blocks.0.ffn.0.bias",
            "blocks.0.ffn.net.0.proj.weight": "blocks.0.ffn.0.weight",
            "blocks.0.ffn.net.2.bias": "blocks.0.ffn.2.bias",
            "blocks.0.ffn.net.2.weight": "blocks.0.ffn.2.weight",
            "blocks.0.norm2.bias": "blocks.0.norm3.bias",
            "blocks.0.norm2.weight": "blocks.0.norm3.weight",
            "blocks.0.scale_shift_table": "blocks.0.modulation",
            "condition_embedder.text_embedder.linear_1.bias": "text_embedding.0.bias",
            "condition_embedder.text_embedder.linear_1.weight": "text_embedding.0.weight",
            "condition_embedder.text_embedder.linear_2.bias": "text_embedding.2.bias",
            "condition_embedder.text_embedder.linear_2.weight": "text_embedding.2.weight",
            "condition_embedder.time_embedder.linear_1.bias": "time_embedding.0.bias",
            "condition_embedder.time_embedder.linear_1.weight": "time_embedding.0.weight",
            "condition_embedder.time_embedder.linear_2.bias": "time_embedding.2.bias",
            "condition_embedder.time_embedder.linear_2.weight": "time_embedding.2.weight",
            "condition_embedder.time_proj.bias": "time_projection.1.bias",
            "condition_embedder.time_proj.weight": "time_projection.1.weight",
            "patch_embedding.bias": "patch_embedding.bias",
            "patch_embedding.weight": "patch_embedding.weight",
            "scale_shift_table": "head.modulation",
            "proj_out.bias": "head.head.bias",
            "proj_out.weight": "head.head.weight",
        }
        state_dict_ = {}
        for name, param in state_dict.items():
            if name in rename_dict:
                state_dict_[rename_dict[name]] = param
            else:
                name_ = ".".join(name.split(".")[:1] + ["0"] + name.split(".")[2:])
                if name_ in rename_dict:
                    name_ = rename_dict[name_]
                    name_ = ".".join(name_.split(".")[:1] + [name.split(".")[1]] + name_.split(".")[2:])
                    state_dict_[name_] = param
        if hash_state_dict_keys(state_dict) == "cb104773c6c2cb6df4f9529ad5c60d0b":
            config = {
                "model_type": "t2v",
                "patch_size": (1, 2, 2),
                "text_len": 512,
                "in_dim": 16,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "window_size": (-1, -1),
                "qk_norm": True,
                "cross_attn_norm": True,
                "eps": 1e-6,
            }
        else:
            config = {}
        return state_dict_, config
    
    def from_civitai(self, state_dict):
        state_dict = {name: param for name, param in state_dict.items() if not name.startswith("vace")}
        if hash_state_dict_keys(state_dict) == "9269f8db9040a9d860eaca435be61814":
            config = {
                "has_image_input": False,
                "patch_size": [1, 2, 2],
                "in_dim": 16,
                "dim": 1536,
                "ffn_dim": 8960,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 12,
                "num_layers": 30,
                "eps": 1e-6
            }
        elif hash_state_dict_keys(state_dict) == "aafcfd9672c3a2456dc46e1cb6e52c70":
            config = {
                "has_image_input": False,
                "patch_size": [1, 2, 2],
                "in_dim": 16,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "eps": 1e-6
            }
        elif hash_state_dict_keys(state_dict) == "6bfcfb3b342cb286ce886889d519a77e":
            config = {
                "has_image_input": True,
                "patch_size": [1, 2, 2],
                "in_dim": 36,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "eps": 1e-6
            }
        elif hash_state_dict_keys(state_dict) == "6d6ccde6845b95ad9114ab993d917893":
            config = {
                "has_image_input": True,
                "patch_size": [1, 2, 2],
                "in_dim": 36,
                "dim": 1536,
                "ffn_dim": 8960,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 12,
                "num_layers": 30,
                "eps": 1e-6
            }
        elif hash_state_dict_keys(state_dict) == "6bfcfb3b342cb286ce886889d519a77e":
            config = {
                "has_image_input": True,
                "patch_size": [1, 2, 2],
                "in_dim": 36,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "eps": 1e-6
            }
        elif hash_state_dict_keys(state_dict) == "349723183fc063b2bfc10bb2835cf677":
            # 1.3B PAI control
            config = {
                "has_image_input": True,
                "patch_size": [1, 2, 2],
                "in_dim": 48,
                "dim": 1536,
                "ffn_dim": 8960,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 12,
                "num_layers": 30,
                "eps": 1e-6
            }
        elif hash_state_dict_keys(state_dict) == "efa44cddf936c70abd0ea28b6cbe946c":
            # 14B PAI control
            config = {
                "has_image_input": True,
                "patch_size": [1, 2, 2],
                "in_dim": 48,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "eps": 1e-6
            }
        elif hash_state_dict_keys(state_dict) == "3ef3b1f8e1dab83d5b71fd7b617f859f":
            config = {
                "has_image_input": True,
                "patch_size": [1, 2, 2],
                "in_dim": 36,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "eps": 1e-6,
                "has_image_pos_emb": True
            }
        elif hash_state_dict_keys(state_dict) == "70ddad9d3a133785da5ea371aae09504":
            # 1.3B PAI control v1.1
            config = {
                "has_image_input": True,
                "patch_size": [1, 2, 2],
                "in_dim": 48,
                "dim": 1536,
                "ffn_dim": 8960,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 12,
                "num_layers": 30,
                "eps": 1e-6,
                "has_ref_conv": True
            }
        elif hash_state_dict_keys(state_dict) == "26bde73488a92e64cc20b0a7485b9e5b":
            # 14B PAI control v1.1
            config = {
                "has_image_input": True,
                "patch_size": [1, 2, 2],
                "in_dim": 48,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "eps": 1e-6,
                "has_ref_conv": True
            }
        elif hash_state_dict_keys(state_dict) == "ac6a5aa74f4a0aab6f64eb9a72f19901":
            # 1.3B PAI control-camera v1.1
            config = {
                "has_image_input": True,
                "patch_size": [1, 2, 2],
                "in_dim": 32,
                "dim": 1536,
                "ffn_dim": 8960,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 12,
                "num_layers": 30,
                "eps": 1e-6,
                "has_ref_conv": False,
                "add_control_adapter": True,
                "in_dim_control_adapter": 24,
            }
        elif hash_state_dict_keys(state_dict) == "b61c605c2adbd23124d152ed28e049ae":
            # 14B PAI control-camera v1.1
            config = {
                "has_image_input": True,
                "patch_size": [1, 2, 2],
                "in_dim": 32,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "eps": 1e-6,
                "has_ref_conv": False,
                "add_control_adapter": True,
                "in_dim_control_adapter": 24,
            }
        elif hash_state_dict_keys(state_dict) == "1f5ab7703c6fc803fdded85ff040c316":
            # Wan-AI/Wan2.2-TI2V-5B
            config = {
                "has_image_input": False,
                "patch_size": [1, 2, 2],
                "in_dim": 48,
                "dim": 3072,
                "ffn_dim": 14336,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 48,
                "num_heads": 24,
                "num_layers": 30,
                "eps": 1e-6,
                "seperated_timestep": True,
                "require_clip_embedding": False,
                "require_vae_embedding": False,
                "fuse_vae_embedding_in_latents": True,
            }
        elif hash_state_dict_keys(state_dict) == "5b013604280dd715f8457c6ed6d6a626":
            # Wan-AI/Wan2.2-I2V-A14B
            config = {
                "has_image_input": False,
                "patch_size": [1, 2, 2],
                "in_dim": 36,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "eps": 1e-6,
                "require_clip_embedding": False,
            }
        else:
            config = {}
        return state_dict, config
