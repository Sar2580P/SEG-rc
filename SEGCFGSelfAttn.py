import torch
import torch.nn.functional as F
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from diffusers.models.attention_processor import Attention, AttnProcessor2_0
from diffusers.models.lora import adjust_lora_scale_text_encoder
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import (
    USE_PEFT_BACKEND,
    deprecate,
    is_invisible_watermark_available,
    is_torch_xla_available,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)
import math
from blur import gaussian_blur_2d
logger = logging.get_logger(__name__)

class SEGCFGSelfAttnProcessor:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self, curr_iter_idx:int, total_iter:int,blur_time_regions:List,  save_attention_maps:bool=False,
                 blur_sigma=1.0, do_cfg=True, inf_blur_threshold=9999.0):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")
        self.blur_sigma = blur_sigma
        self.do_cfg = do_cfg
        if self.blur_sigma > inf_blur_threshold:
            self.inf_blur = True
        else:
            self.inf_blur = False

        self.should_apply_smoothing = self._should_apply_smoothing(curr_iter_idx, total_iter, blur_time_regions)
        self.save_attention_maps = save_attention_maps
        self.attention_maps = []

    def scaled_dot_product_attention_with_map(self, query, key, value, attention_mask=None, dropout_p=0.0):
        """
        Computes the attention map and returns both the attended output and the attention weights.

        Args:
            query (Tensor): Query tensor of shape [B, H, S, D]
            key (Tensor): Key tensor of shape [B, H, S, D]
            value (Tensor): Value tensor of shape [B, H, S, D]
            attention_mask (Tensor, optional): Mask tensor of shape [B, 1, S, S] or None
            dropout_p (float): Dropout probability

        Returns:
            hidden_states (Tensor): Attended output of shape [B, H, S, D]
            attention_map (Tensor): Attention weights of shape [B, H, S, S]
        """
        d_k = query.shape[-1]

        # Compute attention scores (logits)
        attn_logits = torch.matmul(query, key.transpose(-2, -1)) / d_k**0.5  # [B, H, S, S]

        # Apply attention mask if provided
        if attention_mask is not None:
            attn_logits += attention_mask  # Ensure masked positions are not attended

        # Compute attention weights (softmax over last dimension)
        attention_map = F.softmax(attn_logits, dim=-1)  # [B, H, S, S]
        attention_map = torch.mean(attention_map, dim=1)  # [B, S, S]
        attention_map = self.reduce_precision(attention_map)   # to reduce memory footprint for attention_map storage
        # Use PyTorch's built-in function for better efficiency
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=dropout_p, is_causal=False
        )  # [B, H, S, D]

        return hidden_states, attention_map
    
    def reduce_precision(self, attention_map: torch.Tensor) -> torch.Tensor:
        """
        Reduces the precision of the input tensor to unsigned 16-bit integer (uint16) to save memory.

        Args:
            attention_map (Tensor): Input tensor of shape [B, S, S], expected to be in range [0, 1].

        Returns:
            Tensor: Reduced-precision tensor of shape [B, S, S] in uint16 format.
        """
        if torch.any(torch.isnan(attention_map)):
            logger.warning("Attention map contains NaN values. Clipping to [0, 1] before scaling.")
            attention_map = attention_map.clamp(0, 1)
        scaled_attn_maps = (attention_map * 10000).clamp(0, 65535).to(torch.uint16)
        return scaled_attn_maps


    def _should_apply_smoothing(self, curr_t: int, total_t: int, regions: List[str]) -> bool:
        """
        Determines whether smoothing should be applied based on the current timestamp and regions.

        :param curr_t: Current timestamp (in range 0 to total_t-1)
        :param total_t: Total number of timestamps
        :param regions: List containing 'mid', 'begin', 'end' to specify applicable regions
        :return: True if curr_t lies within the specified regions, else False
        """
        third = total_t // 3
        apply_smoothing = False

        if 'begin' in regions and curr_t >= 2 * third:
            apply_smoothing = True
        elif 'mid' in regions and third <= curr_t < 2 * third:
            apply_smoothing = True
        elif 'end' in regions and curr_t < third:
            apply_smoothing = True

        logger.info(f"Smoothing {'applied' if apply_smoothing else 'not applied'} \
                    at curr_t={curr_t}, total_t={total_t}, regions={regions}")

        return apply_smoothing

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        height = width = math.isqrt(query.shape[2])
        if self.should_apply_smoothing:
            if self.do_cfg:
                query_uncond, query_org, query_ptb = query.chunk(3)
                query_ptb = query_ptb.permute(0, 1, 3, 2).view(batch_size//3, attn.heads * head_dim, height, width)

                if not self.inf_blur:
                    kernel_size = math.ceil(6 * self.blur_sigma) + 1 - math.ceil(6 * self.blur_sigma) % 2
                    query_ptb = gaussian_blur_2d(query_ptb, kernel_size, self.blur_sigma)
                else:
                    query_ptb[:] = query_ptb.mean(dim=(-2, -1), keepdim=True)

                query_ptb = query_ptb.view(batch_size//3, attn.heads, head_dim, height * width).permute(0, 1, 3, 2)
                query = torch.cat((query_uncond, query_org, query_ptb), dim=0)
            else:
                query_org, query_ptb = query.chunk(2)
                query_ptb = query_ptb.permute(0, 1, 3, 2).view(batch_size//2, attn.heads * head_dim, height, width)

                if not self.inf_blur:
                    kernel_size = math.ceil(6 * self.blur_sigma) + 1 - math.ceil(6 * self.blur_sigma) % 2
                    query_ptb = gaussian_blur_2d(query_ptb, kernel_size, self.blur_sigma)
                else:
                    query_ptb[:] = query_ptb.mean(dim=(-2, -1), keepdim=True)

                query_ptb = query_ptb.view(batch_size//2, attn.heads, head_dim, height * width).permute(0, 1, 3, 2)
                query = torch.cat((query_org, query_ptb), dim=0)

        if not self.save_attention_maps:
            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False,
            )
        else:
            hidden_states, attention_map = self.scaled_dot_product_attention_with_map(
                query, key, value, attention_mask, dropout_p=0.0
            )
            self.attention_maps.append(attention_map)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states
