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
import numpy as np
from diffusion_utils.blur import (gaussian_blur_2d, time_dependent_scaling , interpolated_box_blur,
                  ema_smoothing_time_dependent, alpha_increasing)
from functools import partial
logger = logging.get_logger(__name__)
from diffusion_utils.metrics import AttentionMetricsLogger

class SEGCFGSelfAttnProcessor:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self, curr_iter_idx:int, total_iter:int,blur_time_regions:List,
                do_cfg=True, inf_gaussian_blur_sigma_threshold=9999.0, metric_logger:AttentionMetricsLogger=None,
                 blurring_technique: str = "gaussian_3_10"):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")
        self.do_cfg = do_cfg
        self.inf_blur = False
        self.apply_blur = self.get_blur_method(blurring_technique ,
                                                gaussian_sigma_threshold = inf_gaussian_blur_sigma_threshold ,
                                                curr_iter_idx=curr_iter_idx, total_iter=total_iter)
        self.should_apply_smoothing = self._should_apply_smoothing(curr_iter_idx, total_iter, blur_time_regions)
        self.metric_logger = metric_logger


    def get_blur_method(self, blurring_technique:str, gaussian_sigma_threshold:float, curr_iter_idx:int, total_iter:int):
        self.blur_method_name = blurring_technique.split("_")[0]

        if self.blur_method_name == "gaussian":
            _, kernel_size, sigma = [x if i == 0 else int(x) for i, x in enumerate(blurring_technique.split("_"))]
            if sigma>gaussian_sigma_threshold:
                self.inf_blur = True
            else:
                if kernel_size==-1:
                    kernel_size = math.ceil(6 * sigma) + 1 - math.ceil(6 * sigma) % 2
                return partial(gaussian_blur_2d, kernel_size=kernel_size, sigma=sigma)
        elif self.blur_method_name == "ema":
            _, alpha_start, alpha_end, mode = [
                                            x if i in (0, 3) else float(x)
                                            for i, x in enumerate(blurring_technique.split("_"))
                                        ]

            alphaFunc = partial(alpha_increasing, alpha_start=alpha_start, alpha_end=alpha_end, mode=mode)
            return partial(ema_smoothing_time_dependent, alpha_fn=alphaFunc)
        elif self.blur_method_name == "temperatureAnnealing":
            _,  scheme, f0 = [x if i <2 else float(x) for i, x in enumerate(blurring_technique.split("_"))]
            return partial(time_dependent_scaling, t=curr_iter_idx, T = total_iter,  scheme=scheme, f0=f0)
        elif self.blur_method_name == "interpolatedBoxBlur":
            _, kernel_size , alpha = blurring_technique.split("_")
            kernel_size, alpha = (int)(kernel_size), (float)(alpha)
            return partial(interpolated_box_blur, kernel_size=kernel_size, alpha=alpha)

        else :
            raise ValueError(f"Blur method {self.blur_method_name} not supported")

    def _should_apply_smoothing(self, curr_t: int, total_t: int, regions: List[str], percent: float=35) -> bool:
        """
        Determines whether smoothing should be applied based on the current timestamp and regions.

        :param curr_t: Current timestamp (in range 0 to total_t-1)
        :param total_t: Total number of timestamps
        :param regions: List containing 'mid', 'begin', 'end' to specify applicable regions
        :param percent: Percentage value to define smoothing regions
        :return: True if curr_t lies within the specified regions, else False
        """
        portion = total_t * (percent / 100)
        half_portion = total_t * (percent / 200)
        mid_start = (total_t // 2) - half_portion
        mid_end = (total_t // 2) + half_portion
        apply_smoothing = False

        if 'begin' in regions and curr_t < portion:
            apply_smoothing = True
        elif 'mid' in regions and mid_start <= curr_t < mid_end:
            apply_smoothing = True
        elif 'end' in regions and curr_t >= (total_t - portion):
            apply_smoothing = True

        logger.info(f"Smoothing {'applied' if apply_smoothing else 'not applied'} "
                    f"at curr_t={curr_t}, total_t={total_t}, regions={regions}")

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
        attn_scale = None
        if self.should_apply_smoothing:
            if self.do_cfg:
                query_uncond, query_org, query_ptb = query.chunk(3)
                query_ptb = query_ptb.permute(0, 1, 3, 2).view(batch_size//3, attn.heads * head_dim, height, width)

                if not self.inf_blur:
                    if self.blur_method_name!= "temperatureAnnealing": query_ptb = self.apply_blur(query_ptb)
                    else: attn_scale = self.apply_blur(query=query_ptb)

                else:
                    query_ptb[:] = query_ptb.mean(dim=(-2, -1), keepdim=True)

                query_ptb = query_ptb.view(batch_size//3, attn.heads, head_dim, height * width).permute(0, 1, 3, 2)
                query = torch.cat((query_uncond, query_org, query_ptb), dim=0)
            else:
                query_org, query_ptb = query.chunk(2)
                query_ptb = query_ptb.permute(0, 1, 3, 2).view(batch_size//2, attn.heads * head_dim, height, width)

                if not self.inf_blur:

                    if self.blur_method_name!= "temperatureAnnealing": query_ptb = self.apply_blur(query_ptb)
                    else: attn_scale = self.apply_blur(query=query_ptb)
                else:
                    query_ptb[:] = query_ptb.mean(dim=(-2, -1), keepdim=True)

                query_ptb = query_ptb.view(batch_size//2, attn.heads, head_dim, height * width).permute(0, 1, 3, 2)
                query = torch.cat((query_org, query_ptb), dim=0)

        hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False, scale=attn_scale
            )


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

        if self.metric_logger is not None:
            self.metric_logger.log_metrics(Q1= residual, Q2=hidden_states)

        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states
