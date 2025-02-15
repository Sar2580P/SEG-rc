import torch.nn.functional as F
import math
from typing import Any, Callable, Optional
import torch
from torch import Tensor
from functools import lru_cache

@lru_cache(maxsize=128)
def get_gaussian_kernel_cached(
    kernel_size: int, 
    sigma: float, 
    C: int, 
    device_str: str, 
    dtype_str: str
) -> Tensor:
    """
    Computes and caches a 2D Gaussian kernel.
    
    :param kernel_size: Size of the kernel (assumed odd).
    :param sigma: Standard deviation for the Gaussian.
    :param C: Number of channels; kernel is expanded to shape (C, 1, kernel_size, kernel_size).
    :param device_str: Device as a string (e.g., "cpu" or "cuda:0").
    :param dtype_str: Data type as a string (e.g., "float32" or "float64").
    :return: A tensor of shape (C, 1, kernel_size, kernel_size) representing the Gaussian kernel.
    """
    device = torch.device(device_str)
    dtype = getattr(torch, dtype_str)
    
    ksize_half = (kernel_size - 1) * 0.5
    x = torch.linspace(-ksize_half, ksize_half, steps=kernel_size, device=device, dtype=dtype)
    pdf = torch.exp(-0.5 * (x / sigma).pow(2))
    x_kernel = pdf / pdf.sum()
    # Outer product to form 2D kernel
    kernel2d = torch.mm(x_kernel[:, None], x_kernel[None, :])
    # Expand for each channel: shape becomes (C, 1, kernel_size, kernel_size)
    kernel2d = kernel2d.expand(C, 1, kernel_size, kernel_size)
    return kernel2d

def get_gaussian_kernel_2d(
    kernel_size: int, 
    sigma: float, 
    C: int, 
    device: torch.device, 
    dtype: torch.dtype
) -> Tensor:
    """
    Wrapper for get_gaussian_kernel_cached that converts device and dtype to strings.
    
    :param kernel_size: Size of the Gaussian kernel.
    :param sigma: Standard deviation of the Gaussian.
    :param C: Number of channels.
    :param device: Torch device.
    :param dtype: Torch data type.
    :return: Gaussian kernel tensor of shape (C, 1, kernel_size, kernel_size).
    """
    return get_gaussian_kernel_cached(
        kernel_size, sigma, C, str(device), str(dtype).split('.')[-1]
    )

def gaussian_blur_2d(img: Tensor, kernel_size: int, sigma: float) -> Tensor:
    """
    Applies a 2D Gaussian blur to the input image using a cached kernel.
    
    :param img: Input tensor of shape (C, H, W).
    :param kernel_size: Size of the Gaussian kernel.
    :param sigma: Standard deviation of the Gaussian kernel.
    :return: Blurred image tensor.
    """
    height = img.shape[-1]
    # Ensure kernel_size is not larger than the image dimension and is odd.
    kernel_size = min(kernel_size, height - (height % 2 - 1))
    
    C = img.shape[-3]
    kernel2d = get_gaussian_kernel_2d(kernel_size, sigma, C, img.device, img.dtype)
    
    padding = [kernel_size // 2] * 4
    img_padded = F.pad(img, padding, mode="reflect")
    return F.conv2d(img_padded, kernel2d, groups=C)


#______________________________________________________________________________________________________
def ema_smoothing_time_dependent(
    query: torch.Tensor,
    alpha_fn: Optional[Callable[[int, int], float]] = None,
) -> torch.Tensor:
    """
    Applies EMA smoothing along the token dimension (index 2) for a given tensor,
    but allows alpha to vary with time (i.e., alpha depends on t).

    Each (batch, attn_head, embed_dim) sequence is processed independently along
    the token dimension (index 2). The recurrence for t >= 1 is:
        y_t = alpha_t * y_{t-1} + (1 - alpha_t) * x_t,
    where alpha_t = alpha_fn(t, T).

    :param query: Input tensor of shape (batch, attn_heads, token_count, embed_dim).
    :param alpha_fn: A callable that takes (t, T) and returns alpha_t (float).
                     If None, we default to a constant alpha = 0.9.
    :return: A tensor of the same shape as 'query', where each sequence has been
             smoothed along the token dimension with a time-dependent alpha.
    """
    out = query.clone()
    B, H, T, D = query.shape

    # If no alpha_fn is provided, use a constant alpha = 0.9
    if alpha_fn is None:
        alpha_fn = lambda t, T: 0.9

    # Initialize the first token as is
    # out[:, :, 0, :] = query[:, :, 0, :]  # (already done by clone)

    # Loop over token dimension
    for t in range(1, T):
        alpha_t = alpha_fn(t=t, T=T)
        out[:, :, t, :] = alpha_t * out[:, :, t - 1, :] + (1 - alpha_t) * query[:, :, t, :]

    return out


def alpha_increasing(
    t: torch.Tensor, 
    T: torch.Tensor, 
    alpha_start: torch.Tensor = torch.tensor(0.75),
    alpha_end: torch.Tensor = torch.tensor(0.99),
    mode: str = "linear"
) -> torch.Tensor:
    """
    Computes a time-dependent alpha value as a torch scalar.
    The value monotonically increases from alpha_start at t=0 to alpha_end at t=T-1.
    
    The final function is:
        alpha(t) = alpha_start + (alpha_end - alpha_start)*ratio_f,
    where ratio_f depends on the scheduling mode:
    
      - "linear":    ratio_f = t / (T - 1)
      - "quadratic": ratio_f = (t / (T - 1))^2
      - "cosine":    ratio_f = (1 - cos(pi * t / (T - 1)))/2
    
    :param t: Current time step (torch scalar), 0 <= t <= T-1.
    :param T: Total time steps (torch scalar, T > 1).
    :param alpha_start: Initial alpha value at t=0.
    :param alpha_end: Final alpha value at t=T-1.
    :param mode: One of "linear", "quadratic", or "cosine".
    :return: A torch scalar representing alpha(t).
    """
    # Ensure T > 1 to avoid division by zero.
    if T <= 1:
        return alpha_start

    ratio = t / (T - 1)  
    if mode == "linear":
        ratio_f = ratio
    elif mode == "quadratic":
        ratio_f = ratio**2
    elif mode == "cosine":
        ratio_f = (1 - torch.cos(torch.tensor(torch.pi) * ratio)) / 2
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    return alpha_start + (alpha_end - alpha_start) * ratio_f


#______________________________________________________________________________________________________
def time_dependent_scaling(
    t: float, 
    T: float, 
    query: torch.Tensor,
    scheme: str = "linear", 
    f0: float = 1.5, 
    **kwargs: Any
) -> float:
    """
    Computes a time-dependent scaling factor for softmax based on the current time stamp.
    The final scaling is: scale(t) = sqrt(d) * f(t), where f(t) anneals from f0 at t=0 to 1 at t=T.
    
    Supported schemes:
      - "linear": f(t) = 1 + (f0 - 1) * (1 - t/T)
      - "cosine": f(t) = 1 + (f0 - 1) * 0.5 * (1 + cos(pi * t / T))
      - "exponential": f(t) = 1 + (f0 - 1) * exp(-lambda_ * t / T) (default lambda_=5)
    
    :param t: Current time step (must be in [0, T]).
    :param T: Total time steps.
    :param d: Dimension size (e.g., embedding dimension).
    :param scheme: Scheduling scheme name.
    :param f0: Initial multiplier at t=0 (typically > 1).
    :param kwargs: Additional parameters for specific schemes.
        For "exponential": lambda_ (float), default is 5.
        For "sigmoid": k (float), default is 10.
        For "polynomial": p (float), default is 2.
    :return: The overall scaling factor (float).
    """
    # Clamp t between 0 and T
    t = max(0.0, min(t, T))
    
    if scheme == "linear":
        f_t = 1 + (f0 - 1) * (1 - t / T)
    elif scheme == "cosine":
        f_t = 1 + (f0 - 1) * 0.5 * (1 + math.cos(math.pi * t / T))
    elif scheme == "exponential":
        lambda_ = kwargs.get("lambda_", 10)
        f_t = 1 + (f0 - 1) * math.exp(-lambda_ * t / T)
    else:
        raise ValueError(f"Unknown scheme: {scheme}")
    
    dim = query.shape[-1]
    return math.sqrt(dim) * f_t
