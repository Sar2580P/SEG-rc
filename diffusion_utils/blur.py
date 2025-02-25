import torch.nn.functional as F
import math
from typing import Any, Callable, Optional
import torch
from torch import Tensor
from functools import lru_cache

@lru_cache(128)
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
# def ema_smoothing_time_dependent(
#     query: torch.Tensor,
#     alpha_fn: Optional[Callable[[int, int], float]] = None,
# ) -> torch.Tensor:
#     """
#     Applies EMA smoothing along the token dimension (index 2) for a given tensor,
#     but allows alpha to vary with time (i.e., alpha depends on t).

#     Each (batch, attn_head, embed_dim) sequence is processed independently along
#     the token dimension (index 2). The recurrence for t >= 1 is:
#         y_t = alpha_t * y_{t-1} + (1 - alpha_t) * x_t,
#     where alpha_t = alpha_fn(t, T).

#     :param query: Input tensor of shape (batch, attn_heads, token_count, embed_dim).
#     :param alpha_fn: A callable that takes (t, T) and returns alpha_t (float).
#                      If None, we default to a constant alpha = 0.9.
#     :return: A tensor of the same shape as 'query', where each sequence has been
#              smoothed along the token dimension with a time-dependent alpha.
#     """
#     out = query.clone()
#     B, H, T, D = query.shape

#     # If no alpha_fn is provided, use a constant alpha = 0.9
#     if alpha_fn is None:
#         alpha_fn = lambda t, T: 0.9

#     # Initialize the first token as is
#     # out[:, :, 0, :] = query[:, :, 0, :]  # (already done by clone)

#     # Loop over token dimension
#     for t in range(1, T):
#         alpha_t = alpha_fn(t=t, T=T)
#         out[:, :, t, :] = (alpha_t) * out[:, :, t - 1, :] + (1-alpha_t) * query[:, :, t, :]
#
    # return out

def ema_vectorized(x: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    Apply Exponential Moving Average (EMA) along the token dimension (dim=2).

    Args:
        x (torch.Tensor): Input tensor of shape (batch, attn_heads, tokens, embedding_dim).
        alpha (float): Smoothing factor (0 < alpha < 1).

    Returns:
        torch.Tensor: EMA-applied tensor of shape (batch, attn_heads, tokens, embedding_dim).
    """
    # Ensure x is of shape (B, H, T, E)
    B, H, T, E = x.shape

    # Flip weights to align with causal computation
    weights = get_ema_weights(alpha, T, device=x.device)

    # Apply EMA along token dimension (dim=2)
    ema = torch.cumsum(weights.view(1, 1, -1, 1) * x.flip(2), dim=2).flip(2)

    return ema

@lru_cache(128)
def get_ema_weights(alpha, T, device=None):
    """
    Compute EMA weights for a given smoothing factor and time steps.

    Args:
        alpha (float): Smoothing factor (0 < alpha < 1).
        T (int): Number of time steps.
        device (torch.device, optional): Torch device. Defaults to None.

    Returns:
        torch.Tensor: EMA weights tensor of shape (T,).
    """
    weights = torch.pow(1 - alpha, torch.arange(T, device=device))
    return alpha * weights.flip(0)


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
        f_t = 1 + (f0 - 1) * 0.1234 * (1 + math.cos(math.pi * t / T))
    elif scheme == "exponential":
        lambda_ = kwargs.get("lambda_", 5)
        f_t = 1 + (f0 - 1) * math.exp(-lambda_ * t / T)
    else:
        raise ValueError(f"Unknown scheme: {scheme}")

    dim = query.shape[-1]
    return (math.sqrt(dim) * f_t)**-1

# # #______________________________________________________________________________________________________
def compute_integral_image_for_kernel(x: torch.Tensor, k: int) -> torch.Tensor:
    """Compute the integral image for x (B, C, T, E, D) using a box filter of size k."""
    half_k = k // 2
    B, C, E, D = x.shape
    N = B * C
    x_flat = x.contiguous().view(N, E, D)

    # Pad using 'constant' mode, which is generally faster than 'replicate'
    x_padded = F.pad(x_flat, pad=(half_k, half_k, half_k, half_k), mode='constant', value=0)

    E_p, D_p = E + 2 * half_k, D + 2 * half_k
    S_flat = x_padded.new_zeros((N, E_p + 1, D_p + 1))

    # Compute integral image using in-place cumulative sum
    torch.cumsum(x_padded, dim=1, out=S_flat[:, 1:, 1:])
    torch.cumsum(S_flat[:, 1:, 1:], dim=2, out=S_flat[:, 1:, 1:])

    return S_flat.view(B, C, E_p + 1, D_p + 1)

def compute_box_blur(x: torch.Tensor, k: int) -> torch.Tensor:
    """Compute the box-blur image using the integral image method."""
    half_k = k // 2
    S = compute_integral_image_for_kernel(x, k)  # Shape (B, C, T, E+2*half_k+1, D+2*half_k+1)
    B, C, S_E, S_D = S.shape
    _,  _, E, D = x.shape

    # Box filter sum calculation
    bottom_right = S[..., half_k + 1 : half_k + 1 + E, half_k + 1 : half_k + 1 + D]
    top_right = S[..., half_k + 1 : half_k + 1 + E, :-k]
    bottom_left = S[..., :-k, half_k + 1 : half_k + 1 + D]
    top_left = S[..., :-k, :-k]

    # Compute average for each k×k region
    return (bottom_right - top_right - bottom_left + top_left) / (k * k)

def interpolated_box_blur(query: torch.Tensor, kernel_size: int, alpha: float) -> torch.Tensor:
    """Interpolates between original image and its box-blurred version."""
    blurred = compute_box_blur(query, kernel_size)
    return torch.lerp(query, blurred, alpha)  # Faster than manual linear interpolation

