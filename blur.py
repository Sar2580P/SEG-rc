import torch
import torch.nn.functional as F

# Cache for storing computed kernels
KERNEL_CACHE = {}

def gaussian_blur_2d(img, kernel_size, sigma):
    """
    Applies a 2D Gaussian blur to the input image.

    :param img: Input tensor of shape (C, H, W).
    :param kernel_size: Size of the Gaussian kernel.
    :param sigma: Standard deviation of the Gaussian kernel.
    :return: Blurred image tensor.
    """
    height = img.shape[-1]
    kernel_size = min(kernel_size, height - (height % 2 - 1))
    
    # Check if the kernel is already computed
    cache_key = (kernel_size, sigma, img.device, img.dtype)
    if cache_key in KERNEL_CACHE:
        kernel2d = KERNEL_CACHE[cache_key]
    else:
        
        ksize_half = (kernel_size - 1) * 0.5
        x = torch.linspace(-ksize_half, ksize_half, steps=kernel_size)
        pdf = torch.exp(-0.5 * (x / sigma).pow(2))
        x_kernel = pdf / pdf.sum()
        x_kernel = x_kernel.to(device=img.device, dtype=img.dtype)

        kernel2d = torch.mm(x_kernel[:, None], x_kernel[None, :])
        kernel2d = kernel2d.expand(img.shape[-3], 1, kernel2d.shape[0], kernel2d.shape[1])

        # Store in cache
        KERNEL_CACHE[cache_key] = kernel2d

    padding = [kernel_size // 2] * 4
    img = F.pad(img, padding, mode="reflect")
    img = F.conv2d(img, kernel2d, groups=img.shape[-3])

    return img
