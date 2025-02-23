# Reproducibility Challenge: Smoothed Energy Guidance

This repository is part of the MLRC Reproducibility Challenge 2025. Our work reproduces and enhances the results from the NeurIPS 2024 paper *Smoothed Energy Guidance (SEG): Guiding Diffusion Models with Reduced Energy Curvature of Attention*.

## Overview

The original SEG paper faced several challenges:
- **Kernel Size Ablation:** No study was performed to identify the optimal kernel size.
- **Blurring Strategies:** Alternative blurring methods were unexplored, leading to unnecessary smoothing across all iterations. This not only reduced image clarity but also increased computational costs.

To address these issues, our approach:
- Conducts a comprehensive ablation study on kernel size selection.
- Implements efficient alternatives, including Exponential Moving Average (EMA) and BoxBlur using integral images.
- Achieves improved image quality and reduced computational overhead.

## Getting Started

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Sar2580P/SEG-rc/tree/main
   cd SEG-rc
   ```
2. **Install dependencies:**
   ```bash
   poetry install
   bash scripts/activate.sh
   ```
3. **Run experiments:**
   ```bash
   python main.py
   ```


## Configuration Parameters

This project is highly configurable via a [config](config.yaml) file. Below is a guide to the main parameters and their roles:

- **`seed`**:  
  Sets the random seed to ensure experiment reproducibility.

- **`num_inference_steps`**:  
  Defines the number of inference steps for the diffusion model during image generation.

- **`seg_applied_layers`**:  
  Specifies which layers of the model the Smoothed Energy Guidance (SEG) is applied to (e.g., `[mid]`, `[up, down]`).

- **`blur_time_regions`**:  
  Determines the phase of the denoising process where the blurring is applied (e.g., `[begin]`, `[begin mid end]`).

- **`metric_tracked_block`**:  
  Indicates which block of the model is used for tracking performance metrics (e.g., `mid`, `down`)

- **`blurring_technique`**:  
  Sets the blurring method and its parameters through a specific naming convention. The format differs based on the technique:

  - **BoxBlur**:  
    Use the format:  
    ``` 
    "interpolatedBoxBlur_kernelSize_alpha" 
    ```  
    where:  
    - **`kernelSize`** is the size of the box blur kernel.  
    - **`alpha`** is the interpolation strength controlling the extent of blurring.

  - **Gaussian Blur**:  
    Use the format:  
    ``` 
    "gaussian_kernelSize_sigma" 
    ```  
    where:  
    - **`kernelSize`** is the kernel size for the Gaussian blur. If set to `-1`, it is calculated as:  
      ```
      kernel_size = math.ceil(6 * sigma) + 1 - math.ceil(6 * sigma) % 2
      ```  
    - **`sigma`** is the standard deviation of the Gaussian function.

  - **EMA (Exponential Moving Average)**:  
    Use the format:  
    ``` 
    "ema_startAlpha_endAlpha_linear" 
    ```  
    where:  
    - **`startAlpha`** and **`endAlpha`** define the range between which the alpha value increases linearly during the process.

- **`guidance_scale`**:  
  The scale factor applied to guidance during generation, impacting how strongly the model follows the prompt.

- **`seg_scale`**:  
  A scaling factor specific to the Smoothed Energy Guidance, influencing its impact on the output.

- **`should_log_metrics`**:  
  A boolean flag that, when set to `True`, enables logging of performance metrics.

- **`pics_save_dir`** and **`metric_save_dir`**:  
  These directories are used to save generated images and metrics, respectively. Their paths are dynamically generated using parameter values to facilitate organized experiment tracking.

- **`cfg_prompts`**:  
  A list of textual prompts that direct the image generation process.

This structured configuration allows you to effortlessly experiment with different settings and blurring techniques while ensuring clear reproducibility and efficient experiment management.

## Results

Our experiments demonstrate:  
- **Enhanced image clarity.** ✨  
- **Lower computational costs.** ⚡  
- **Robust performance improvements across various settings.** 🚀

Happy experimenting! 🎉