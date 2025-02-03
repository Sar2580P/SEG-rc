from pipeline_seg import StableDiffusionXLSEGPipeline
import torch
from utils import read_yaml
import os
from typing import List
import matplotlib.pyplot as plt
from typing import List
from PIL import Image
from tqdm import tqdm

try:
    pipe = StableDiffusionXLSEGPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16
    )
    device="cuda"
    pipe = pipe.to(device)
    print("Model loaded successfully")
except Exception as e:
    print(f"Error: {e}")

config = read_yaml("config.yaml")
prompts = [
"",
]

PICS_SAVE_DIR = os.path.join(config['save_dir'], f"seed-{config['seed']}" ,
                         f"blur_regions-{'___'.join(config['blur_time_regions'])})",
                         f"seg_applied_layers-{'___'.join(config['seg_applied_layers'])}")
ATTN_SAVE_DIR = os.path.join(config['atten_save_dir'], f"seed-{config['seed']}" ,
                         f"blur_regions-{'___'.join(config['blur_time_regions'])})",
                         f"seg_applied_layers-{'___'.join(config['seg_applied_layers'])}")
os.makedirs(PICS_SAVE_DIR, exist_ok=True)
os.makedirs(ATTN_SAVE_DIR, exist_ok=True)



def create_plot(images: List[Image.Image], titles: List[str],
                rows: int, cols: int, save_path:str):
    """
    :param images: List of PIL images to display.
    :param titles: List of titles corresponding to each image.
    :param rows: Number of rows in the plot.
    :param cols: Number of columns in the plot.
    """
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flatten() if rows * cols > 1 else [axes]  # Flatten axes for easy iteration

    for i, ax in enumerate(axes):
        if i < len(images):
            ax.imshow(images[i])
            ax.set_title(titles[i], fontsize=10)
            ax.axis("off")
        else:
            ax.axis("off")  # Hide extra subplots if images are fewer than grid cells

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


if __name__=="__main__":
    for prompt in prompts:
        generator = torch.Generator(device="cuda").manual_seed(config["seed"])
        for guidance_scale in tqdm(config['guidance_scales'], desc="guidance_scales"):
            outputs =[]
            titles = []
            for seg_scale in config['seg_scales']:
                for seg_blur_sigma in config['seg_blur_sigmas']:
                    titles.append(f"seg_blur_sigma-{seg_blur_sigma}_seg_scale-{seg_scale}_guidance_scale-{guidance_scale}")
                    outputs += pipe(
                            [prompt],
                            num_inference_steps=config['num_inference_steps'],
                            guidance_scale=guidance_scale,
                            seg_scale=seg_scale,
                            seg_blur_sigma=seg_blur_sigma,
                            seg_applied_layers=config['seg_applied_layers'],
                            blur_time_regions=config['blur_time_regions'],
                            generator=generator,
                            save_attention_maps=config['save_attention_maps'],
                            save_path_attention_maps = os.path.join(ATTN_SAVE_DIR, f"seg_blur_sigma-{seg_blur_sigma}_seg_scale-{seg_scale}_guidance_scale-{guidance_scale}")\
                                                        if config['save_attention_maps'] else None

                        ).images
            save_path = os.path.join(PICS_SAVE_DIR, f"guidance_scales-{guidance_scale}.png")
            create_plot(outputs, titles,  rows=len(config['seg_scales']),
                        cols=len(config['seg_blur_sigmas']), save_path=save_path)

