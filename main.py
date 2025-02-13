from pipeline_seg import StableDiffusionXLSEGPipeline
import torch
from utils import read_yaml
import os
from typing import List
import matplotlib.pyplot as plt
from typing import List
from PIL import Image
from tqdm import tqdm
import textwrap

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


def create_plot(images, titles, rows, cols, save_path):
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3), constrained_layout=True)
    axes = axes.flatten() if rows * cols > 1 else [axes]  # Flatten axes for easy iteration

    for i, ax in enumerate(axes):
        if i < len(images):
            ax.imshow(images[i])
            ax.axis("off")

            # Wrap text to avoid horizontal overlap
            wrapped_title = textwrap.fill(titles[i], width=30)  # Adjust width for better readability

            # Place title as text below the image
            ax.text(0.5, -0.1, wrapped_title, fontsize=9, ha="center", va="top", transform=ax.transAxes, wrap=True)
        else:
            ax.axis("off")  # Hide extra subplots if images are fewer than grid cells

    # Adjust layout to prevent overlapping
    plt.subplots_adjust(top=0.95, bottom=0.05, wspace=0.3, hspace=0.6)  # Increase wspace & hspace
    plt.savefig(save_path, bbox_inches="tight", dpi=300)  # Save with proper bounding box
    plt.close()

config = {
'seed' : 4,
'num_inference_steps' : 30,
'seg_applied_layers' : ['down' , 'mid'],
'guidance_scales' : [0 , 3],
'blur_time_regions' : ['begin', 'mid'],
'seg_scales' : [0, 3],
'seg_blur_sigmas' : [0, 1 ,10, 10000],
'pics_save_dir' : 'results/pics',
'save_attention_maps' :  False,
'atten_save_dir' : 'results/attn_maps',
'sample_ct_attn_maps' :  3,
'should_log_metrics' : True
}

_config = read_yaml("config.yaml")
prompts = [
"a jellyfish playing the drums in an underwater concert",
]
config.update(_config)

PICS_SAVE_DIR = os.path.join(config['pics_save_dir'], f"seed-{config['seed']}" ,
                         f"blur_regions-{'___'.join(config['blur_time_regions'])})",
                         f"seg_applied_layers-{'___'.join(config['seg_applied_layers'])}")
ATTN_SAVE_DIR = os.path.join(config['atten_save_dir'], f"seed-{config['seed']}" ,
                         f"blur_regions-{'___'.join(config['blur_time_regions'])})",
                         f"seg_applied_layers-{'___'.join(config['seg_applied_layers'])}")
os.makedirs(PICS_SAVE_DIR, exist_ok=True)
os.makedirs(ATTN_SAVE_DIR, exist_ok=True)



if __name__=="__main__":
    for prompt in prompts:
        generator = torch.Generator(device="cuda").manual_seed(config["seed"])
        generator.seed()
        for guidance_scale in tqdm(config['guidance_scales'], desc="guidance_scales"):
            outputs =[]
            titles = []
            for seg_scale in config['seg_scales']:
                for seg_blur_sigma in config['seg_blur_sigmas']:
                    if seg_blur_sigma==0 and seg_scale>0:   # invalid case
                        continue
                    elif seg_scale==0 and guidance_scale==0:  #simplest  case, skipping...
                        continue
                    titles.append(f"seg_blur_sigma-{seg_blur_sigma}_seg_scale-{seg_scale}_guidance_scale-{guidance_scale}")
                    outputs += pipe(
                            [prompt],
                            num_inference_steps=config['num_inference_steps'],
                            guidance_scale=guidance_scale,
                            should_log_metrics = config['should_log_metrics'],
                            metric_tracked_block = config['metric_tracked_block'],
                            metric_save_dir = f"results/metrics/seed-{config['seed']}",
                            seg_scale=seg_scale,
                            seg_blur_sigma=seg_blur_sigma,
                            seg_applied_layers=config['seg_applied_layers'],
                            blur_time_regions=config['blur_time_regions'],
                            generator=generator,
                            save_attention_maps=config['save_attention_maps'],
                            save_path_attention_maps = os.path.join(ATTN_SAVE_DIR, f"seg_blur_sigma-{seg_blur_sigma}_seg_scale-{seg_scale}_guidance_scale-{guidance_scale}")\
                                                        if config['save_attention_maps'] else None ,
                            sample_ct_attn_maps = config['sample_ct_attn_maps']

                        ).images
            save_path = os.path.join(PICS_SAVE_DIR, f"guidance_scales-{guidance_scale}.png")
            create_plot(outputs, titles,  rows=len(config['seg_scales']),
                        cols=len(config['seg_blur_sigmas']), save_path=save_path)

