from diffusion_utils.pipeline_seg import StableDiffusionXLSEGPipeline
import torch
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from omegaconf import OmegaConf

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

conf = OmegaConf.load('config.yaml')
config = OmegaConf.create(OmegaConf.to_yaml(conf, resolve=True))
    
os.makedirs(config['pics_save_dir'], exist_ok=True)
os.makedirs(config['atten_save_dir'], exist_ok=True)
os.makedirs(config['metric_save_dir'], exist_ok=True)



if __name__=="__main__":
    
    if config['guidance_scale']==0:
        config['cfg_prompts'] = [""]  # only one prompt for guidance_scale=0

    for idx, (prompt) in tqdm(enumerate(config['cfg_prompts']), desc = "Generating images for prompts..."):
        name = f"segScale-{config['seg_scale']}_guidanceScale-{config['guidance_scale']}/steps-{config['num_inference_steps']}_" + \
                (f"prompt-{idx}" if config['guidance_scale']>0 else "")

        generator = torch.Generator(device="cuda").manual_seed(config["seed"])
        output = pipe(
                [prompt],
                num_inference_steps=config['num_inference_steps'],
                guidance_scale=config['guidance_scale'],
                should_log_metrics = config['should_log_metrics'],
                metric_tracked_block = config['metric_tracked_block'],
                metric_save_dir = os.path.join(config['metric_save_dir'], name) ,
                seg_scale=config['seg_scale'],
                seg_applied_layers=config['seg_applied_layers'],
                blur_time_regions=config['blur_time_regions'],
                blurring_technique = config['blurring_technique'],
                generator=generator,
            ).images
        
        
        save_path = os.path.join(config['pics_save_dir'], f"{name}.png")
        
        # save the image
        plt.imsave(save_path, output[0])

