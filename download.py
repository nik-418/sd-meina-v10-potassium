# This file runs during container build time to get model weights built into the container

# In this example: A Huggingface Stable Diffusion custom model
from diffusers import DDPMScheduler, DiffusionPipeline, StableDiffusionUpscalePipeline
import torch

def download_model():
    # do a dry run of loading the huggingface model, which will download weights

    repo_id="Meina/MeinaUnreal_V3"

    ddpm = DDPMScheduler.from_pretrained(repo_id, subfolder="scheduler")

    DiffusionPipeline.from_pretrained(
        repo_id, 
        use_safetensors=True,
        torch_dtype=torch.float16,
        scheduler=ddpm,
        # safety_checker = None,
    )

    # model_id = "stabilityai/stable-diffusion-x4-upscaler"

    # StableDiffusionUpscalePipeline.from_pretrained(
    #     model_id,   
    #     revision="fp16",
    #     torch_dtype=torch.float16,
    # )

if __name__ == "__main__":
    download_model()
