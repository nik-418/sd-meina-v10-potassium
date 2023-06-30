from potassium import Potassium, Request, Response
from diffusers import DiffusionPipeline, DDPMScheduler
import torch

app = Potassium("my_app")

# @app.init runs at startup, and loads models into the app's context
@app.init
def init():
    repo_id="Meina/MeinaUnreal_V3"
    ddpm = DDPMScheduler.from_pretrained(repo_id, subfolder="scheduler")

    model = DiffusionPipeline.from_pretrained(
        repo_id, 
        use_safetensors=True,
        torch_dtype=torch.float16,
        scheduler=ddpm,
        safety_checker = None,
    )

    context = {
        "model": model,
    }

    return context

# @app.handler runs for every call
@app.handler()
def handler(context: dict, request: Request) -> Response:
    model = context.get("model")
    model.enable_xformers_memory_efficient_attention() # only on gpu
    model.enable_sequential_cpu_offload() # without .to("cuda")

    prompt = request.json.get("prompt")
    negative_prompt = "(worst quality, low quality:1.4), monochrome, zombie, (interlocked fingers), cleavage, nudity, naked, nude"

    image = model(
        prompt=prompt,
        negative_prompt=negative_prompt,
        guidance_scale=7,
        num_inference_steps=25,
        width=512,
        height=768,
    ).images[0]

    # save image to jpg file
    image.save("image.jpg")

    outputs = model(prompt)

    return Response(
        json = {"outputs": outputs[0]}, 
        status=200
    )

if __name__ == "__main__":
    app.serve()
