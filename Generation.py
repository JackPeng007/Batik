import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import time
from diffusers import DiffusionPipeline, StableDiffusionPipeline
import torch
from PIL import Image

lcm_speedup = False

model_id = "sd-legacy/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, safety_checker=None,
                                               requires_safety_checker=False)
pipe.to("cuda")

lora_path = r"Models\model.safetensors"
weight_name = 'model.safetensors'

pipe.load_lora_weights(pretrained_model_name_or_path_or_dict=lora_path, weight_name=weight_name, adapter_name="lora")

if lcm_speedup:
    lcm_lora_path = "Models\lcm_lora.safetensors"
    pipe.load_lora_weights(pretrained_model_name_or_path_or_dict=lcm_lora_path, weight_name="lcm_lora.safetensors",
                           adapter_name="lcm")
    pipe.set_adapters(["lora", "lcm"], adapter_weights=[1.0, 1.0])

if lcm_speedup:
    num_inference_steps = 8
    guidance_scale = 2
else:
    num_inference_steps = 30
    guidance_scale = 7.5

num_samples_per_prompt = 4

all_images = []


prompts = [
    "bird, A traditional Miao batik pattern depicting an elegant phoenix with elaborately detailed feather textures, gracefully interacting with stylized floral motifs and leaves, presented in black on a white background.",
]
output_size = (512, 512)
images_per_row = 4

for idx, text in enumerate(prompts):

    result = pipe(
        text,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        num_images_per_prompt=num_samples_per_prompt
    )

    generated_imgs = result.images

    # 构建拼接图像画布
    stitched_image = Image.new("RGB", (images_per_row * output_size[0], output_size[1]))

    for i, img in enumerate(generated_imgs[:images_per_row]):
        x_offset = i * output_size[0]
        resized_img = img.resize(output_size)
        stitched_image.paste(resized_img, (x_offset, 0))

    stitched_image.save(f"generated_grid_{idx+1}.png")
