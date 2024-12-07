import torch
import numpy as np
from PIL import Image
import cv2
from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline, AutoencoderKL
from diffusers.utils import load_image

class ControlNetImageGenerator:
    def __init__(self):
        # Load models
        self.controlnet = ControlNetModel.from_pretrained(
            "diffusers/controlnet-canny-sdxl-1.0",
            torch_dtype=torch.float16
         )
        self.vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
        self.pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            controlnet=self.controlnet,
            vae=self.vae,
            torch_dtype=torch.float16,
        )
        self.pipe.enable_model_cpu_offload()

    def preprocess_image(self, image):
        image = np.array(image)
        image = cv2.Canny(image, 80, 200)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        return Image.fromarray(image)

def generate_image_canny( prompt, negative_prompt, image, controlnet_conditioning_scale=0.5,steps=50):
        model=ControlNetImageGenerator()
        processed_image = model.preprocess_image(image)
        images = model.pipe(
            prompt,
        guidance_scale=9,       
        target_size =(1024,1024),
        num_inference_steps=steps, 
        negative_prompt=negative_prompt, 
        image=processed_image, 
        controlnet_conditioning_scale=controlnet_conditioning_scale,
        ).images
        return images

# Usage example
if __name__ == "__main__":
    generator = ControlNetImageGenerator()
    prompt = "aerial view, a futuristic research complex in a bright foggy jungle, hard lighting"
    negative_prompt = 'low quality, bad quality, sketches'
    image = load_image("https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/hf-logo.png")

    result_image = generator.generate_image(prompt, negative_prompt, image)
    result_image.save("hug_lab.png")
