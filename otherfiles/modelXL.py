import numpy as np
import torch
import cv2
from PIL import Image
from diffusers import (
    StableDiffusionXLControlNetXSPipeline,
    ControlNetXSAdapter,
    AutoencoderKL,
)

class ImageGenerator:
    def __init__(self, vae_model, controlnet_model, pipeline_model, conditioning_scale=0.5):
        self.conditioning_scale = conditioning_scale
        self.vae = AutoencoderKL.from_pretrained(vae_model, torch_dtype=torch.float16)
        self.controlnet = ControlNetXSAdapter.from_pretrained(controlnet_model, torch_dtype=torch.float16)
        self.pipe = StableDiffusionXLControlNetXSPipeline.from_pretrained(pipeline_model, controlnet=self.controlnet, torch_dtype=torch.float16)
      #  self.pipe.enable_model_cpu_offload()

    def load_and_process_image(self, pil_image):
        image = np.array(pil_image)
        image = cv2.Canny(image, 100, 200)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        return Image.fromarray(image)

    def generate_image(self, prompt, canny_image, height=None, width=None, num_inference_steps=50, guidance_scale=5.0,
                       negative_prompt=None, num_images_per_prompt=2, output_type="pil"):
        result = self.pipe(
            prompt=prompt,
            image=canny_image,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images_per_prompt,
            output_type=output_type,
            controlnet_conditioning_scale=self.conditioning_scale,
        )
        return result.images

def generate(pil_image, prompt, controlnet_model, height=None, width=None,
                             vae_model="madebyollin/sdxl-vae-fp16-fix", pipeline_model="stabilityai/stable-diffusion-xl-base-1.0",
                             num_inference_steps=50, guidance_scale=5.0, negative_prompt=None, num_images_per_prompt=1):
    generator = ImageGenerator(vae_model, controlnet_model, pipeline_model)
    canny_image = generator.load_and_process_image(pil_image)
    generated_images = generator.generate_image(prompt, canny_image, height, width,
                                                num_inference_steps, guidance_scale, negative_prompt, num_images_per_prompt)
    return generated_images

# Example usage
if __name__ == "__main__":
    controlnet_model = "UmerHA/Testing-ConrolNetXS-SDXL-canny"

    # Load a PIL image (replace with your actual image path)
    pil_image = Image.open("path/to/your/image.png")  # Replace with your actual image path
    prompt = "aerial view, a futuristic research complex in a bright foggy jungle, hard lighting"

    generated_image = generate_image_from_pil(pil_image, prompt, controlnet_model,
                                               height=512, width=512, num_inference_steps=50, guidance_scale=7.5)

    # Save or display the generated image
    generated_image.show()  # or generated_image.save("output.png")
