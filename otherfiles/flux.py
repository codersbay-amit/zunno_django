import torch
from diffusers import FluxPipeline
from PIL import Image
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

class StableDiffusionImageGenerator:
    def __init__(self, model_name="CompVis/stable-diffusion-v1-4", device="cuda", torch_dtype=torch.float16):
        """
        Initializes the StableDiffusionImageGenerator with the specified model and precision.
        
        Args:
            model_name (str): The name of the model to load from Hugging Face.
            device (str): The device to run the model on, can be "cuda" or "cpu".
            torch_dtype (torch.dtype): The tensor precision to use for the model.
        """
        # Load the pre-trained model pipeline
        self.pipe = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch_dtype)
        self.pipe = self.pipe.to(device)
        self.pipe.enable_attention_slicing()  # This helps optimize memory usage

    def generate_image(self, prompt, seed=0, guidance_scale=7.5, num_inference_steps=50):
        """
        Generate an image based on the given prompt.

        Args:
            prompt (str): The text prompt to generate the image from.
            seed (int): Random seed for reproducibility.
            guidance_scale (float): The guidance scale for controlling the creativity vs. prompt alignment.
            num_inference_steps (int): The number of inference steps for generation.

        Returns:
            PIL.Image: The generated image.
        """
        # Set the seed for reproducibility
        generator = torch.manual_seed(seed)

        # Generate the image
        image = self.pipe(prompt, guidance_scale=guidance_scale, num_inference_steps=num_inference_steps, generator=generator).images[0]

        return image

class FluxImageGenerator:
    def __init__(self, model_name="black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16):
        """
        Initializes the FluxImageGenerator with the specified model and precision.
        """
        # Load the pre-trained model pipeline
        self.pipe = FluxPipeline.from_pretrained(model_name, torch_dtype=torch_dtype)
        self.pipe.enable_model_cpu_offload()  # Offload to CPU to save VRAM (if needed)

    def generate_image(self, prompt, guidance_scale=0.0, num_inference_steps=4, max_sequence_length=256, seed=0):
        """
        Generate an image based on the given prompt.

        Args:
            prompt (str): The text prompt to generate the image from.
            guidance_scale (float): The guidance scale for controlling the creativity vs. prompt alignment.
            num_inference_steps (int): The number of inference steps for generation.
            max_sequence_length (int): Maximum length of the sequence (e.g., for prompt tokenization).
            seed (int): Random seed for generation to ensure reproducibility.

        Returns:
            PIL.Image: The generated image.
        """
        # Set the seed for reproducibility
        generator = torch.Generator("cpu").manual_seed(seed)

        # Generate the image
        image = self.pipe(
            prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            max_sequence_length=max_sequence_length,
            generator=generator
        ).images[0]

        return image
