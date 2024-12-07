import torch
from diffusers import AutoPipelineForInpainting
from PIL import Image

class InpaintingModel:
    def __init__(self, model_name="diffusers/stable-diffusion-xl-1.0-inpainting-0.1", device="cuda", torch_dtype=torch.float16):
        """
        Initializes the inpainting model pipeline.
        Arguments:
            - model_name (str): Model identifier from Hugging Face.
            - device (str): Device to run on (default: 'cuda').
            - torch_dtype (torch.dtype): Torch data type (default: torch.float16 for memory efficiency).
        """
        self.device = device
        self.model_name = model_name
        self.torch_dtype = torch_dtype
        self.pipe = None  # Model will be loaded when needed

    def load_model(self):
        """
        Load the inpainting model into memory.
        """
        self.pipe = AutoPipelineForInpainting.from_pretrained(
            self.model_name, torch_dtype=self.torch_dtype, variant="fp16"
        ).to(self.device)
        self.pipe.enable_xformers_memory_efficient_attention()

    def inpaint(self, image: Image, mask_image: Image, prompt: str, negative_prompt: str = "", 
                guidance_scale: float = 11.5, steps: int = 20, strength: float = 1.0) -> Image:
        """
        Perform inpainting on the given image and mask with the provided prompt and settings.
        Arguments:
            - image (PIL.Image): The image to inpaint.
            - mask_image (PIL.Image): The mask specifying where to inpaint.
            - prompt (str): The prompt for inpainting.
            - negative_prompt (str): The negative prompt for inpainting (optional).
            - guidance_scale (float): The guidance scale for the inpainting (default: 7.5).
            - steps (int): Number of inference steps to perform (default: 20).
            - strength (float): Strength of inpainting (default: 1.0).
        Returns:
            - PIL.Image: The resulting inpainted image.
        """
        if self.pipe is None:
            self.load_model()

        # Resize images to the expected input size (1024x1024)
        image = image.convert("RGB").resize((1024, 1024))
        mask_image = mask_image.convert("RGB").resize((1024, 1024))

        # Perform inpainting (without scheduler)
        result = self.pipe(
            prompt=prompt,
            negative_prompt="No text, distorted anatomy, extra limbs, missing limbs, malformed hands, malformed feet, extra fingers, extra toes, blurry face, unrealistic facial features, unnatural expressions, out of proportion body, disjointed body parts, unnatural pose, unnatural lighting, incorrect skin tone, blurry eyes, cross-eyed, asymmetrical face, deformed head, disproportionate head, awkward posture, unrealistic clothing, wrong age, wrinkles, sagging skin, low-quality resolution, flat face, distorted mouth, messy hair, poorly drawn hair, extra body parts, awkward perspective, out of place shadows, oversaturated skin, unnatural shading, clothing artifacts, incorrect facial features ,extra objects, random artifacts, mismatched shapes, unnatural proportions, blurry details, irrelevant items, distorted anatomy, broken objects, inconsistent blending, clutter, noise, unrelated elements, incorrect lighting, incorrect shadows, out of place, incorrect style, wrong perspective, disjointed composition, unnatural texture, over-detailed, oversaturated colors, unnatural transitions",
            image=image,
            mask_image=mask_image,
            guidance_scale=guidance_scale,
            num_inference_steps=steps,
            strength=strength
        )

        # Return the inpainted image
        return result.images[0]

    def cleanup(self):
        """
        Clean up resources by deleting the model and clearing GPU memory.
        """
        if self.pipe is not None:
            del self.pipe
            torch.cuda.empty_cache()
            self.pipe = None

    def __del__(self):
        """
        Ensure that the model is cleaned up when the object is deleted.
        """
        self.cleanup()