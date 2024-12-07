import os
import numpy as np
import cv2
from rembg import remove
import base64
import gc
import random
import subprocess
import torch
import requests
from io import BytesIO
from PIL import Image
from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from otherfiles.colors import nearest_color
from otherfiles.modelXL import generate
from otherfiles.modelXL_Depth import generate_image
from otherfiles.modelXL_Canny import generate_image_canny
from otherfiles.yolo_prediction import getBoxes
from otherfiles.rem import remove_png_files
from otherfiles.image_utils import paste_image_in_bbox,draw_multiline_text_in_bbox, remove_text_with_easyocr,create_button,draw_multiline_text_in_bbox_right,draw_multiline_text_in_bbox_center,create_class_mask, Mid_GEN
from otherfiles.ollamma import ollama_generate
from enhancer.services import enhance
from .models import BrandCreation,get_string
from parameters.serializers import BrandCreationSerializer
from otherfiles.flux import FluxImageGenerator,StableDiffusionImageGenerator
from otherfiles.model_with_inpaint import InpaintingModel

# Load template matcher
from otherfiles.clip import ImageTextMatcher
template_matcher = ImageTextMatcher()
from PIL import Image, ImageDraw
import numpy as np

def create_bbox_mask(image: Image, bbox):
    """
    Create a mask based on the bounding box area, ensuring bbox coordinates are integers.
    
    Args:
        image: PIL Image, the original image.
        bbox: tuple, the bounding box coordinates (x_min, y_min, x_max, y_max), can be floats.
        
    Returns:
        mask: PIL Image, the generated mask where the bbox area is white and the rest is black.
    """
    # Convert bbox coordinates to integers (round the float values)
    bbox = tuple(map(lambda x: int(round(x)), bbox))
    
    # Convert the image to grayscale for the mask (using a black canvas)
    width, height = image.size
    mask = Image.new("L", (width, height), 0)  # 0 means black

    # Draw a white rectangle in the bbox area
    draw = ImageDraw.Draw(mask)
    draw.rectangle(bbox, fill=255)  # 255 means white

    return mask
def get_bounding_box_position(image_size, bounding_box):
    """
    Estimates the position of a bounding box relative to the image size.

    :param image_size: Tuple of (width, height) of the image
    :param bounding_box: Tuple of (x, y, width, height) of the bounding box
    :return: String indicating the position of the bounding box
    """
    image_width, image_height = image_size
    box_x, box_y, box_width, box_height = bounding_box
    
    # Calculate the center of the bounding box
    box_center_x = box_x + box_width / 2

    # Determine the position
    if box_center_x < image_width / 3:
        return "left"
    elif box_center_x < 2 * image_width / 3:
        return "center"
    else:
        return "right"

def clear_cuda_cache():
    """Clear CUDA cache to free up memory."""
    gc.collect()
    torch.cuda.empty_cache()

def find_and_kill_process_by_name(process_name):
    """Kill all processes matching the given process name."""
    try:
        result = subprocess.run(['pgrep', process_name], stdout=subprocess.PIPE, text=True)
        process_ids = result.stdout.strip().splitlines()

        if not process_ids:
            return

        for pid in process_ids:
            subprocess.run(['sudo', 'kill', '-9', pid])
            
    except Exception as e:
        pass

def resize_image(image, max_size_kb=300):
    """Resize image to ensure it is under the specified size."""
    img_bytes = BytesIO()
    image.save(img_bytes, format='PNG')
    while img_bytes.tell() / 1024 > max_size_kb:
        new_size = (int(image.width * 0.9), int(image.height * 0.9))
        image = image.resize(new_size, Image.LANCZOS)
        img_bytes = BytesIO()
        image.save(img_bytes, format='PNG')
    return image

def get_boxes(template):
    """Get bounding boxes for the given template."""
    name = f'untitled{random.choice(range(0, 10000))}'
    template.save(f'{name}.png')
    boxes, empty_template_image = getBoxes(f'{name}.png')
    return boxes, empty_template_image, f'{name}.png'

class BrandCreationAPIView(APIView):
    def get(self, request):
        """Retrieve all brand creations."""
        clear_cuda_cache()
        brand_creations = BrandCreation.objects.all()
        serializer = BrandCreationSerializer(brand_creations, many=True)
        return Response({})

    def post(self, request):
        """Create a new brand."""
        clear_cuda_cache()
        serializer = BrandCreationSerializer(data=request.data)
        if serializer.is_valid():
            instance = serializer.save()
            logo_url = instance.logo
            mask=None
            # Generate initial template and remove text
            empty_template = Image.open(template_matcher.fetch_images_based_on_text(get_string(instance=instance)))
            if 'productPrompt' in request.data.keys():
                prompt=request.data['productPrompt']
                mask=create_class_mask(empty_template,'product')
                model=InpaintingModel()
                empty_template=model.inpaint(empty_template,mask,prompt)
                del model
                clear_cuda_cache()
            if 'productImage' in request.data.keys():
                product_url=request.data['productImage']
                product_image = Image.open(BytesIO(requests.get(product_url).content))
            elif 'productPrompt' in request.data.keys():
                pass
            else:
                 return Response({'message':'productPrompt or productImage field required'}, status=status.HTTP_400_BAD_REQUEST)
            boxes, empty, image_name = get_boxes(empty_template)
            empty_template = remove_text_with_easyocr(empty_template)
            if 'productImage' in request.data.keys():
                product_image=remove(product_image)
            if 'product' in boxes.keys() and 'productImage' in request.data.keys():
            	empty_template=paste_image_in_bbox(empty_template,product_image,boxes['product'][0],is_white=True)

            # Generate prompt and data
            prompt = self.create_prompt(instance)
            data = ollama_generate(instance=instance)

            # Kill previous process if necessary
            find_and_kill_process_by_name('ollama_llama_se')
            clear_cuda_cache()

            # Draw title and logo
           # final_template = self.draw_title(empty_template, data, boxes, instance.colors)
            final_template = self.add_logo(empty_template, logo_url, boxes)
             # Generate final images
            if 'title' in boxes.keys():
                final_template = self.draw_title(final_template, data, boxes, instance.colors)
            
                
            generated_images = self.generate_images(final_template, prompt)
            clear_cuda_cache()
            generated_images_base64 = self.convert_images_to_base64(generated_images, final_template, instance, data, boxes)

            # Prepare response data
            response_data = {
                'generated_images': generated_images_base64
            }

            # Cleanup
            remove_png_files()
            return Response(response_data, status=status.HTTP_201_CREATED)
        print(serializer.errors)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def create_prompt(self, instance):
        advanced_data_entry = instance.advancedData.first()
        """Create a prompt for the image generation."""
        return (
           f"""prompt = f"Create an image that captures a {instance.visualStyle} brand.
             Use colors: {', '.join([nearest_color(color.colorCode)[0] for color in instance.colors.all()])}.
               Set a {advanced_data_entry.mood} atmosphere with a 
               {advanced_data_entry.background}. 
               The lighting should be {advanced_data_entry.lighting},
                 and the composition should be {advanced_data_entry.composition} 
                 from a {advanced_data_entry.perspective} viewpoint." and analys the structure if you found anything then color accordingly eg tree fruits  
           """
        )
    def draw_subtitle(self, image, data, boxes, colors):
        """Draw the title on the image."""
        color_arr=colors.all()
        print(boxes.keys())
        primary_color=[color.colorCode for color in color_arr if color.type=='primary'][0] 
        secondary_color = [color.colorCode for color in color_arr if color.type!='primary'][0]
        print('subtitle',primary_color,secondary_color)
        _,gradient_start=nearest_color(primary_color)
        _,gradient_end=nearest_color(secondary_color)
        if 'Subheading' in boxes.keys():
            box_position=get_bounding_box_position(image.size,boxes['Subheading'][0]) 
            if box_position=='left':
                return draw_multiline_text_in_bbox(
                image=image, 
                text=data['description'], 
                bbox=boxes['Subheading'][0],
                gradient_start=gradient_start,
                gradient_end=gradient_end
                )
            if box_position=='right':
                return draw_multiline_text_in_bbox_center(
                image=image, 
                text=data['description'], 
                bbox=boxes['Subheading'][0],
                gradient_start=gradient_start,
                gradient_end=gradient_end
                )
            if box_position=='center':
                return draw_multiline_text_in_bbox_center(
                image=image, 
                text=data['description'], 
                bbox=boxes['Subheading'][0],
                gradient_start=gradient_start,
                gradient_end=gradient_end
                )
        else:
            return image
        if 'Subheading' in boxes.keys():
            return draw_multiline_text_in_bbox(
            image=image, 
            text=data['description'], 
            bbox=boxes['Subheading'][0],
            gradient_start=gradient_start,
            gradient_end=gradient_end
        )
        else:
            return image
    def draw_title(self, image, data, boxes, colors):
        """Draw the title on the image."""
        color_arr=colors.all()
        print(boxes.keys())
        primary_color=[color.colorCode for color in color_arr if color.type=='primary'][0]
        secondary_color = [color.colorCode for color in color_arr if color.type!='primary'][0]
        _,gradient_start=nearest_color(primary_color)
        _,gradient_end=nearest_color(secondary_color)
        if 'title' in boxes.keys():
            box_position=get_bounding_box_position(image.size,boxes['title'][0]) 
            if box_position=='left':
                return draw_multiline_text_in_bbox(
                image=image, 
                text=data['title'], 
                bbox=boxes['title'][0],
                gradient_start=gradient_start,
                gradient_end=gradient_end
                )
            if box_position=='right':
                return draw_multiline_text_in_bbox_center(
                image=image, 
                text=data['title'], 
                bbox=boxes['title'][0],
                gradient_start=gradient_start,
                gradient_end=gradient_end
                )
            if box_position=='center':
                return draw_multiline_text_in_bbox_center(
                image=image, 
                text=data['title'], 
                bbox=boxes['title'][0],
                gradient_start=gradient_start,
                gradient_end=gradient_end
                )
        else:
            return image

    def add_logo(self,image, logo_url, boxes):
      """Inpaint the areas within the given boxes and then add the logo to the image."""
      if 'logo' in boxes.keys():
        # Convert PIL image to OpenCV format for inpainting
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Create a mask for inpainting using all title boxes
        mask = np.zeros(image_cv.shape[:2], dtype=np.uint8)

        # Iterate through all title boxes for inpainting
        for box in boxes['logo']:
            x, y, w, h = box
            mask[int(y):int(h), int(x):int(w)] = 255  # Fill the mask with 255 in the specified box area

        # Inpaint the image
        image_inpainted = cv2.inpaint(image_cv, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

        # Convert back to PIL image
        image = Image.fromarray(cv2.cvtColor(image_inpainted, cv2.COLOR_BGR2RGB))

        # Fetch and prepare the logo image
        logo_image = Image.open(BytesIO(requests.get(logo_url).content)).convert("RGBA")  # Use RGBA for transparency
        logo_image = remove(logo_image)  # Assuming 'remove' handles background removal properly
        
        # Resize logo
        logo_image = resize_image(logo_image, 100)

        # Get the coordinates of the first box for pasting
        first_box = boxes['logo'][0]
        x, y, w, h = first_box
        x, y, w, h=int(x), int(y), int(w), int(h)
        # Create a mask from the logo image to handle transparency
        logo_mask = logo_image.split()[3]  # Get the alpha channel as mask

        # Resize logo to fit the first box
        paste_size = (int(w - x)+20, int(h - y)+10)
        logo_image = logo_image.resize(paste_size, Image.BICUBIC)
        logo_mask = logo_mask.resize(paste_size, Image.BICUBIC)
        # Paste the logo with transparency
        image.paste(logo_image, (int(x), int(y)), mask=logo_mask)  # Use mask for transparency

      return image
    def generate_images(self, template, prompt):
        """Generate images using the provided template and prompt."""
        additional_prompt = "best quality, extremely detailed"
        negative_prompt = """bad colors, oversaturated skin, unnatural shading, clothing artifacts, incorrect facial features"""
        return generate_image_canny(
            prompt=prompt+additional_prompt, 
            negative_prompt=negative_prompt, 
            image=template,
            controlnet_conditioning_scale=0.8, 
            steps=30
        )

    def convert_images_to_base64(self, images, final_template, instance, data, boxes):
        """Convert generated images to Base64 format."""
        generated_images_base64 = []
        for img in images:
            img = img.resize(final_template.size)
            if 'action button' in boxes.keys() :
                color_arr=instance.colors.all()
                print(boxes.keys())
                primary_color=[color.colorCode for color in color_arr if color.type=='primary'][0]
                secondary_color = [color.colorCode for color in color_arr if color.type!='primary'][0]
                _,font_color= nearest_color(primary_color)
                _, fill_color= nearest_color(secondary_color)
                img=create_button(image=img, text=instance.ctaText, bbox=boxes['action button'][0], radius=15, font_color= font_color, fill_color= fill_color)
            if 'title' in boxes.keys():
            	img = self.draw_title(img, data, boxes, instance.colors)
            if 'logo' in boxes.keys():
                img=self.add_logo(img,instance.logo, boxes)
            if 'Subheading' in boxes.keys():
                img=self.draw_subtitle(img, data, boxes, instance.colors)
            img = enhance(img)

            buffered = BytesIO()
            img.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            generated_images_base64.append(img_base64)

        return generated_images_base64

    def convert_image_to_base64(self, img):
        """Convert a single image to Base64 format."""
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

