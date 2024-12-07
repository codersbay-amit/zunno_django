import imgkit
from PIL import Image,ImageDraw
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from PIL import Image, ImageDraw

from PIL import Image
import requests
from io import BytesIO
import time


def create_class_mask(pil_image,target_class_name, conf_threshold=0.3):
    """
    Generates a mask image for a specified class using a YOLO segmentation model.
    Parameters:
    - pil_image (PIL.Image): PIL Image object of the input image.
    - model_path (str): Path to the YOLO model file.
    - target_class_name (str): Name of the class to create a mask for.
    - conf_threshold (float): Confidence threshold for model predictions (default: 0.3).
    Returns:
    - PIL.Image: Mask image with white (255) areas for the specified class and black (0) elsewhere.
    """
    # Load the YOLO segmentation model
    model = YOLO("best.pt")
    # Convert the PIL image to a format compatible with the model if needed (e.g., numpy array)
    image_array = np.array(pil_image)
    # Make predictions on the given image with segmentation enabled
    results = model.predict(image_array, conf=conf_threshold, task="segment")
    detection = results[0]
    # Create a blank mask image with the same dimensions as the original image
    mask_image = Image.new("L", pil_image.size, 0)  # "L" mode for grayscale mask
    draw = ImageDraw.Draw(mask_image)
    # Iterate through each detected mask and draw polygons for the target class
    if detection.masks is not None:
        for i, mask in enumerate(detection.masks.xy):
            class_id = int(detection.boxes.cls[i].item())
            class_name = detection.names[class_id]
            # Check if the class name matches the target class
            if class_name == target_class_name:
                # Flatten polygon coordinates for PIL drawing
                flattened_polygon = [(int(x), int(y)) for x, y in mask.tolist()]
                # Draw the polygon on the mask (255 for white mask area)
                draw.polygon(flattened_polygon, outline=255, fill=255)
    return mask_image
def crop_black_surrounding(pil_image):
    # Convert PIL Image to NumPy array
    image_array = np.array(pil_image)

    # Create a mask for non-white pixels
    threshold = 240
    mask = np.all(image_array > threshold, axis=-1)
    non_white_mask = ~mask

    # Find the bounding box of the non-white regions
    coords = np.argwhere(non_white_mask)
    if coords.size == 0:
        return pil_image  # Return the original image if no non-white areas found

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    # Crop the image using the bounding box
    cropped_image = pil_image.crop((x_min, y_min, x_max + 1, y_max + 1))

    # Create a mask for rounded corners
    rounded_mask = Image.new('L', cropped_image.size, 0)  # 'L' mode for a grayscale mask
    draw = ImageDraw.Draw(rounded_mask)
    draw.rounded_rectangle((3, 3, cropped_image.width-3, cropped_image.height-3), radius=20, fill=255)

    # Apply the mask to the cropped image
    rounded_cropped_image = Image.new('RGBA', cropped_image.size)
    rounded_cropped_image.paste(cropped_image, (0, 0), rounded_mask)
    rounded_cropped_image.show()
    return rounded_cropped_image



def create_button_html(
    text="Click Me!",  # Default button text
    text_color=(2, 0, 255),  # Default text color (white)
    background_color=(255, 255, 255, 0.1),  # Default background color (white with 10% opacity)
    border_color="black",  # Default border color (white)
    padding="15px 32px", 
    font_size="32px", 
    margin="4px 2px", 
    border_radius="4px", 
    on_click="alert('Button clicked!')"
):
    # Convert RGB tuples to CSS rgb format
    background_color_css =f"rgb({background_color[0]}, {background_color[1]}, {background_color[2]})"
    border_color_css = 'black' #background_color #f"rgb({border_color[0]}, {border_color[1]}, {border_color[2]})"
    text_color_css = f"rgb({text_color[0]}, {text_color[1]}, {text_color[2]})"    
    # Define the button HTML
    button_html = f'''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Button Example</title>
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">
        <style>
            .my-button {{
                background-color: {background_color_css}; /* Background color */
                border: 0.5px solid {border_color_css}; /* Border color */
                color: {text_color_css}; /* Text color */
                padding: {padding}; /* Padding */
                text-align: center; /* Centered text */
                font-weight: bolder; /* Font weight */
                font-size: {font_size}; /* Font size */
                margin: {margin}; /* Margin */
                cursor: pointer; /* Pointer cursor */
                border-radius: {border_radius}; /* Rounded corners */
               
            }}
        </style>
    </head>
    <body style='background-color:transparent'>
        <button class="my-button" 
                id="myButton" 
                name="buttonName" 
                type="button" 
                onclick="{on_click}" 
                aria-label="Example button">
            {text} <i class="fas fa-arrow-right"></i> <!-- Icon -->
        </button>
    </body>
    </html>
    '''
    
    return button_html

# Create the HTML content using the function
def create_button1(
    text="Click Me!",  # Default button text
    text_color=(2, 0, 255),  # Default text color (white)
    background_color=(255, 255, 255, 0.1),  # Default background color (white with 10% opacity)
    border_color="black",  # Default border color (white)
    padding="15px 32px", 
    font_size="32px", 
    margin="4px 2px", 
    border_radius="4px", 
    on_click="alert('Button clicked!')"
):
    print(text_color,background_color)
    html_content = create_button_html(
    text=text,  # Default button text
    text_color=text_color,  # Default text color (white)
    background_color=background_color,  # Default background color (white with 10% opacity)
    border_color="black",  # Default border color (white)
    padding="4px", 
    font_size="15px", 
    margin="4px 2px", 
    border_radius="8px", 
    on_click="alert('Button clicked!')"
    )
    # Define your wkhtmltoimage configuration
    config = imgkit.config(wkhtmltoimage='/usr/bin/wkhtmltoimage')
    # Convert the HTML string to an image
    imgkit.from_string(html_content, 'captured_image.png', config=config)
    return crop_black_surrounding(Image.open('captured_image.png'))

def create_button(image:Image, text, bbox, radius=15, icon_path='arrow-right-double-line.png', font_color='black', fill_color='white'):
    # Create a new image with the specified background color
    print(font_color,fill_color)
    button = create_button1(text=text, text_color=font_color, background_color=fill_color)
    
    # Ensure both images are in RGBA mode
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    if button.mode != 'RGBA':
        button = button.convert('RGBA')

    # Get the position from bbox and convert to integers
    top_left = (int(bbox[0]), int(bbox[1]))
    
    # Use the size of the button to create a box
    button_size = button.size  # This is a tuple (width, height)
    
    # Create a box using only the top-left point and the button size, converting to int
    box = (top_left[0], top_left[1], top_left[0] + int(button_size[0]), top_left[1] + int(button_size[1]))

    # Paste the button onto the image
    image.paste(button, box, button)  # Using button as mask for transparency
    
    return image

from PIL import Image
def paste_image_on_background(base_image:Image, paste_image:Image, bbox)->Image:
    """
    Pastes one PIL image onto another using the provided bounding box.
    Preserves the aspect ratio of the pasted image.

    Args:
        base_image (Image): The base image as a PIL Image object.
        paste_image (Image): The image to paste as a PIL Image object with transparency.
        bbox (list): Bounding box [x1, y1, x2, y2] where:
            - (x1, y1) is the top-left corner.
            - (x2, y2) is the bottom-right corner.

    Returns:
        Image: The combined image with the pasted image.
    """
    # Validate the bounding box
    if len(bbox) != 4 or any(not isinstance(i, (int, float)) for i in bbox):
        raise ValueError("Bounding box must be a list of four numeric values [x1, y1, x2, y2].")

    # Calculate the position to paste the image (top-left corner of the bounding box)
    x1, y1, x2, y2 = map(int, bbox)  # Convert to integers
    paste_position = (x1, y1)

    # Get the size of the bounding box
    bbox_width = x2 - x1
    bbox_height = y2 - y1

    # Get the original size of the paste image
    paste_width, paste_height = paste_image.size

    # Calculate the aspect ratios
    aspect_ratio = paste_width / paste_height
    bbox_aspect_ratio = bbox_width / bbox_height

    # Determine new size while maintaining aspect ratio
    if bbox_aspect_ratio > aspect_ratio:
        # Bounding box is wider than the pasted image
        new_height = bbox_height
        new_width = int(new_height * aspect_ratio)
    else:
        # Bounding box is taller than the pasted image
        new_width = bbox_width
        new_height = int(new_width / aspect_ratio)

    # Resize the pasted image to fit the bounding box while maintaining aspect ratio
    paste_image = paste_image.resize((new_width, new_height), Image.LANCZOS)

    # Ensure the pasted image is in 'RGBA' mode to preserve transparency
    if paste_image.mode != 'RGBA':
        paste_image = paste_image.convert('RGBA')

    # Check if the pasted image fits within the base image dimensions
    if (x1 < 0 or y1 < 0 or x2 > base_image.width or y2 > base_image.height):
        raise ValueError("Bounding box is out of base image bounds.")

    # Paste the image onto the base image using the alpha channel as a mask
    base_image.paste(paste_image, paste_position, mask=paste_image)

    return base_image




from PIL import Image, ImageDraw, ImageFont

def draw_multiline_text_in_bbox(image: Image.Image, text: str, bbox: tuple, 
                                  font_path: str = "arial.ttf", 
                                  gradient_start: tuple = (100, 0, 0), 
                                  gradient_end: tuple = (0, 0, 100)) -> Image.Image:
    """
    Draws multiline text inside a given bounding box on the provided image with a 3D effect and gradient color.
    Args:
        image (Image.Image): The image to draw on.
        text (str): The text to be drawn, which can contain line breaks.
        bbox (tuple): The bounding box as (left, upper, right, lower).
        font_path (str): The path to the TTF font file to be used.
        gradient_start (tuple): The RGB color to start the gradient (default red).
        gradient_end (tuple): The RGB color to end the gradient (default blue).
    Returns:
        Image.Image: The image with the text drawn inside the bounding box.
    """
    # Create a drawing context
    text=text.replace('"','')
    print(gradient_start,gradient_end)
    draw = ImageDraw.Draw(image)
    # Extract bounding box coordinates
    left, upper, right, lower = bbox
    bbox_width = int((right - left) * 0.9)
    bbox_height = int((lower - upper) * 0.9)
    
    # Set initial font size
    font_size = bbox_height // 2  # Start with a reasonable size
    gradient_start=(gradient_start[0]+50%255,gradient_start[1]+50%255,gradient_start[2]+50%255)
    gradient_end=(gradient_start[0]+80%255,gradient_start[1]+80%255,gradient_start[2]+80%255)
    # Load font
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        font = ImageFont.load_default(font_size)

    # Reduce font size until the text fits in the bounding box
    while True:
        # Split the text into lines that fit within the bounding box width
        lines = []
        words = text.split()
        current_line = ""
        for word in words:
            test_line = f"{current_line} {word}".strip()
            text_bbox = draw.textbbox((left, upper), test_line, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            if text_width <= bbox_width:
                current_line = test_line
            else:
                lines.append(current_line)
                current_line = word
        lines.append(current_line)  # Add the last line

        # Calculate total text height
        total_text_height = sum(draw.textbbox((0, 0), line, font=font)[3] - draw.textbbox((0, 0), line, font=font)[1] for line in lines)
        if total_text_height <= bbox_height:
            break
        font_size -= 1
        try:
            font = ImageFont.truetype(font_path, font_size)
        except IOError:
            font = ImageFont.load_default(font_size)

    # Calculate starting y position to center the text vertically in the bbox
    y = upper + (bbox_height - total_text_height) / 2

    # Draw each line of text with 3D effect and gradient color
    for i, line in enumerate(lines):
        # 3D effect: draw shadow first
        shadow_offset = 0  # Change this for more or less shadow
        shadow_color = (50, 50, 50)  # Dark gray shadow color
        shadow_text_bbox = draw.textbbox((left + shadow_offset, y + shadow_offset), line, font=font)
        shadow_width = shadow_text_bbox[2] - shadow_text_bbox[0]
        # while(shadow_width<bbox_width):
        #     font_size+=1
        #     font = ImageFont.truetype(font_path, font_size)
        #     shadow_text_bbox = draw.textbbox((left + shadow_offset, y + shadow_offset), line, font=font)
        #     shadow_width = shadow_text_bbox[2] - shadow_text_bbox[0]
        shadow_x = left + shadow_offset  # Left align the shadow
        draw.text((shadow_x, y + shadow_offset), line, font=font, fill=shadow_color)

        # Calculate gradient color for the line
        r = int(gradient_start[0] + (gradient_end[0] - gradient_start[0]) * (i / len(lines)))
        g = int(gradient_start[1] + (gradient_end[1] - gradient_start[1]) * (i / len(lines)))
        b = int(gradient_start[2] + (gradient_end[2] - gradient_start[2]) * (i / len(lines)))
        gradient_color = (r, g, b)

        # Draw the main text with gradient color
        text_bbox = draw.textbbox((left, y), line, font=font)
        x = left  # Left align the text
        draw.text((x, y), line, font=font, fill=gradient_color)  # Use gradient color

        y += text_bbox[3] - text_bbox[1] + 5  # Move y position down for the next line

    return image
from PIL import Image, ImageDraw, ImageFont

def draw_multiline_text_in_bbox_right(image: Image.Image, text: str, bbox: tuple, 
                                  font_path: str = "arial.ttf", 
                                  gradient_start: tuple = (100, 0, 0), 
                                  gradient_end: tuple = (0, 0, 100)) -> Image.Image:
    """
    Draws multiline text inside a given bounding box on the provided image with a 3D effect and gradient color.
    Args:
        image (Image.Image): The image to draw on.
        text (str): The text to be drawn, which can contain line breaks.
        bbox (tuple): The bounding box as (left, upper, right, lower).
        font_path (str): The path to the TTF font file to be used.
        gradient_start (tuple): The RGB color to start the gradient (default red).
        gradient_end (tuple): The RGB color to end the gradient (default blue).
    Returns:
        Image.Image: The image with the text drawn inside the bounding box.
    """
    # Create a drawing context
    text = text.replace('"', '')
    draw = ImageDraw.Draw(image)
    
    # Extract bounding box coordinates
    left, upper, right, lower = bbox
    bbox_width = int((right - left) * 0.9)
    bbox_height = int((lower - upper) * 0.9)
    
    # Set initial font size
    font_size = bbox_height // 2  # Start with a reasonable size

    # Load font
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        font = ImageFont.load_default()

    # Reduce font size until the text fits in the bounding box
    while True:
        # Split the text into lines that fit within the bounding box width
        lines = []
        words = text.split()
        current_line = ""
        
        for word in words:
            test_line = f"{current_line} {word}".strip()
            text_bbox = draw.textbbox((left, upper), test_line, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            if text_width <= bbox_width:
                current_line = test_line
            else:
                lines.append(current_line)
                current_line = word
        lines.append(current_line)  # Add the last line

        # Calculate total text height
        total_text_height = sum(draw.textbbox((0, 0), line, font=font)[3] - draw.textbbox((0, 0), line, font=font)[1] for line in lines)
        if total_text_height <= bbox_height:
            break
        font_size -= 1
        try:
            font = ImageFont.truetype(font_path, font_size)
        except IOError:
            font = ImageFont.load_default()

    # Calculate starting y position to center the text vertically in the bbox
    y = upper + (bbox_height - total_text_height) / 2

    # Draw each line of text with 3D effect and gradient color
    for i, line in enumerate(lines):
        # 3D effect: draw shadow first
        shadow_offset = 0  # Change this for more or less shadow
        shadow_color = (50, 50, 50)  # Dark gray shadow color
        shadow_text_bbox = draw.textbbox((left + shadow_offset, y + shadow_offset), line, font=font)
        shadow_x = right - shadow_text_bbox[2] + shadow_offset  # Right align the shadow
        draw.text((shadow_x, y + shadow_offset), line, font=font, fill=shadow_color)

        # Calculate gradient color for the line
        r = int(gradient_start[0] + (gradient_end[0] - gradient_start[0]) * (i / len(lines)))
        g = int(gradient_start[1] + (gradient_end[1] - gradient_start[1]) * (i / len(lines)))
        b = int(gradient_start[2] + (gradient_end[2] - gradient_start[2]) * (i / len(lines)))
        gradient_color = (r, g, b)

        # Draw the main text with gradient color
        text_bbox = draw.textbbox((left, y), line, font=font)
        x = right - text_bbox[2]  # Right align the text
        draw.text((x, y), line, font=font, fill=gradient_color)

        y += text_bbox[3] - text_bbox[1] + 5  # Move y position down for the next line

    return image


def draw_multiline_text_in_bbox_center(image: Image.Image, text: str, bbox: tuple, 
                                  font_path: str = "arial.ttf", 
                                  gradient_start: tuple = (100, 0, 0), 
                                  gradient_end: tuple = (0, 0, 100)) -> Image.Image:
    print(gradient_start,gradient_end)
    """
    Draws multiline text inside a given bounding box on the provided image with a 3D effect and gradient color.
    Args:
        image (Image.Image): The image to draw on.
        text (str): The text to be drawn, which can contain line breaks.
        bbox (tuple): The bounding box as (left, upper, right, lower).
        font_path (str): The path to the TTF font file to be used.
        gradient_start (tuple): The RGB color to start the gradient (default red).
        gradient_end (tuple): The RGB color to end the gradient (default blue).
    Returns:
        Image.Image: The image with the text drawn inside the bounding box.
    """
    gradient_start=(gradient_start[0]+50%255,gradient_start[1]+50%255,gradient_start[2]+50%255)
    gradient_end=(gradient_start[0]+80%255,gradient_start[1]+80%255,gradient_start[2]+80%255)
    draw = ImageDraw.Draw(image)
    left, upper, right, lower = bbox
    bbox_width = int((right - left) * 0.9)
    bbox_height = int((lower - upper) * 0.9)
    
    font_size = bbox_height // 2

    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        font = ImageFont.load_default(font_size)

    while True:
        lines = []
        words = text.split()
        current_line = ""
        for word in words:
            test_line = f"{current_line} {word}".strip()
            text_bbox = draw.textbbox((left, upper), test_line, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            if text_width <= bbox_width:
                current_line = test_line
            else:
                lines.append(current_line)
                current_line = word
        lines.append(current_line)

        total_text_height = sum(draw.textbbox((0, 0), line, font=font)[3] - draw.textbbox((0, 0), line, font=font)[1] for line in lines)
        if total_text_height <= bbox_height:
            break
        font_size -= 1
        try:
            font = ImageFont.truetype(font_path, font_size)
        except IOError:
            font = ImageFont.load_default(font_size)

    y = upper + (bbox_height - total_text_height) / 2

    for i, line in enumerate(lines):
        shadow_offset = 1  # Change this for more or less shadow
        shadow_color = (50, 50, 50)
        shadow_text_bbox = draw.textbbox((left + shadow_offset, y + shadow_offset), line, font=font)
        
        shadow_x = left + shadow_offset
#        draw.text((shadow_x, y + shadow_offset), line, font=font, fill=shadow_color)

        r = int(gradient_start[0] + (gradient_end[0] - gradient_start[0]) * (i / len(lines)))
        g = int(gradient_start[1] + (gradient_end[1] - gradient_start[1]) * (i / len(lines)))
        b = int(gradient_start[2] + (gradient_end[2] - gradient_start[2]) * (i / len(lines)))
        gradient_color = (r, g, b)

        text_bbox = draw.textbbox((left, y), line, font=font)
        line_width = text_bbox[2] - text_bbox[0]
        x = left + (bbox_width - line_width) / 2  # Center the line
        draw.text((x, y), line, font=font, fill=gradient_color)

        y += text_bbox[3] - text_bbox[1] + 5  # Move y position down for the next line

    return image
import cv2
import numpy as np
import easyocr
from PIL import Image

def remove_text_with_easyocr(pil_image):
    """
    Detects and removes text from a PIL image using EasyOCR and OpenCV.
    :param pil_image: Input PIL image.
    :return: Processed PIL image with text removed.
    """
    # Initialize EasyOCR Reader
    reader = easyocr.Reader(['en'])

    # Convert PIL image to NumPy array and then to BGR format
    img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    # Perform OCR to detect text
    results = reader.readtext(img)

    # Create a mask for the detected text
    mask = np.zeros(img.shape[:2], dtype=np.uint8)

    for (bbox, text, prob) in results:
        # Get the bounding box coordinates
        pts = np.array(bbox, dtype=np.int32)
        cv2.fillConvexPoly(mask, pts, 255)  # Fill the detected text area

    # Inpaint the image using the mask
    result = cv2.inpaint(img, mask, inpaintRadius=1, flags=cv2.INPAINT_NS)

    # Convert BGR to RGB for PIL
    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

    # Convert NumPy array back to PIL Image
    result_pil = Image.fromarray(result_rgb)

    return result_pil

# Example usage
# processed_image = remove_text_with_easyocr(pil_image)

from PIL import Image, ImageDraw, ImageFont

def create_rounded_rectangle(draw, bbox, radius, outline=None, fill=None):
    print(fill)
    # Create a mask for the rounded rectangle
    mask = Image.new('L', (int(bbox[2]) - int(bbox[0]), int(bbox[3] - bbox[1])), 0)
    draw_mask = ImageDraw.Draw(mask)

    # Draw the rounded rectangle on the mask
    draw_mask.rounded_rectangle([0, 0,int( bbox[2] - bbox[0]), int(bbox[3] - bbox[1])], radius, fill=fill)

    # Draw the filled rectangle
    if fill:
        draw.rectangle(bbox, fill=fill)

    # Draw the outline
    if outline:
        draw.rectangle(bbox, outline=outline)

    # Apply the mask
    draw.bitmap((bbox[0], bbox[1]), mask, fill=fill)

# Example usage
from PIL import Image, ImageDraw

def paste_image_in_bbox(base_image, image_to_paste, bbox,is_white=False):
    # Ensure both images are in RGBA mode
    base_image = base_image.convert("RGBA")
    image_to_paste = image_to_paste.convert("RGBA")

    # Unpack the bounding box (x, y, width, height)
    x, y, bbox_width, bbox_height = bbox
    x, y, bbox_width, bbox_height = int(x), int(y), int(bbox_width), int(bbox_height)

    # Create a draw object to fill the bounding box with white
    if is_white:
        draw = ImageDraw.Draw(base_image)
        draw.rectangle([x, y, x + bbox_width, y + bbox_height], fill=(255, 255, 255, 255))

    # Calculate the aspect ratio of the image to paste
    aspect_ratio = image_to_paste.width / image_to_paste.height

    # Determine new size while maintaining aspect ratio
    if bbox_width / bbox_height > aspect_ratio:
        new_height = bbox_height
        new_width = int(bbox_height * aspect_ratio)
    else:
        new_width = bbox_width
        new_height = int(bbox_width / aspect_ratio)

    # Resize the image to paste
    image_to_paste_resized = image_to_paste.resize((new_width, new_height), Image.LANCZOS)

    # Calculate position to center the resized image within the bounding box
    x_offset = x + (bbox_width - new_width) // 2
    y_offset = y + (bbox_height - new_height) // 2

    # Paste the resized image onto the base image at the calculated position
    base_image.paste(image_to_paste_resized, (x_offset, y_offset), image_to_paste_resized)

    return base_image 
