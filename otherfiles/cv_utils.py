import cv2
import numpy as np


def resize_image(input_image, resolution, interpolation=None):
    H, W, C = input_image.shape
    H = float(H)
    W = float(W)
    k = float(resolution) / max(H, W)
    H *= k
    W *= k
    H = int(np.round(H / 64.0)) * 64
    W = int(np.round(W / 64.0)) * 64
    if interpolation is None:
        interpolation = cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA
    img = cv2.resize(input_image, (W, H), interpolation=interpolation)
    return img

from PIL import Image, ImageDraw, ImageFont

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
    draw = ImageDraw.Draw(image)
    # Extract bounding box coordinates
    left, upper, right, lower = bbox
    bbox_width = right - left
    bbox_height = lower - upper
    
    # Draw the bounding box
    draw.rectangle(bbox, outline="blue", width=2)  # Draw the bounding box in blue

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
        font = ImageFont.truetype(font_path, font_size)

    # Calculate starting y position to center the text vertically in the bbox
    y = upper + (bbox_height - total_text_height) / 2

    # Draw each line of text with 3D effect and gradient color
    for i, line in enumerate(lines):
        # 3D effect: draw shadow first
        shadow_offset = 2  # Change this for more or less shadow
        shadow_color = (50, 50, 50)  # Dark gray shadow color
        draw.text((left + shadow_offset, y + shadow_offset), line, font=font, fill=shadow_color)

        # Calculate gradient color for the line
        r = int(gradient_start[0] + (gradient_end[0] - gradient_start[0]) * (i / len(lines)))
        g = int(gradient_start[1] + (gradient_end[1] - gradient_start[1]) * (i / len(lines)))
        b = int(gradient_start[2] + (gradient_end[2] - gradient_start[2]) * (i / len(lines)))
        gradient_color = (r, g, b)

        # Draw the main text with gradient color
        draw.text((left, y), line, font=font, fill=gradient_color)  # Left-aligned

        # Move y position down for the next line
        y += draw.textbbox((0, 0), line, font=font)[3] - draw.textbbox((0, 0), line, font=font)[1] + 5

    return image

