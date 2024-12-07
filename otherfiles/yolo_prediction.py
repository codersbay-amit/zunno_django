import numpy as np
import matplotlib.pyplot as plt
import cv2
from ultralytics import YOLO
from PIL import Image

from otherfiles.inpaint import resize_image_aspect_ratio,inpaint_image
from PIL import Image, ImageDraw
import cv2
import numpy as np

def draw_rectangle(bbox, image, color='blue'):
    """
    Draw a rectangle on an image.

    Parameters:
    - bbox: A tuple (x_min, y_min, width, height)
    - image_size: A tuple (width, height) for the image
    - color: Color of the rectangle
    """
    # Create a new image with white background
    img = image
    draw = ImageDraw.Draw(img)

    # Calculate the coordinates of the rectangle
    x_min, y_min, width, height = bbox
    x_max = x_min + width
    y_max = y_min + height

    # Draw the rectangle
    draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=3)

    # Show the image
    img.show()
def predict_and_display_masks(bounding_boxes,original_image):
    
    """
    Creates a masked image from the specified bounding boxes.

    Parameters:
    - image_path (str): The path to the input image.
    - bounding_boxes (list): A list of bounding boxes defined as tuples (x1, y1, x2, y2).

    Returns:
    - masked_image (numpy.ndarray): The masked image with bounding boxes white and the rest black.
    """
    # Load the image
    image = original_image

    # Create a mask initialized to black
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    # Draw white rectangles for each bounding box on the mask
    for key in bounding_boxes.keys():
        if key!='product':
            data=bounding_boxes[key]
            for d in data:
                (x1, y1, x2, y2)=d
                cv2.rectangle(mask, (int(x1), int(y1)), (int(x2), int(y2)), 255, thickness=cv2.FILLED)

    # Create the masked image
    

    return mask

# Example usage


    
    
# Load the model
def getBoxes(image_path):
    original_image=cv2.imread(image_path)
    model = YOLO("best.pt")
    results = model.predict(image_path, conf=0.2)
    detections = results[0]
    extracted_masks=results[0].masks.data
    bounding_boxes = {}
    boxes = detections.boxes
    class_ids = boxes.cls.tolist()  # Class IDs
    # Loop through boxes again to build the dictionary
    for i in range(len(boxes.xyxy)):
        class_id = int(class_ids[i])
        class_name = detections.names[class_id]  # Get class name from class ID
        # Store bounding box in the dictionary
        if class_name not in bounding_boxes:
            bounding_boxes[class_name] = []
  
        bounding_boxes[class_name].append((boxes.xyxy[i][0].item(), boxes.xyxy[i][1].item(), boxes.xyxy[i][2].item(), boxes.xyxy[i][3].item()))
    masked_image=predict_and_display_masks(bounding_boxes=bounding_boxes,original_image=original_image)
    Image.fromarray(masked_image).show()
    inpainted_image=inpaint_image(original_image=original_image,mask_image=masked_image)
    Image.fromarray(detections.plot()).show()
    return bounding_boxes,inpainted_image



