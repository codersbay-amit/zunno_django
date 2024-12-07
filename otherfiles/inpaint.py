import cv2
import cv2

def resize_image_aspect_ratio(image, max_size=640):
    """
    Resize an image to have its larger side equal to max_size while maintaining aspect ratio.

    Parameters:
    - image: numpy array, the input image.
    - max_size: int, the size of the larger side after resizing.

    Returns:
    - numpy array: the resized image.
    """
    # Get original dimensions
    h, w = image.shape[:2]

    # Calculate the scaling factor
    if h > w:
        scale = max_size / h
    else:
        scale = max_size / w

    # Calculate new dimensions
    new_w = int(w * scale)
    new_h = int(h * scale)

    # Resize the image
    resized_image = cv2.resize(image, (new_w, new_h))

    return resized_image

def inpaint_image(original_image, mask_image, inpaint_radius=20, method=cv2.INPAINT_NS):
    if original_image is None or mask_image is None:
        raise ValueError("Both original_image and mask_image must be provided.")
    
    # Inpaint the image using the mask

    inpainted_image = cv2.inpaint(original_image, mask_image, inpaintRadius=inpaint_radius, flags=method)
    
    return inpainted_image
