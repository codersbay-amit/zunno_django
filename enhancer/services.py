
import os
from PIL import Image
from io import BytesIO
import uuid
import numpy as np
import base64
from enhancer.enhancer.enhancer import Enhancer


TEMP_PATH = 'temp'
ENHANCE_METHOD = os.getenv('METHOD')
BACKGROUND_ENHANCEMENT = os.getenv('BACKGROUND_ENHANCEMENT')
if ENHANCE_METHOD is None:
    ENHANCE_METHOD = 'gfpgan'

if BACKGROUND_ENHANCEMENT is None:
    BACKGROUND_ENHANCEMENT = True
else:
    BACKGROUND_ENHANCEMENT = True if BACKGROUND_ENHANCEMENT == 'True' else False

enhancer = Enhancer(background_enhancement=BACKGROUND_ENHANCEMENT, upscale=2)


def enhance(image:Image) -> Image:
   
    restored_image = enhancer.enhance(np.array(image))

    final_image = Image.fromarray(restored_image)
    return final_image
    # buffered = BytesIO()
    # final_image.save(buffered, format="JPEG")
    # encoded_img = base64.b64encode(buffered.getvalue())
    
    # return encoded_img
        