import requests
import json
import base64
from io import BytesIO
from PIL import Image
import random

def show_image(base64_string):
    image_data = base64.b64decode(base64_string)
    image = Image.open(BytesIO(image_data))
    image.show()

# Define the API endpoint for campaign creation
url = 'https://9c3d-13-127-18-35.ngrok-free.app/create/'  # Update with your actual endpoint

# Define multiple campaign variations
campaign_variations = [
    {
        "keywords": ["sale", "discount", "limited time"],
        "cta_text": "Shop Now!",
        "current_campaign": "Spring Sale 2024",
        "season": "Spring",
        "mood": "Cheerful",
        "visual_elements": {"colors": ["red", "blue"]},
        "content_type": "Promotional",
        "advanced_data": {
            "style": "Modern",
            "colorScheme": "Bright",
            "lighting": "Natural",
            "composition": "Balanced",
            "perspective": "Wide Angle",
            "mood": "Joyful",
            "background": "Floral",
            "timePeriod": "2024"
        }
    },
    {
        "keywords": ["sale", "discount", "limited time"],
        "cta_text": "Shop Now!",
        "current_campaign": "Spring Sale 2024",
        "season": "Spring",
        "mood": "Cheerful",
        "visual_elements": {"colors": ["red", "blue"]},
        "content_type": "Promotional",
        "advanced_data": {
            "style": "Modern",
            "colorScheme": "Bright",
            "lighting": "Natural",
            "composition": "Balanced",
            "perspective": "Wide Angle",
            "mood": "Joyful",
            "background": "Floral",
            "timePeriod": "2024"
        }
    }
   
]

# Randomly select one campaign variation
selected_campaign = random.choice(campaign_variations)

# Send a POST request to create the campaign
response = requests.post(url, json=selected_campaign)

# Check the response status
if response.status_code == 201:
    show_image(response.json().get("generated_images")[0])
    print("Campaign created successfully!")
    print("Selected Campaign Data:", json.dumps(selected_campaign, indent=2))
    print("Response Data:", response.json())
else:
    print("Failed to create campaign.")
    print("Status Code:", response.status_code)
    print("Response Data:", response.text)
