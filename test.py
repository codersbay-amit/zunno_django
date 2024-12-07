import requests
import json
from PIL import Image
from io import BytesIO
import base64
# Your API URL
url = "https://0346-65-0-176-205.ngrok-free.app/create_brand/"

data={
  "logo": "https://png.pngitem.com/pimgs/s/216-2162996_logotipo-unilever-hd-png-download.png",
  "name": "EcoGlow Cosmetics",
  "colors": [
    {
      "colorCode": "#0000FF",
      "type": "primary",
      "_id": "6712790c216842eba6ddf120"
    },
    {
      "colorCode": "#FFFFFF",
      "type": "secondary",
      "_id": "6712790c216842eba6ddf121"
    }
  ],
  "titleFont": "Arial Bold",
  "subTitleFont": "Arial",
  "bodyFont": "Verdana",
  "tagLine": "Beauty with a Conscience",
  "industry": "Cosmetics",
  "productPrompt":'beuty products',
  "demographic": "Eco-conscious beauty enthusiasts aged 18-40",
  "psychographic": "Individuals who seek cruelty-free, natural beauty products",
  "audienceInterest": "Skincare, makeup, eco-friendly beauty",
  "toneOfVoice": "Empowering",
  "messageStyle": "Informative",
  "visualStyle": "Chic",
  "competitorBrands": ["Plum Goodness", "Forest Essentials", "Kama Ayurveda"],
  "benchmarkBrands": ["Fenty Beauty", "Charlotte Tilbury"],
  "brandGuidelinedocs": [
    "https://example.com/cosmetics-guideline1.pdf",
    "https://example.com/cosmetics-guideline2.pdf"
  ],
  "copyrightText": "© 2024 EcoGlow Cosmetics. All rights reserved.",
  "createdBy": "6710fc3695ec5c530bc58842",
  "createdAt": "2024-10-18T15:04:44.646Z",
  "updatedAt": "2024-10-18T15:04:44.646Z",
  "__v": 0,
  "keywords": ["cosmetics", "natural beauty", "eco-friendly", "skincare"],
  "ctaText": "Discover Your Glow!",
  "currentCampaign": "Autumn Beauty Sale 2024",
  "visualElements": ["pink", "gold"],
  "contentType": "image",
  "advanced": {
    "style": "Modern",
    "colorScheme": "neutral",
    "lighting": "Soft",
    "composition": "Balanced",
    "perspective": "Close-Up",
    "mood": "Inviting",
    "background": "Floral",
    "timePeriod": "2024"
  }
}

response = requests.post(url, json=data)
# Check the response
if response.status_code == 201:
    print("Data posted successfully!")
    # Extract the image data from the response
    response_data = response.json()
    generated_images_base64 = response_data.get('generated_images')[0]
    # If the image data is in base64 format, decode it
    if generated_images_base64:
        # Decode the base64 image
        image_data = BytesIO(base64.b64decode(generated_images_base64))
        # Open the image
        image = Image.open(image_data)
        # Display the image
        image.show()
    else:
        print("No image data found in the response.")
else:
    print(f"Failed to post data: {response.status_code}, {response.text}")