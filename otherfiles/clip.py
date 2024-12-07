import os
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import random
from otherfiles.yolo_prediction import getBoxes

class ImageTextMatcher:
    def __init__(self, model_name="openai/clip-vit-base-patch16", embedding_file='image_embeddings.pt'):
        # Load the CLIP model and processor
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.image_embeddings = None
        self.image_files = []
        self.embedding_file = embedding_file

        # Load embeddings if they exist
        if os.path.exists(self.embedding_file):
            self.load_embeddings()
        else:
            self.load_images_and_create_embeddings()
            self.load_embeddings()
    
    def load_images_and_create_embeddings(self, image_folder='templates'):
        """Loads images from a folder and creates embeddings."""
        self.image_files = []
        image_embeddings = []

        for file in os.listdir(image_folder):
            if file.endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(image_folder, file)
                image = Image.open(img_path).convert("RGB")
                self.image_files.append(file)

                # Process the image and create embeddings
                inputs = self.processor(images=image, return_tensors="pt")
                with torch.no_grad():
                    embeddings = self.model.get_image_features(**inputs)
                image_embeddings.append(embeddings)

        # Convert list of tensors to a single tensor
        self.image_embeddings = torch.stack(image_embeddings).squeeze()
        # Save the embeddings for future use
        torch.save((self.image_files, self.image_embeddings), self.embedding_file)
        
        return self.image_files

    def load_embeddings(self):
        """Loads previously saved embeddings."""
        self.image_files, self.image_embeddings = torch.load(self.embedding_file)

    def fetch_images_based_on_text(self, text, top_n=20):
        """Fetches images based on text input."""
        if self.image_embeddings is None or not self.image_files:
            raise ValueError("Image embeddings not created. Please load images first.")

        text = text.replace('-', '').replace('{', '').replace('}', '').replace("'", '').replace(':', '').replace('[', '').replace(']', '')
        # Process the text input to create its embedding
        inputs = self.processor(text=[text], return_tensors="pt", padding=True)
        with torch.no_grad():
            text_embedding = self.model.get_text_features(**inputs)

        # Calculate cosine similarity between text and image embeddings
        similarities = torch.nn.functional.cosine_similarity(text_embedding, self.image_embeddings)
        # Get top N matches
        top_indices = similarities.topk(top_n).indices
        # Retrieve matching images
        matched_images = [self.image_files[i] for i in top_indices]
        random.shuffle(matched_images)
        boxes, image_name = getBoxes('templates/'   + matched_images[0])
        while 'title' not in boxes.keys()  or 'product' not in boxes.keys():
            random.shuffle(matched_images)
            print(boxes.keys())
            boxes, image_name = getBoxes('templates/'   + matched_images[0])
        print('templates/'   + matched_images[0])
        return 'templates/'   + matched_images[0]
