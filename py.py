import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import os                       # for working with files
import numpy as np              # for numerical computationss
import pandas as pd             # for working with dataframes
import torch                    # Pytorch module
import matplotlib.pyplot as plt # for plotting informations on graph and images using tensors
import torch.nn as nn           # for creating  neural networks
from torch.utils.data import DataLoader # for dataloaders
from PIL import Image           # for checking images
import torch.nn.functional as F # for functions for calculating loss
import torchvision.transforms as transforms   # for transforming images into tensors
from torchvision.utils import make_grid       # for data checking
from torchvision.datasets import ImageFolder  # for working with classes and images
from torchsummary import summary              # for getting the summary of our model


# Load the model
model_path = '/home/hacker69i/Desktop/python/PlantVillage-Dataset/plant-disease-model-complete.pth'  # Update with your model path
model = torch.load(model_path, map_location=torch.device('cpu'))
model.eval()

# Transformation without resizing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image
])

def predict_image(img, model):
    """Converts image to tensor and returns the predicted class with the highest probability"""
    # Convert to a tensor
    img = transform(img).unsqueeze(0)
    # Get predictions from the model
    with torch.no_grad():
        outputs = model(img)
        _, predicted = torch.max(outputs, 1)
        return predicted.item()

def predict_user_image():
    # Get user input for image path
    user_input = input("Enter the image path: ")
    image = Image.open(user_input)
    predicted_class = predict_image(image, model)
    print(f"The predicted class index is: {predicted_class}")
    
    # Visualize the image and its predicted label
    plt.imshow(image)
    plt.title(f'Predicted: {predicted_class}')
    plt.axis('off')
    plt.show()

    return image

# Call the function to predict the user-provided image
user_image = predict_user_image()
