import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image, ImageDraw
import os

# Load the model
model = models.mobilenet_v3_large(weights=None)
num_features = model.classifier[3].in_features
model.classifier[3] = nn.Linear(num_features, 25)
# model.load_state_dict(torch.load('model2.pth'))
model.load_state_dict(torch.load('model2.pth', map_location=torch.device('cpu')))  # Ensure it works on CPU

# Assuming you are using a GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()  # Set the model to evaluation mode

# Define the transformation
data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# List of classes
num_classes = ['Tomato Bacterial Spot', 'Tomato Early Blight', 'Tomato Late Blight', 'Tomato Leaf Mold', 'Tomato Septoria Leaf Spot', 'Tomato Spider Mites Two Spotted Spider Mite', 'Tomato Target Spot', 'Tomato Tomato Yellow Leaf Curl Virus', 'Tomato Tomato Mosaic Virus', 'Tomato Healthy', 'Apple Healthy', 'Apple Rust', 'Apple Scab', 'Corn Blight', 'Corn Common Rust', 'Corn Gray Leaf Spot', 'Corn Healthy', 'Cucumber Anthracnose', 'Cucumber Bacterial Wilt', 'Cucumber Downy Mildew', 'Cucumber Gummy Stem Blight', 'Cucumber Healthy', 'Wheat Healthy', 'Wheat septoria', 'Wheat stripe rust']

def predict_and_save(image_path, processed_path):
    img = Image.open(image_path)
    transformed_img = data_transforms(img).unsqueeze(0)  # Add batch dimension
    transformed_img = transformed_img.to(device)

    with torch.no_grad():  # No need to track gradients for inference
        outputs = model(transformed_img)
        probabilities = nn.functional.softmax(outputs, dim=1)
        predicted = probabilities.argmax(1)
        predicted_class = predicted.item()

    # Get predicted class and confidence for each class in percentage
    result = {
        "predicted_class": num_classes[predicted_class],
        "probabilities": {num_classes[idx]: f"{prob.item() * 100:.2f}%" for idx, prob in enumerate(probabilities[0])}
    }

    # Save the processed image
    img_draw = ImageDraw.Draw(img)
    img_draw.text((10, 10), f"Class: {num_classes[predicted_class]}", fill="red")
    img.save(processed_path)

    return result
