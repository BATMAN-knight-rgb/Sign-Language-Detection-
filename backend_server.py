import torch
import torch.nn as nn
from flask import Flask, request, jsonify
from PIL import Image
import io
import os
import numpy as np
import torchvision.transforms as transforms

# Define the model architecture
class SignLanguageCNN(nn.Module):
    def __init__(self, num_classes=36):
        super(SignLanguageCNN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Linear(128 * 25 * 25, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.model(x)

# Initialize Flask app
app = Flask(__name__)

# Class names mapping
class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
               'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
               'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
               'U', 'V', 'W', 'X', 'Y', 'Z']

# Load the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SignLanguageCNN(num_classes=36)
model_path = 'saved_models/sign_language_cnn.pth'
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Preprocessing transformation
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((200, 200)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get image from POST request
        image_bytes = request.data
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Transform image
        image = transform(image)
        image = image.unsqueeze(0)  # Add batch dimension
        image = image.to(device)

        # Predict
        with torch.no_grad():
            output = model(image)
            _, predicted = output.max(1)
        
        predicted_class = predicted.item()
        gesture = class_names[predicted_class]
        
        return jsonify({"gesture": gesture})
    
    except Exception as e:
        print(f"Prediction error: {str(e)}")  # Log the error for debugging
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
