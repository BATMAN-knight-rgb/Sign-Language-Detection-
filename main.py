import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os

# Image preprocessing
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((200, 200)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Path to your dataset
data_path = "D:\\LingunaManus\\Indian"

# Set the number of images you want per class (for quick testing)
num_images_per_class = 10  # Adjust this based on your requirement

# Modify the dataset loading to limit the number of images per class
class CustomImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None, num_images_per_class=10):
        super().__init__(root, transform)
        # Limit the number of images per class
        self.samples = self.samples[:num_images_per_class * len(self.classes)]

# Load the dataset using the custom class
train_dataset = CustomImageFolder(root=data_path, transform=transform, num_images_per_class=num_images_per_class)

# Split into train and validation datasets (80/20 split)
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_data, val_data = torch.utils.data.random_split(train_dataset, [train_size, val_size])

# Data loaders
batch_size = 32
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

# Model definition
class SignLanguageCNN(nn.Module):
    def __init__(self, num_classes=36):  # 26 letters + 10 numbers
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

# Initialize the model, criterion, and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SignLanguageCNN(num_classes=36).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 5  # You can set this based on your testing
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    train_acc = 100. * correct / total

    # Validation
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            loss = criterion(outputs, targets)
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            val_total += targets.size(0)
            val_correct += predicted.eq(targets).sum().item()

    val_acc = 100. * val_correct / val_total

    print(f'Epoch [{epoch+1}/{num_epochs}] '
          f'Train Loss: {train_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}% '
          f'| Val Loss: {val_loss/len(val_loader):.4f} | Val Acc: {val_acc:.2f}%')

# Save the trained model
os.makedirs('saved_models', exist_ok=True)
torch.save(model.state_dict(), 'saved_models/sign_language_cnn.pth')

print("Model saved successfully!")

