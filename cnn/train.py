# ==============================
# CNN on MNIST — Full Version with Predictions + Auto Download
# ==============================

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
# from google.colab import files  # comment this line if not using Colab
import os
os.chdir('cnn')  # Change to the directory where cnn.py is located

# -----------------------------
# 1. Define CNN Model
# -----------------------------
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 64 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# -----------------------------
# 2. Load Data
# -----------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
testloader = DataLoader(testset, batch_size=64, shuffle=False)

# -----------------------------
# 3. Setup
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
best_acc = 0.0
train_losses, test_losses = [], []

# -----------------------------
# 4. Training Loop
# -----------------------------
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_train_loss = running_loss / len(trainloader)
    train_losses.append(avg_train_loss)

    # Validation
    model.eval()
    test_loss = 0.0
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_test_loss = test_loss / len(testloader)
    test_losses.append(avg_test_loss)
    accuracy = 100 * correct / total

    print(f"Epoch [{epoch+1}/{num_epochs}] "
          f"| Train Loss: {avg_train_loss:.4f} "
          f"| Test Loss: {avg_test_loss:.4f} "
          f"| Accuracy: {accuracy:.2f}%")

    # Save best model
    if accuracy > best_acc:
        best_acc = accuracy
        torch.save(model.state_dict(), "best_mnist_cnn.pth")

print(f"\n✅ Training complete. Best Accuracy: {best_acc:.2f}%")
print("💾 Best model saved as best_mnist_cnn.pth")

# -----------------------------
# 5. Plot Loss Graphs
# -----------------------------
plt.figure(figsize=(8,5))
plt.plot(range(1, num_epochs+1), train_losses, label='Training Loss')
plt.plot(range(1, num_epochs+1), test_losses, label='Testing Loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training vs Testing Loss")
plt.legend()
plt.grid(True)
plt.show()

# -----------------------------
# 6. Show Example Predictions
# -----------------------------
model.load_state_dict(torch.load("best_mnist_cnn.pth"))
model.eval()

# Get a batch of test images
dataiter = iter(testloader)
images, labels = next(dataiter)
images, labels = images.to(device), labels.to(device)

# Predict
outputs = model(images)
_, preds = torch.max(outputs, 1)

# Display first 10 test images
plt.figure(figsize=(12, 5))
for i in range(10):
    plt.subplot(2, 5, i+1)
    img = images[i].cpu().numpy().squeeze()
    plt.imshow(img, cmap='gray')
    plt.title(f"Pred: {preds[i].item()}\nTrue: {labels[i].item()}")
    plt.axis('off')
plt.suptitle("Example Predictions (Predicted vs Actual)", fontsize=14)
plt.show()

# -----------------------------
# 7. Auto-Download Best Model
# -----------------------------
if os.path.exists("best_mnist_cnn.pth"):
    print("⬇️ Downloading the best trained model...")
    # files.download("best_mnist_cnn.pth")
else:
    print("⚠️ No model found to download.")
