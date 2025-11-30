# mnist_infer_cv2.py
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path

model = None
device = None

# ---- 1) Rebuild the SAME model you trained ----
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

# ---- 2) Minimal preprocessing for an OpenCV image ----
def preprocess_as_mnist(img_org, invert=False):
    # MNIST is 1×28×28 grayscale, normalized with mean/std below
    # if img_org is not empty
    if img_org is not None:
        img = cv2.resize(img_org, (28, 28), interpolation=cv2.INTER_AREA)
    # Any pixel not 0 becomes 255
    img = np.where(img != 0, 255, img)
    cv2.imwrite("Resized Input.png", img)
    if invert:                                           # if your digit is black-on-white vs white-on-black
        img = 255 - img
    img = img.astype("float32") / 255.0
    # Standard MNIST normalization (PyTorch examples)
    mean, std = 0.1307, 0.3081
    img = (img - mean) / std
    img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)  # (1,1,28,28)
    return img

# ---- 3) Load checkpoint + run inference ----
@torch.no_grad()
def run_cnn(img_org, invert=False):
    global model, device

    # Ensure model is loaded
    if model is None:
        load_model()

    x = preprocess_as_mnist(img_org, invert=invert).to(device)

    logits = model(x)
    probs = F.softmax(logits, dim=1)[0]
    pred = int(torch.argmax(probs).item())
    print(f"Predicted digit: {pred}")
    print("Probabilities:", probs.cpu().numpy())
    return pred


def load_model():
    """Load the model into memory and return the device used."""
    global model, device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN().to(device)
    # path to saved model (file located in the same `cnn` directory as this script)
    model_path = str(Path(__file__).resolve().parent / "mnist_cnn.pth")
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()


def close_model():
    """Free model resources (move to CPU, delete and clear cache)."""
    global model, device
    if model is None:
        return
    try:
        # move to CPU to release GPU tensors
        model.to("cpu")
    except Exception:
        pass
    del model
    model = None
    # clear cached memory
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass

if __name__ == "__main__":
    load_model()
