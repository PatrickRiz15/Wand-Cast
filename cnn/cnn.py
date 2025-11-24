# mnist_infer_cv2.py
import cv2, torch, torch.nn as nn, torch.nn.functional as F

# ---- 1) Rebuild the SAME model you trained ----
class Net(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool  = nn.MaxPool2d(2)
        self.fc1   = nn.Linear(64 * 7 * 7, 128)
        self.fc2   = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 28->14
        x = self.pool(F.relu(self.conv2(x)))  # 14->7
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# ---- 2) Minimal preprocessing for an OpenCV image ----
def preprocess_as_mnist(img_org, invert=False):
    # MNIST is 1×28×28 grayscale, normalized with mean/std below
    # img = cv2.cvtColor(img_org, cv2.COLOR_BGR2GRAY)      # (H,W)
    img = cv2.resize(img_org, (28, 28), interpolation=cv2.INTER_AREA)
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net().to(device)
    model_path = "minist_cnn.pth"  # path to saved model
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()

    # img_org = cv2.imread(image_path, cv2.IMREAD_COLOR)
    # if img_org is None:
    #     raise FileNotFoundError(f"Couldn't read image: {image_path}")

    x = preprocess_as_mnist(img_org, invert=invert).to(device)

    logits = model(x)
    probs = F.softmax(logits, dim=1)[0]
    pred = int(torch.argmax(probs).item())
    print(f"Predicted digit: {pred}")
    print("Probabilities:", probs.cpu().numpy())

if __name__ == "__main__":
    # Quick hardcoded example — edit paths below or wrap with argparse if you want
    # run_cnn(
    #     model_path="best_mnist_cnn.pth",
    #     image_path="digit.png"
    #     # invert=True,      # set False/True depending on your image foreground/background
    # )
    pass
