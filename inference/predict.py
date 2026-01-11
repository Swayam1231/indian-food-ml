import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import sys
import os

# --------------------
# Config
# --------------------
MODEL_PATH = "models/food_model.pth"
DATASET_DIR = "dataset/train"
IMAGE_SIZE = (224, 224)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------
# Load class names
# --------------------
classes = sorted([
    d for d in os.listdir(DATASET_DIR)
    if os.path.isdir(os.path.join(DATASET_DIR, d))
])

NUM_CLASSES = len(classes)

print("Loaded", NUM_CLASSES, "classes")

# --------------------
# Image transforms
# --------------------
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# --------------------
# Load model
# --------------------
model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, NUM_CLASSES)

state = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state)

model = model.to(DEVICE)
model.eval()

print("Model loaded successfully.")

# --------------------
# Predict function
# --------------------
def predict(image_path, topk=3):
    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(img)
        probs = torch.softmax(outputs, dim=1)[0]

    values, indices = torch.topk(probs, topk)

    results = []
    for v, i in zip(values, indices):
        results.append((classes[i.item()], float(v.item())))

    return results

# --------------------
# Main
# --------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python inference/predict.py path_to_image.jpg")
        sys.exit(1)

    image_path = sys.argv[1]

    if not os.path.exists(image_path):
        print("Image not found:", image_path)
        sys.exit(1)

    results = predict(image_path, topk=3)

    print("\nPrediction Results:")
    for i, (cls, prob) in enumerate(results, 1):
        print(f"{i}. {cls}  ->  {prob*100:.2f}%")
