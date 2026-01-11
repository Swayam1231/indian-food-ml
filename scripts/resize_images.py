import os
from PIL import Image

BASE = "dataset/train"
SIZE = (224, 224)

for cls in os.listdir(BASE):
    cls_path = os.path.join(BASE, cls)
    if not os.path.isdir(cls_path):
        continue

    for file in os.listdir(cls_path):
        path = os.path.join(cls_path, file)
        try:
            img = Image.open(path).convert("RGB")
            img = img.resize(SIZE)
            img.save(path, "JPEG", quality=90)
        except Exception:
            print("Error processing:", path)

print("Done resizing.")
