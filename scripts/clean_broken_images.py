import os
from PIL import Image

BASE = "dataset/train"

deleted = 0
skipped_dirs = 0

def is_image(fname):
    return fname.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp"))

for cls in os.listdir(BASE):
    cls_path = os.path.join(BASE, cls)

    if not os.path.isdir(cls_path):
        continue

    for item in os.listdir(cls_path):
        path = os.path.join(cls_path, item)

        # Skip directories
        if os.path.isdir(path):
            skipped_dirs += 1
            continue

        # Skip non-image files
        if not is_image(item):
            continue

        try:
            with Image.open(path) as img:
                img.verify()
        except Exception:
            print("Deleting broken image:", path)
            try:
                os.remove(path)
                deleted += 1
            except Exception as e:
                print("Could not delete:", path, "Reason:", e)

print("Done.")
print("Deleted broken images:", deleted)
print("Skipped directories:", skipped_dirs)
