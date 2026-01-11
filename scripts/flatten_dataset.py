import os
import shutil
from pathlib import Path

BASE = "dataset/train"

def is_image(fname):
    return fname.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp"))

for cls in os.listdir(BASE):
    cls_path = os.path.join(BASE, cls)

    if not os.path.isdir(cls_path):
        continue

    print("Processing:", cls)

    # Walk all subfolders
    for root, dirs, files in os.walk(cls_path):
        for file in files:
            if not is_image(file):
                continue

            src = os.path.join(root, file)
            dst = os.path.join(cls_path, file)

            # Skip if already in root folder
            if os.path.abspath(src) == os.path.abspath(dst):
                continue

            # Avoid overwriting
            if os.path.exists(dst):
                name, ext = os.path.splitext(file)
                dst = os.path.join(
                    cls_path,
                    f"{name}_{os.urandom(4).hex()}{ext}"
                )

            shutil.move(src, dst)

    # Remove empty subfolders
    for root, dirs, files in os.walk(cls_path, topdown=False):
        for d in dirs:
            p = os.path.join(root, d)
            if os.path.isdir(p) and not os.listdir(p):
                os.rmdir(p)

print("Done flattening dataset.")
