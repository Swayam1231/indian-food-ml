import os
from classes import CLASSES

BASE = "dataset"

for split in ["train", "val"]:
    for cls in CLASSES:
        path = os.path.join(BASE, split, cls)
        os.makedirs(path, exist_ok=True)
        print("Created:", path)
