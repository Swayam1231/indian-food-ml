import os
import random
import shutil

TRAIN = "dataset/train"
VAL = "dataset/val"

os.makedirs(VAL, exist_ok=True)

for cls in os.listdir(TRAIN):
    src_cls = os.path.join(TRAIN, cls)
    dst_cls = os.path.join(VAL, cls)
    os.makedirs(dst_cls, exist_ok=True)

    images = os.listdir(src_cls)
    random.shuffle(images)

    n_val = int(0.2 * len(images))

    for img in images[:n_val]:
        shutil.move(
            os.path.join(src_cls, img),
            os.path.join(dst_cls, img)
        )

    print(cls, "-> moved", n_val, "to val")

print("Split done.")
