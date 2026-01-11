import os

BASE = "dataset/train"

for name in os.listdir(BASE):
    old_path = os.path.join(BASE, name)
    if not os.path.isdir(old_path):
        continue

    new_name = name.replace(" ", "_").lower()
    new_path = os.path.join(BASE, new_name)

    if old_path != new_path:
        print(f"Renaming: {name} -> {new_name}")
        os.rename(old_path, new_path)

print("Done renaming.")
