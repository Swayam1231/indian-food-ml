from bing_image_downloader import downloader
import os

# =========================
# CONFIG: CHANGE THIS
# =========================
CLASS_NAME = "dosa"   
QUERIES = [
    "{name} food",
    "indian {name}",
    "{name} close up",
    "{name} on plate",
    "{name} restaurant style",
]

IMAGES_PER_QUERY = 40  
BASE_DIR = "dataset/train"

# =========================
# SCRIPT
# =========================

CLASS_DIR = os.path.join(BASE_DIR, CLASS_NAME)
os.makedirs(CLASS_DIR, exist_ok=True)

for q in QUERIES:
    query = q.format(name=CLASS_NAME.replace("_", " "))
    print(f"\n=== Downloading: {query} ===\n")

    try:
        downloader.download(
            query,
            limit=IMAGES_PER_QUERY,
            output_dir=CLASS_DIR,
            adult_filter_off=True,
            force_replace=False,
            timeout=60,
            verbose=True
        )
    except Exception as e:
        print("Error for query:", query)
        print(e)

print("\nDone downloading for:", CLASS_NAME)
print("Now run: python scripts/flatten_dataset.py")
print("Then run: python scripts/clean_broken_images.py")
print("Then run: python scripts/resize_images.py")
