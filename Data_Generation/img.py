import io
import os
import time
from pathlib import Path

import requests
from PIL import Image
from ddgs import DDGS

# ==========================
# Configuration
# ==========================

QUERIES_FILE = "queries.txt"      # One query per line
OUTPUT_DIR = "downloads"          # Base output directory
OUTPUT_FOLDER = "Elon Musk"         # All images are saved here
IMAGES_PER_QUERY = 3              # Images to download for each query
TIMEOUT = 15                      # Request timeout (seconds)

# ==========================

# Create the output folder
SAVE_DIR = Path(OUTPUT_DIR) / OUTPUT_FOLDER
SAVE_DIR.mkdir(parents=True, exist_ok=True)


def sanitize_filename(name):
    """Remove invalid filename characters."""
    invalid = '<>:"/\\|?*'
    for char in invalid:
        name = name.replace(char, "_")
    return name.strip()


def download_as_jpeg(url, save_path):
    """Download an image and save it as JPEG."""
    try:
        response = requests.get(
            url,
            timeout=TIMEOUT,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 Chrome/126 Safari/537.36"
                )
            },
        )

        response.raise_for_status()

        image = Image.open(io.BytesIO(response.content))

        # Convert RGBA/WebP/etc. to RGB
        if image.mode != "RGB":
            image = image.convert("RGB")

        image.save(save_path, "JPEG", quality=95)

        return True

    except Exception as e:
        print(f"      Failed: {e}")
        return False


def process_query(query):
    """Search and download images for a single query."""
    print(f"\nSearching: {query}")

    downloaded = 0

    try:
        with DDGS() as ddgs:

            results = ddgs.images(
                query,
                max_results=IMAGES_PER_QUERY * 3,
            )

            for result in results:

                if downloaded >= IMAGES_PER_QUERY:
                    break

                url = result.get("image")

                if not url:
                    continue

                filename = (
                    f"{sanitize_filename(query)}_{downloaded + 1:03}.jpg"
                )
                save_path = SAVE_DIR / filename

                print(f"   Downloading {downloaded + 1}: {url}")

                if download_as_jpeg(url, save_path):
                    downloaded += 1

                time.sleep(0.3)

    except Exception as e:
        print(f"Search failed: {e}")

    print(f"Downloaded {downloaded} image(s).")


def main():
    """Read queries and download images."""

    if not os.path.exists(QUERIES_FILE):
        print(f"{QUERIES_FILE} not found.")
        return

    with open(QUERIES_FILE, "r", encoding="utf-8") as f:
        queries = [line.strip() for line in f if line.strip()]

    print(f"Loaded {len(queries)} queries.")
    print(f"Saving images to: {SAVE_DIR}\n")

    for query in queries:
        process_query(query)

    print("\nDone!")


if __name__ == "__main__":
    main()