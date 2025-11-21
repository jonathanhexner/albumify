#!/usr/bin/env python
"""
Download diverse test images for quality assessment testing.
Downloads portrait, landscape, and macro images from Unsplash.
"""

import requests
from pathlib import Path
import time

# Create output directory
output_dir = Path("examples/images")
output_dir.mkdir(parents=True, exist_ok=True)

# Unsplash direct image URLs (free to use, no API key needed)
# These are high-quality, diverse images
images = [
    {
        "name": "portrait_01.jpg",
        "url": "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=1920&q=80",
        "description": "Portrait - Professional headshot"
    },
    {
        "name": "portrait_02.jpg",
        "url": "https://images.unsplash.com/photo-1494790108377-be9c29b29330?w=1920&q=80",
        "description": "Portrait - Natural lighting"
    },
    {
        "name": "landscape_01.jpg",
        "url": "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=1920&q=80",
        "description": "Landscape - Mountain vista"
    },
    {
        "name": "landscape_02.jpg",
        "url": "https://images.unsplash.com/photo-1470071459604-3b5ec3a7fe05?w=1920&q=80",
        "description": "Landscape - Forest scene"
    },
    {
        "name": "macro_01.jpg",
        "url": "https://images.unsplash.com/photo-1519904981063-b0cf448d479e?w=1920&q=80",
        "description": "Macro - Flower detail"
    }
]

print("Downloading test images for quality assessment...")
print(f"Output directory: {output_dir}\n")

for img_info in images:
    name = img_info["name"]
    url = img_info["url"]
    description = img_info["description"]
    
    output_path = output_dir / name
    
    print(f"Downloading: {name} ({description})...")
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        with open(output_path, 'wb') as f:
            f.write(response.content)
        
        file_size = output_path.stat().st_size / 1024  # KB
        print(f"  [OK] Saved: {output_path} ({file_size:.1f} KB)")
        
        # Be polite to Unsplash
        time.sleep(0.5)
        
    except Exception as e:
        print(f"  [ERROR] Failed to download {name}: {e}")

print(f"\n[OK] Download complete! Images saved to: {output_dir}")
print(f"\nDownloaded {len(images)} images:")
for img_info in images:
    print(f"  - {img_info['name']}: {img_info['description']}")

