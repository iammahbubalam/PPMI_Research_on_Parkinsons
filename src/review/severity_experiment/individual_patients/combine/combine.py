#!/usr/bin/env python3
"""
Combine 6 patient progression images into a single 2x3 grid at 300 DPI
"""

from PIL import Image
import os

# Patient IDs to combine (from the attached images)
patient_ids = ['3028', '3056', '3105', '3323', '3385', '3792']

# Base directory
base_dir = '/home/m8m/Projects/PPMI_Research_on_Parkinsons/src/review/severity_experiment/individual_patients'

# Load all images
images = []
for patient_id in patient_ids:
    img_path = os.path.join(base_dir, f'patient_{patient_id}.png')
    
    if not os.path.exists(img_path):
        print(f"Warning: {img_path} not found!")
        continue
    
    img = Image.open(img_path)
    images.append(img)
    print(f"Loaded Patient {patient_id}: {img.size}")

if len(images) != 6:
    print(f"Error: Expected 6 images, found {len(images)}")
    exit(1)

# Get dimensions (assume all images same size)
width, height = images[0].size

# Add padding between images
padding = 50  # pixels

# Calculate combined image size (2 rows × 3 columns)
combined_width = width * 3 + padding * 2
combined_height = height * 2 + padding * 1

# Create new image with white background
combined = Image.new('RGB', (combined_width, combined_height), 'white')

# Paste images in 2x3 grid
for idx, img in enumerate(images):
    row = idx // 3
    col = idx % 3
    
    x = col * (width + padding)
    y = row * (height + padding)
    
    combined.paste(img, (x, y))
    print(f"Placed Patient {patient_ids[idx]} at ({x}, {y})")

# Save at 300 DPI
output_path = os.path.join(base_dir, 'combine', 'combined_patients_300dpi.png')
combined.save(output_path, dpi=(300, 300))

print(f"\n✅ Combined image saved to: {output_path}")
print(f"   Resolution: 300 DPI")
print(f"   Dimensions: {combined_width} × {combined_height} pixels")
print(f"   Layout: 2 rows × 3 columns")
print(f"   Padding: {padding} pixels between images")
