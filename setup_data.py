import os
import shutil
import glob

# Categories
categories = ['Asthama', 'CROUP', 'LTRI', 'NORMAL', 'PNEUMONIA', 'URTI']

RESIZED_DIR = "RESIZED"
os.makedirs(RESIZED_DIR, exist_ok=True)

for cat in categories:
    os.makedirs(os.path.join(RESIZED_DIR, cat), exist_ok=True)

for cat in categories:
    src_folder = os.path.join("Source", cat)
    dst_folder = os.path.join(RESIZED_DIR, cat)

    if not os.path.exists(src_folder):
        print(f"Warning: Source folder '{src_folder}' does not exist!")
        continue

    # Case-insensitive .wav matching
    files = [f for f in os.listdir(src_folder) if f.lower().endswith('.wav')]

    if not files:
        print(f"Warning: No audio files found for category '{cat}' in {src_folder}!")
        continue

    for f in files:
        shutil.move(os.path.join(src_folder, f), os.path.join(dst_folder, f))

    print(f"✅ Moved {len(files)} files for category '{cat}' to {dst_folder}")

print("\nAll audio files organized successfully into RESIZED/!")
