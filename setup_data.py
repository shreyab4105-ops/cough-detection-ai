import os
import shutil
import glob

# Define all categories exactly as your folders are named in Source/
categories = ['Asthama', 'CROUP', 'LTRI', 'NORMAL', 'PNEUMONIA', 'URTI']

# Create RESIZED folder and subfolders for each category
RESIZED_DIR = "RESIZED"
os.makedirs(RESIZED_DIR, exist_ok=True)

for cat in categories:
    os.makedirs(os.path.join(RESIZED_DIR, cat), exist_ok=True)

# Copy .wav/.WAV files from Source/<CATEGORY> to RESIZED/<CATEGORY>
for cat in categories:
    src_folder = os.path.join("Source", cat)
    dst_folder = os.path.join(RESIZED_DIR, cat)

    if not os.path.exists(src_folder):
        print(f"Warning: Source folder '{src_folder}' does not exist!")
        continue

    # Grab all .wav/.WAV files
    files = glob.glob(os.path.join(src_folder, "*.wav")) + glob.glob(os.path.join(src_folder, "*.WAV"))

    if not files:
        print(f"Warning: No audio files found for category '{cat}' in {src_folder}!")
        continue

    for f in files:
        shutil.copy2(f, os.path.join(dst_folder, os.path.basename(f)))  # Copy instead of move

    print(f"✅ Copied {len(files)} files for category '{cat}' to {dst_folder}")

print("\nAll audio files copied successfully into RESIZED/!")
