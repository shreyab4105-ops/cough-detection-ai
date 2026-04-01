import os
import shutil
import glob

def setup():
    categories = ['Asthama', 'CROUP', 'LTRI', 'NORMAL', 'PNEUMONIA', 'URTI']

    # Create RESIZED folder and subfolders for each category
    for cat in categories:
        os.makedirs(f'RESIZED/{cat}', exist_ok=True)

    # Move files recursively from Source/<CATEGORY> to RESIZED/<CATEGORY>
    for cat in categories:
        src_folder = f'Source/{cat}'  # change to 'Shreya/{cat}' if needed
        dst_folder = f'RESIZED/{cat}'

        # Recursively find all wav files
        files = glob.glob(os.path.join(src_folder, '**', '*.wav'), recursive=True)
        if not files:
            print(f"Warning: No .wav files found for category '{cat}' in {src_folder}!")
            continue

        for f in files:
            shutil.move(f, os.path.join(dst_folder, os.path.basename(f)))
        print(f"Moved {len(files)} files for category '{cat}' to RESIZED/{cat}")

    print("All categories organized successfully in RESIZED/.")

if __name__ == "__main__":
    setup()


