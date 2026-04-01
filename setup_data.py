import os
import shutil
import glob

def setup():
    # Define all categories
    categories = ['Asthama', 'CROUP', 'LTRI', 'NORMAL', 'PNEUMONIA', 'URTI']
    
    # Create RESIZED folder and subfolders for each category
    for cat in categories:
        os.makedirs(f'RESIZED/{cat}', exist_ok=True)
    
    # Move files from Source/<CATEGORY> to RESIZED/<CATEGORY>
    for cat in categories:
        src_folder = f'Source/{cat}'
        dst_folder = f'RESIZED/{cat}'

        if not os.path.exists(src_folder):
            print(f"Warning: Source folder '{src_folder}' does not exist!")
            continue

        # Grab all audio files (wav or mp3, any case)
        files = glob.glob(f'{src_folder}/*.*')
        files = [f for f in files if f.lower().endswith(('.WAV', '.mp3'))]

        if not files:
            print(f"Warning: No audio files found for category '{cat}' in {src_folder}!")
            continue

        for f in files:
            shutil.move(f, os.path.join(dst_folder, os.path.basename(f)))
        
        print(f"Moved {len(files)} files for category '{cat}' to {dst_folder}")

    print("✅ All categories organized successfully in RESIZED/.")

if __name__ == "__main__":
    setup()
