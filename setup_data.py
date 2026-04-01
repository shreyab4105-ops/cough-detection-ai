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
        
        # Debug: check if source folder exists
        if not os.path.exists(src_folder):
            print(f"WARNING: Source folder not found: {src_folder}")
            continue
        
        files = glob.glob(f'{src_folder}/*.wav')
        print(f"DEBUG: Looking in {src_folder}, found {len(files)} files")
        
        if not files:
            print(f"Warning: No .wav files found for category '{cat}' in {src_folder}!")
            continue
        
        for f in files:
            shutil.move(f, os.path.join(dst_folder, os.path.basename(f)))
        print(f"Moved {len(files)} files for category '{cat}' to RESIZED/{cat}")
    
    print("All categories organized successfully in RESIZED/.")

if __name__ == "__main__":
    setup()
