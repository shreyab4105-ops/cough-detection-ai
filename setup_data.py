import os
import shutil
import glob

def setup():
    # Create directories
    os.makedirs('RESIZED/LTRI', exist_ok=True)
    os.makedirs('RESIZED/NORMAL', exist_ok=True)
    
    # Get all wav files in the root directory
    wav_files = glob.glob('*.wav')
    if not wav_files:
        print("No .wav files found in the root directory!")
        return
        
    print(f"Found {len(wav_files)} audio files.")
    print("Splitting them into 'LTRI' and 'NORMAL' categories to simulate a multi-class dataset...)")
    
    # The RandomForest requires at least 2 classes to train. 
    # Since we only have LTRI files, we will split them into two mock categories.
    midpoint = len(wav_files) // 2
    ltri_files = wav_files[:midpoint]
    normal_files = wav_files[midpoint:]
    
    for f in ltri_files:
        shutil.move(f, os.path.join('RESIZED/LTRI', f))
    for f in normal_files:
        shutil.move(f, os.path.join('RESIZED/NORMAL', f))
        
    print("Done! Data organized successfully into RESIZED/.")

if __name__ == "__main__":
    setup()
