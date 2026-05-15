import os
import shutil
import glob

categories = ['Asthama','CROUP','LTRI','NORMAL','PNEUMONIA','URTI']

RESIZED_DIR = "RESIZED"
os.makedirs(RESIZED_DIR, exist_ok=True)

for cat in categories:
    os.makedirs(os.path.join(RESIZED_DIR, cat), exist_ok=True)

for cat in categories:

    src = os.path.join("Source", cat)
    dst = os.path.join(RESIZED_DIR, cat)

    files = glob.glob(os.path.join(src, "*.wav")) + glob.glob(os.path.join(src, "*.WAV"))

    for f in files:
        shutil.copy2(f, os.path.join(dst, os.path.basename(f)))

    print(f"{cat} done")
