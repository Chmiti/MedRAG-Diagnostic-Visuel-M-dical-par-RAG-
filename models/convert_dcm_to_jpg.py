import os
import pydicom
import numpy as np
from PIL import Image
from tqdm import tqdm

# === CONFIGURATION ===
INPUT_FOLDER = "C:\GitHub\Diagnostic Visuel Médical par RAG/data/stage_2_train_images"
OUTPUT_FOLDER = "data/images"

# Créer le dossier de sortie s'il n'existe pas
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Lister tous les fichiers .dcm
files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith(".dcm")]

print(f"🔍 Nombre total d'images à convertir : {len(files)}")

for filename in tqdm(files, desc="🩻 Conversion en cours"):
    try:
        path = os.path.join(INPUT_FOLDER, filename)
        dcm = pydicom.dcmread(path)
        img = dcm.pixel_array

        # Normalisation [0,255]
        img = (img - np.min(img)) / (np.max(img) - np.min(img))
        img = (img * 255).astype(np.uint8)

        img_rgb = Image.fromarray(img).convert("RGB")

        output_name = filename.replace(".dcm", ".jpg")
        img_rgb.save(os.path.join(OUTPUT_FOLDER, output_name))
    except Exception as e:
        print(f"❌ Erreur sur {filename} : {e}")
