import os
import pydicom
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

# === CONFIGURATION ===
CSV_PATH = "data/stage_2_detailed_class_info.csv"
DCM_FOLDER = "data/stage_2_train_images"
OUTPUT_FOLDER = "data/images"

# === Charger les IDs des patients avec label
df = pd.read_csv(CSV_PATH)
labeled_ids = set(df['patientId'].astype(str) + ".dcm")

# === Créer dossier de sortie
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# === Conversion ciblée
for filename in tqdm(os.listdir(DCM_FOLDER), desc="🩻 Conversion ciblée"):
    if filename not in labeled_ids:
        continue
    try:
        path = os.path.join(DCM_FOLDER, filename)
        dcm = pydicom.dcmread(path)
        img = dcm.pixel_array
        img = (img - np.min(img)) / (np.max(img) - np.min(img))
        img = (img * 255).astype(np.uint8)
        img_rgb = Image.fromarray(img).convert("RGB")
        output_name = filename.replace(".dcm", ".jpg")
        img_rgb.save(os.path.join(OUTPUT_FOLDER, output_name))
    except Exception as e:
        print(f"❌ Erreur sur {filename} : {e}")
