import pandas as pd
import json
import os

# === Chemins ===
CSV_PATH = "data/stage_2_detailed_class_info.csv"
OUTPUT_PATH = "data/metadata_test.json"

# === Chargement du CSV
df = pd.read_csv(CSV_PATH)

# === Création du dictionnaire image → phrase de diagnostic
metadata = {}
for _, row in df.iterrows():
    image_name = f"{row['patientId']}.jpg"
    label = row['class']

    if label == "Lung Opacity":
        description = "La radiographie montre une opacité pulmonaire, ce qui est compatible avec une pneumonie."
    else:
        description = "La radiographie ne montre aucune anomalie, elle est considérée comme normale."

    metadata[image_name] = description

# === Sauvegarde en JSON
with open(OUTPUT_PATH, "w") as f:
    json.dump(metadata, f, indent=2)

print(f"✅ metadata_test.json généré avec {len(metadata)} entrées")
