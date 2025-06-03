import pandas as pd
import json
import os

# === Chemins ===
CSV_PATH = "C:/GitHub/Diagnostic Visuel Médical par RAG/data/stage_2_detailed_class_info.csv"
OUTPUT_PATH = "data/metadata.json"

# === Chargement du CSV
df = pd.read_csv(CSV_PATH)

# === Création du dictionnaire image → diagnostic
metadata = {}
for _, row in df.iterrows():
    image_name = f"{row['patientId']}.jpg"
    label = row['class']
    metadata[image_name] = label

# === Sauvegarde en JSON
with open(OUTPUT_PATH, "w") as f:
    json.dump(metadata, f, indent=2)

print(f"✅ metadata.json généré avec {len(metadata)} entrées")
