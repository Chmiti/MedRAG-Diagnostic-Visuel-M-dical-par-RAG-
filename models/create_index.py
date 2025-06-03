import os
import faiss
import torch
import numpy as np
import json
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel

# === Chemins ===
IMAGE_FOLDER = "data/images"
METADATA_FILE = "data/metadata.json"
INDEX_FILE = "data/index.faiss"

# === Chargement du modèle CLIP
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name).to(device)
processor = CLIPProcessor.from_pretrained(model_name)

# === Initialisation
vectors = []
filenames = []

# === Encodage de chaque image
files = [f for f in os.listdir(IMAGE_FOLDER) if f.endswith(".jpg")]
print(f"🔍 {len(files)} images à encoder...")

for filename in tqdm(files, desc="🧠 Encodage CLIP"):
    try:
        image_path = os.path.join(IMAGE_FOLDER, filename)
        image = Image.open(image_path).convert("RGB")

        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.get_image_features(**inputs)
            embedding = outputs[0].cpu().numpy()
            embedding = embedding / np.linalg.norm(embedding)

        vectors.append(embedding)
        filenames.append(filename)

    except Exception as e:
        print(f"❌ Erreur sur {filename} : {e}")

# === Création de l’index FAISS
dimension = vectors[0].shape[0]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(vectors).astype("float32"))

# === Sauvegarde de l’index
faiss.write_index(index, INDEX_FILE)
print(f"✅ Index FAISS enregistré : {INDEX_FILE}")
