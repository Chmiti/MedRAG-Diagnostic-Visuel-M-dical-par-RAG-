import os
import faiss
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel

# === CONFIGURATION ===
INPUT_FOLDER = "data/test_jpg"
INDEX_FILE = "data/index_test.faiss"
BATCH_SIZE = 64  # ⚡

# === Chargement du modèle CLIP
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name).to(device)
processor = CLIPProcessor.from_pretrained(model_name)

# === Récupération des fichiers
files = sorted([f for f in os.listdir(INPUT_FOLDER) if f.endswith(".jpg")])
print(f"🔍 {len(files)} images à encoder...")

# === Encodage par batch
vectors = []
for i in tqdm(range(0, len(files), BATCH_SIZE), desc="🧠 Encodage CLIP (test)"):
    batch_files = files[i:i + BATCH_SIZE]
    images = [Image.open(os.path.join(INPUT_FOLDER, f)).convert("RGB") for f in batch_files]
    inputs = processor(images=images, return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        outputs = model.get_image_features(**inputs)
        batch_embeddings = outputs.cpu().numpy()
        batch_embeddings /= np.linalg.norm(batch_embeddings, axis=1, keepdims=True)

    vectors.append(batch_embeddings)

# === Création de l'index
vectors = np.vstack(vectors).astype("float32")
index = faiss.IndexFlatL2(vectors.shape[1])
index.add(vectors)
faiss.write_index(index, INDEX_FILE)
print(f"✅ Index FAISS enregistré : {INDEX_FILE}")
