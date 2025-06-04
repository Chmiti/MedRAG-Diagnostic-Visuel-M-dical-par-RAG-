import os
import faiss
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel

# === CONFIGURATION ===
INPUT_FOLDER = "data/images"  # ou "data/test_jpg"
INDEX_FILE = "data/index.faiss"  # ou "data/index_test.faiss"
BATCH_SIZE = 64  # 👈 Ajustable selon ta RAM / VRAM

# === Chargement du modèle CLIP
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name).to(device)
processor = CLIPProcessor.from_pretrained(model_name)

# === Chargement des fichiers image
files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith(".jpg")]
print(f"🔍 {len(files)} images à encoder...")

# === Initialisation
all_embeddings = []
filenames = []

# === Traitement en batchs
for i in tqdm(range(0, len(files), BATCH_SIZE), desc="🧠 Encodage CLIP (batched)"):
    batch_files = files[i:i + BATCH_SIZE]
    images = []

    for filename in batch_files:
        path = os.path.join(INPUT_FOLDER, filename)
        try:
            image = Image.open(path).convert("RGB")
            images.append(image)
            filenames.append(filename)
        except Exception as e:
            print(f"❌ Erreur sur {filename} : {e}")
    
    # Transformer le batch d'images
    inputs = processor(images=images, return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        outputs = model.get_image_features(**inputs)
        embeddings = outputs.cpu().numpy()
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    all_embeddings.append(embeddings)

# === Concaténation des vecteurs
vectors = np.concatenate(all_embeddings).astype("float32")

# === Création de l’index FAISS
dimension = vectors.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(vectors)

# === Sauvegarde de l’index
faiss.write_index(index, INDEX_FILE)
print(f"✅ Index FAISS enregistré : {INDEX_FILE}")
