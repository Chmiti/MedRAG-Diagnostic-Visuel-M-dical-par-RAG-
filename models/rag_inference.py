import faiss
import json
import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import os
from openai import OpenAI
from dotenv import load_dotenv

# === Chargement de la clé API OpenAI depuis .env
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# === Chargement du modèle CLIP
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name).to(device)
processor = CLIPProcessor.from_pretrained(model_name)

# === Fonction principale
def answer_question(image_path, question, k=5, index_path="data/index.faiss", metadata_path="data/metadata.json"):
    # 1. Encode l’image avec CLIP
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        embedding = model.get_image_features(**inputs)[0].cpu().numpy()
        embedding = embedding / np.linalg.norm(embedding)

    # 2. Recherche les k plus proches dans FAISS
    index = faiss.read_index(index_path)
    distances, indices = index.search(np.array([embedding]).astype("float32"), k)

    # 3. Chargement des métadonnées
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    all_files = list(metadata.keys())
    context = ""
    for idx in indices[0]:
        filename = all_files[idx]
        description = metadata.get(filename, "Aucun diagnostic trouvé.")
        context += f"- {filename} : {description}\n"

    # 4. Prompt strict pour GPT
    prompt = f"""Voici une radiographie et une question clinique :

🖼️ Image : {os.path.basename(image_path)}
❓ Question : {question}

Je t’ai retrouvé {k} cas médicaux similaires :
{context}

🧠 En te basant uniquement sur ces cas documentés, donne une réponse **courte et explicite** :

- "oui, c'est une pneumonie"
- ou "non, ce n’est pas une pneumonie"

Ne fais **aucune hypothèse** ni explication.
"""

    # 5. Appel à l’API OpenAI
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content.strip()
