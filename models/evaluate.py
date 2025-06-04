import os
import json
import sys

# Accès au dossier des fonctions
sys.path.append("models")
from rag_inference import answer_question

# === Chemins (TEST uniquement)
TEST_FOLDER = "data/test_jpg"
METADATA_PATH = "data/metadata_test.json"
INDEX_PATH = "data/index_test.faiss"
QUESTION = "Est-ce une pneumonie ?"

# === Chargement des réponses attendues
with open(METADATA_PATH, "r") as f:
    ground_truth = json.load(f)

# === Evaluation
total = 0
correct = 0

for filename in os.listdir(TEST_FOLDER):
    if not filename.endswith(".jpg"):
        continue

    image_path = os.path.join(TEST_FOLDER, filename)
    vrai_diagnostic = ground_truth.get(filename, "").lower().strip()

    print(f"🤖 Question à {filename}")
    reponse = answer_question(
        image_path=image_path,
        question=QUESTION,
        k=5,
        index_path=INDEX_PATH,
        metadata_path=METADATA_PATH
    ).lower()

    total += 1

    if "pneumonie" in vrai_diagnostic and "pneumonie" in reponse:
        correct += 1
    elif "pneumonie" not in vrai_diagnostic and "pneumonie" not in reponse:
        correct += 1
    else:
        print(f"❌ Mauvaise réponse sur {filename}")
        print(f"   ➤ Attendu : {vrai_diagnostic}")
        print(f"   ➤ Généré  : {reponse}")
        print()

# === Résumé
print(f"\n✅ Score final : {correct}/{total} → {100 * correct / total:.2f}% de bonnes réponses")
