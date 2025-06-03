# 🧠 MedRAG — Diagnostic Visuel Médical par RAG

Un système d’assistance médicale intelligent qui utilise l’intelligence artificielle pour analyser des images médicales (radiographies, IRM...) et fournir un **diagnostic raisonné** basé sur des cas similaires documentés, grâce à la méthode **Retrieval-Augmented Generation (RAG)**.

---

## 🧩 Aperçu stratégique des tâches

| Tâche                                           | Détail                                                                                                                                                                           | Utilité                                                                             |
| ----------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| **1. Collecte et nettoyage de dataset médical** | Téléchargement d’un dataset structuré comme RSNA Pneumonia Detection ou MedPix. Nettoyage des métadonnées (texte, étiquettes médicales)                                          | Avoir une base fiable pour entraîner et indexer sans erreurs ni ambiguïtés          |
| **2. Fine-tuning de CLIP/BLIP**                 | Adapter CLIP (ou BLIP) pour encoder des images médicales spécifiques. Cela améliore la recherche de similarité (ex: trouver des radios de pneumonie similaires à l’image donnée) | Réduction massive des hallucinations en liant image ↔ cas clinique réels            |
| **3. Construction de l’index FAISS enrichi**    | Encodage des images médicales et stockage avec leurs descriptions (diagnostics, résumés médicaux, traitements associés)                                                          | Créer une base de connaissances visuelle + textuelle à interroger de manière fiable |
| **4. Mécanisme RAG intelligent**                | Lorsqu’on pose une question, on récupère les 5 cas les plus similaires → GPT n’invente rien, il **s’appuie uniquement sur ces 5 cas** pour générer sa réponse                    | Contrôle du contenu généré, fiabilité médicale, meilleure crédibilité               |
| **5. Interface Gradio claire**                  | Upload image + pose de question → Affichage : image originale, images similaires, diagnostic probable, explication détaillée                                                     | Visualisation facile + valeur ajoutée UX pour la recherche médicale                 |

---

## 📸 Fonctionnalités principales

- 🔬 Téléversement d’images médicales (ex. radio thoracique)
- 🧠 Recherche de cas similaires grâce à **CLIP fine-tuné + FAISS**
- 🤖 Génération de diagnostics à l’aide d’un **LLM** entraîné sur un corpus médical
- 💬 Interface intuitive avec **Gradio** pour poser des questions médicales et obtenir une explication détaillée
- 🔐 Réduction des hallucinations par **RAG contrôlé** (le LLM s’appuie uniquement sur des cas réels)

---

## 🛠️ Stack Technique

| Composant           | Rôle                                                        |
|---------------------|-------------------------------------------------------------|
| CLIP (fine-tuné)     | Encodage visuel des images médicales                      |
| FAISS               | Recherche rapide de similarité visuelle + textuelle        |
| LLM (Mistral / GPT) | Génération de diagnostic raisonné                          |
| Dataset MedPix / RSNA | Cas médicaux réels et annotés                          |
| Gradio              | Interface utilisateur interactive                          |
| PyTorch / Transformers | Frameworks IA utilisés                                |

---

## 🔧 Lancer le projet

```bash
# 1. Création de l’environnement virtuel
python -m venv venv
source venv/bin/activate  # ou venv\Scripts\activate sous Windows

# 2. Installation des dépendances
pip install -r requirements.txt

# 3. Lancement de l’application Gradio
python app.py
