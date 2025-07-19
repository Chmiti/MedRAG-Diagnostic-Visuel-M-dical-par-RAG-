# DoctorIA – Diagnostic assisté par IA de la pneumonie à partir de radiographies

## Objectif

Ce projet a pour but de détecter automatiquement la présence de pneumonie à partir de radiographies thoraciques (formats DICOM ou JPG), en s’appuyant sur un modèle CNN fine-tuné, une visualisation explicative par Grad-CAM, et la génération automatique de rapports médicaux via un modèle de langage (LLM) de l'API OpenAI.

---

## Données utilisées

- Nom du dataset : RSNA Pneumonia Detection Challenge Dataset
- Format des fichiers : `.dcm` (DICOM)
- Classes : Pneumonie (1), Absence de pneumonie (0)

---

## Modèle utilisé

- Architecture de base : EfficientNet-B0 (PyTorch)
- Modifications :
  - remplacement de la couche finale par `nn.Linear(1280, 2)` (classification binaire)
- Poids fine-tunés sur le dataset RSNA

---

## Méthodes de fine-tuning

- Optimiseur : Adam
- Scheduler : ReduceLROnPlateau
- Critère de perte : CrossEntropyLoss avec pondération des classes :

```python
class_weights = torch.tensor([0.65, 1.35]).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)


## Data Augmentation (agressive)

- RandomRotation(±10°)
- HorizontalFlip(p=0.5)
- ColorJitter(brightness=0.1, contrast=0.1)
- Resize(224x224)
- Normalize

## Pourquoi prioriser le Recall ?

Dans un contexte médical, notamment pour la détection de pneumonie :
un faux négatif (ne pas détecter une pneumonie existante) peut avoir de graves conséquences cliniques,
alors qu’un faux positif peut être corrigé avec des examens complémentaires.

Objectif prioritaire : Ne rater aucun cas pathologique, quitte à avoir quelques faux positifs.

## Pipeline IA – Étapes

- Upload d'une radiographie (JPG ou DICOM)
- Prétraitement (resize, normalisation)
- Classification binaire avec EfficientNet fine-tuné
- Visualisation des zones activées par Grad-CAM
- Génération automatique du rapport médical avec l'API OpenAI (GPT-4)

Rapport médical automatique
Généré dynamiquement avec un prompt médical :

prompt = f"""
Tu es un médecin spécialiste. Le modèle IA a détecté une probabilité élevée de {'pneumonie' if label else 'absence de pneumonie'} sur l'image.
Rédige un rapport médical synthétique, en langage professionnel.
"""
API utilisée : openai.ChatCompletion avec gpt-4

## Résultats obtenus

- Accuracy : ~84.5 
- Recall : ~90.2 % (objectif prioritaire atteint)
- Precision : ~71 %
- F1-Score : ~0.80
- Loss final : ~0.12

## Courbes d'entraînement

## Exemple de prédiction avec Grad-CAM



