import os
import pandas as pd
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from PIL import Image
import pydicom
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import cv2

# === CONFIGURATION ===
DATA_DIR = "data/stage_2_train_images"
CSV_PATH = "data/stage_2_detailed_class_info.csv"
BATCH_SIZE = 16
EPOCHS = 15
THRESHOLD = 0.4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === CSV CHARGEMENT ===
df = pd.read_csv(CSV_PATH)
df["label"] = df["class"].map({
    "Lung Opacity": 1,
    "Normal": 0,
    "No Lung Opacity / Not Normal": 0
})
df["filename"] = df["patientId"] + ".dcm"
df = df[["filename", "label"]].dropna()

# === DATASET ===
class DICOMDataset(Dataset):
    def __init__(self, folder_path, label_df, transform=None):
        self.folder_path = folder_path
        self.label_df = label_df
        self.transform = transform
        self.file_list = self.label_df['filename'].tolist()
        self.label_dict = dict(zip(self.label_df['filename'], self.label_df['label']))

    def __getitem__(self, idx):
        filename = self.file_list[idx]
        path = os.path.join(self.folder_path, filename)
        dicom = pydicom.dcmread(path)
        img = dicom.pixel_array.astype('float32')
        img = (img - img.min()) / (img.max() - img.min())
        img = Image.fromarray((img * 255).astype('uint8')).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = int(self.label_dict[filename])
        return img, label, filename

    def __len__(self):
        return len(self.file_list)

# === TRANSFORMATIONS ===
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor()
])
transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# === DATALOADER ===
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)
train_ds = DICOMDataset(DATA_DIR, train_df, transform_train)
val_ds = DICOMDataset(DATA_DIR, val_df, transform_val)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

# === MODELE ===
weights = EfficientNet_B0_Weights.DEFAULT
model = efficientnet_b0(weights=weights)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
model = model.to(DEVICE)

# === LOSS PONDÉRÉ + OPTIMIZER ===
class_counts = df['label'].value_counts().to_dict()
total = sum(class_counts.values())
class_weights = [total / class_counts[i] for i in sorted(class_counts.keys())]
criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float).to(DEVICE))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# === ENTRAINEMENT ===
print("🔧 Début entraînement...")
train_losses, val_accuracies, val_precisions, val_recalls, val_f1s = [], [], [], [], []

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for imgs, labels, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    train_losses.append(total_loss / len(train_loader))

    # === VALIDATION ===
    model.eval()
    val_preds, val_labels = [], []
    with torch.no_grad():
        for imgs, labels, _ in val_loader:
            imgs = imgs.to(DEVICE)
            outputs = model(imgs)
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            preds = (probs > THRESHOLD).astype(int)
            val_preds.extend(preds)
            val_labels.extend(labels.numpy())

    acc = accuracy_score(val_labels, val_preds)
    prec = precision_score(val_labels, val_preds)
    rec = recall_score(val_labels, val_preds)
    f1 = f1_score(val_labels, val_preds)

    val_accuracies.append(acc)
    val_precisions.append(prec)
    val_recalls.append(rec)
    val_f1s.append(f1)

    print(f"📊 Epoch {epoch+1} | Loss: {train_losses[-1]:.4f} | Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f}")

# === SAUVEGARDE ===
torch.save(model.state_dict(), "efficientnet_dicom_finetuned_final.pth")
print("✅ Modèle sauvegardé.")

# === PLOT DES METRIQUES ===
plt.figure(figsize=(12, 6))
plt.plot(train_losses, label='Loss')
plt.plot(val_accuracies, label='Accuracy')
plt.plot(val_precisions, label='Precision')
plt.plot(val_recalls, label='Recall')
plt.plot(val_f1s, label='F1 Score')
plt.xlabel('Epoch')
plt.ylabel('Score')
plt.legend()
plt.title("Courbes d'apprentissage")
plt.grid(True)
plt.savefig("training_metrics.png")
plt.show()

# === GRAD-CAM SIMPLE ===
def generate_gradcam(model, image_tensor, class_idx=None):
    image_tensor = image_tensor.unsqueeze(0).to(DEVICE)
    model.eval()
    gradients = []
    activations = []

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def forward_hook(module, input, output):
        activations.append(output)

    final_conv = model.features[-1]
    handle1 = final_conv.register_forward_hook(forward_hook)
    handle2 = final_conv.register_backward_hook(backward_hook)

    output = model(image_tensor)
    class_idx = class_idx or output.argmax().item()
    loss = output[0, class_idx]
    model.zero_grad()
    loss.backward()

    grads = gradients[0].cpu().detach().numpy()[0]
    acts = activations[0].cpu().detach().numpy()[0]

    weights = grads.mean(axis=(1, 2))
    cam = np.zeros(acts.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * acts[i]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    cam = cam - cam.min()
    cam = cam / cam.max()

    handle1.remove()
    handle2.remove()
    return cam

# === EXEMPLE VISUALISATION ===
sample_img, _, _ = val_ds[0]
heatmap = generate_gradcam(model, sample_img)
sample_np = sample_img.permute(1, 2, 0).numpy()
sample_np = (sample_np - sample_np.min()) / (sample_np.max() - sample_np.min())

plt.imshow(sample_np)
plt.imshow(heatmap, cmap='jet', alpha=0.5)
plt.title("Grad-CAM overlay")
plt.axis('off')
plt.savefig("gradcam_overlay.png")
plt.show()
