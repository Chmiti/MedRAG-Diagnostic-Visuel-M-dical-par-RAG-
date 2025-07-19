import gradio as gr
from PIL import Image
import numpy as np
import torch
import torchvision.models as models
from torchvision import transforms
import matplotlib.pyplot as plt
import cv2
import pydicom
import os
import io

from generate_report import generate_medical_report
from gradcam_utils import generate_gradcam

# === Load model ===
model = models.efficientnet_b0(weights=None)
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 2)
model.load_state_dict(torch.load("efficientnet_dicom_finetuned_final.pth", map_location='cpu'))
model.eval()

# === Image Transform ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# === DICOM handling ===
def read_dicom_as_pil(file_path):
    dicom = pydicom.dcmread(file_path)
    img = dicom.pixel_array.astype("float32")
    img = (img - img.min()) / (img.max() - img.min())
    img = (img * 255).astype("uint8")
    return Image.fromarray(img).convert("RGB")

# === Main logic ===
def classify_and_generate(file_obj):
    file_path = file_obj.name

    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".dcm":
        image = read_dicom_as_pil(file_path)
    else:
        image = Image.open(file_path).convert("RGB")

    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.softmax(output, dim=1)[0, 1].item()
        label = int(prob > 0.4)

    # Grad-CAM
    target_layer = model.features[-1]
    heatmap = generate_gradcam(model, transform(image), target_layer, class_idx=1)
    heatmap = cv2.resize(heatmap, (224, 224))
    img_np = np.array(image.resize((224, 224)))
    heatmap_color = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay_img = cv2.addWeighted(img_np, 0.6, heatmap_color, 0.4, 0)

    # Plot image to buffer
    fig, ax = plt.subplots()
    ax.imshow(overlay_img)
    ax.axis('off')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    overlay_pil = Image.open(buf)

    # Generate report
    report = generate_medical_report(label, activated_zones_info="Zones d'activation observées via Grad-CAM.")
    return report, overlay_pil

# === Gradio Interface ===
interface = gr.Interface(
    fn=classify_and_generate,
    inputs=gr.File(label="Upload DICOM (.dcm) or Image (.jpg/.png)"),
    outputs=["text", "image"],
    title="🩻 Diagnostic IA : Pneumonie",
    description="Chargez une radiographie (JPG, PNG ou DICOM) pour obtenir un diagnostic IA avec visualisation Grad-CAM."
)

if __name__ == "__main__":
    interface.launch()
