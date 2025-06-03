import gradio as gr
from rag_inference import answer_question
from PIL import Image
import os
import shutil

UPLOAD_DIR = "data/query_temp"
os.makedirs(UPLOAD_DIR, exist_ok=True)

def process(image, question):
    # Enregistre temporairement l'image
    image_path = os.path.join(UPLOAD_DIR, "query.jpg")
    image.save(image_path)

    # Lance le pipeline RAG
    try:
        response = answer_question(image_path, question)
    except Exception as e:
        response = f"❌ Erreur : {str(e)}"

    return response

# === Interface Gradio
demo = gr.Interface(
    fn=process,
    inputs=[
        gr.Image(type="pil", label="📤 Téléverse une image de radio"),
        gr.Textbox(lines=2, placeholder="Ex: Quels cas similaires montrent une pneumonie ?", label="❓ Question médicale")
    ],
    outputs=gr.Textbox(label="🧠 Réponse générée par GPT"),
    title="🩺 Diagnostic Médical Assisté par IA (MedRAG)",
    description="Téléverse une radio, pose ta question clinique, et l'IA te répond à partir de cas réels similaires."
)

if __name__ == "__main__":
    demo.launch()
