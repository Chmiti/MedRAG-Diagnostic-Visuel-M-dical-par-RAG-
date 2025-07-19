import openai
import os

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))  # ou remplace par une chaîne de clé

def generate_medical_report(prediction_label, activated_zones_info=""):
    prompt = f"""
    Tu es un médecin spécialiste. Le modèle IA a détecté une probabilité élevée de {"pneumonie" if prediction_label else "absence de pneumonie"} sur l'image.
    {activated_zones_info}
    Rédige un rapport médical synthétique, en langage professionnel et compréhensible pour un autre médecin.
    """

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Tu es un médecin radiologue expérimenté."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content
