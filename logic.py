import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Load the CSV
df = pd.read_csv("symptoms_dataset.csv")

# Load the model and tokenizer (only once)
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def predict_department(symptom_input):
    symptom_input = symptom_input.lower()

    # Try exact match from CSV
    for _, row in df.iterrows():
        if row["symptom"] in symptom_input:
            return {
                "source": "dataset",
                "department": row["department"],
                "explanation": f"The symptom '{row['symptom']}' is typically treated by {row['department']} specialists."
            }

    # Fallback to LLM
    prompt = (
        f"A patient has these symptoms: '{symptom_input}'. "
        "Which medical department (e.g., Cardiology, Neurology, Dermatology, etc.) should they consult? "
        "Just return the department name."
    )

    # Use FLAN-T5 to get prediction
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=50)
    department = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    # Create generic explanation
    return {
        "source": "llm",
        "department": department,
        "explanation": f"Based on the given symptoms, the patient should consult the {department} department."
    }
