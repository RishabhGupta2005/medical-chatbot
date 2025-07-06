import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Load the CSV
df = pd.read_csv("symptoms_dataset.csv")

# Use a generation-capable model
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def predict_department(symptom_input):
    symptom_input = symptom_input.lower()

    # Try exact match with dataset
    for _, row in df.iterrows():
        if row["symptom"] in symptom_input:
            return f"📋 From dataset: {row['department']}"

    # Fallback: Use LLM
    prompt = (
        f"A patient has these symptoms: '{symptom_input}'. "
        "Which medical department (e.g., Cardiology, Neurology, Dermatology, etc.) should they consult? "
        "Just return the department name."
    )

    # Tokenize and generate
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=50)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return f"🤖 From LLM: {response.strip()}"
