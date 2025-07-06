import streamlit as st
from logic import predict_department

st.set_page_config(page_title="Medical Chatbot", page_icon="💬")
st.title("🩺 Smart Medical Symptom Checker")
st.write("Describe your symptoms below:")

user_input = st.text_input("Enter your symptom(s):")

if user_input:
    with st.spinner("Analyzing..."):
        result = predict_department(user_input)
        st.success(f"🏥 Department: {result['department']}\n\nℹ️ {result['explanation']}")
