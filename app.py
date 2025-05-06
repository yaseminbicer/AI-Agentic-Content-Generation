import streamlit as st
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
# Ortam değişkenlerini yükle

load_dotenv()
login(os.getenv("HUGGINGFACEHUB_API_TOKEN"))

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "mistralai/Mistral-7B-Instruct-v0.2"

tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16 if device == "cuda" else torch.float32).to(device)

def generate_response(prompt):
    messages = [{"role": "user", "content": prompt}]
    encoded_input = tokenizer.apply_chat_template(messages, return_tensors="pt").to(device)
    generated_output = model.generate(encoded_input, max_new_tokens=256, do_sample=True)
    return tokenizer.decode(generated_output[0], skip_special_tokens=True)

# Embedding modeli
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

# Retriever'lar
essay_retriever = Chroma(persist_directory="essay_db", embedding_function=embedding_model).as_retriever()
fitness_retriever = Chroma(persist_directory="fitness_db", embedding_function=embedding_model).as_retriever()
social_retriever = Chroma(persist_directory="social_db", embedding_function=embedding_model).as_retriever()

# Streamlit Arayüzü
st.set_page_config(page_title="AI Content Generator", layout="centered")
st.title("🧙‍♂️ Agentic Content Generation Tool")

task_type = st.selectbox("Please select a content type:", [
    "Essay Generator",
    "Fitness Plan Generator",
    "Social Media Content Generator"
])

prompt = st.text_area("Enter your prompt:")

if st.button("💪 Generate"):
    if not prompt:
        st.warning("Please enter a prompt.")
    else:
        with st.spinner("Generating response..."):
            try:
                response = generate_response(prompt)
                st.success("✅ Response generated!")
                st.write(response)
            except Exception as e:
                st.error(f"An error occurred: {e}")