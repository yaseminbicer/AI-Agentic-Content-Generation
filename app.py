import streamlit as st
from dotenv import load_dotenv

load_dotenv()


st.set_page_config(page_title="🧠 AI Content Generator", layout="centered")
st.title("🧠 Agentic Content Generator")

task_type = st.selectbox("🎯 Select content type:", [
    "Fitness Plan Generator", "Essay Generator", "Social Media Content Generator"
])

prompt = st.text_area("💬 Enter your prompt:")

from agent_router import run_agent

if st.button("🚀 Generate"):
    if not prompt:
        st.warning("⚠️ Please enter a prompt.")
    else:
        with st.spinner("🤖 Generating response..."):
            try:
                response = run_agent(task_type, prompt)
            except Exception as e:
                st.error(f"❌ Error: {e}")
                response = ""

        if response:
            with st.chat_message("assistant"):
                st.markdown("💬 Response:")
                st.markdown(response)
