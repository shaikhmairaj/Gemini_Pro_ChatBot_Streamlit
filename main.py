import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as gen_ai
from PyPDF2 import PdfReader

# Load environment variables
load_dotenv()

# Configure Streamlit page settings
st.set_page_config(
    page_title="Chat with Gemini-Pro!",
    page_icon=":brain:",
    layout="centered",
)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or st.secrets["GOOGLE_API_KEY"]

# Set up Google Gemini model
gen_ai.configure(api_key=GOOGLE_API_KEY)
model = gen_ai.GenerativeModel('gemini-2.0-flash')


def translate_role_for_streamlit(user_role):
    return "assistant" if user_role == "model" else user_role


# Initialize chat session
if "chat_session" not in st.session_state:
    st.session_state.chat_session = model.start_chat(history=[])

if "file_content" not in st.session_state:
    st.session_state.file_content = None

st.title("🤖 Gemini Pro - ChatBot")

# --- File Upload Section ---
uploaded_file = st.file_uploader("📂 Upload a PDF, TXT, or CSV file", type=["pdf", "txt", "csv"])
if uploaded_file:
    if uploaded_file.type == "application/pdf":
        reader = PdfReader(uploaded_file)
        st.session_state.file_content = "\n".join([page.extract_text() for page in reader.pages])
    elif uploaded_file.type == "text/plain":
        st.session_state.file_content = uploaded_file.read().decode("utf-8")
    elif uploaded_file.type == "text/csv":
        df = pd.read_csv(uploaded_file)
        st.session_state.file_content = df.to_string()

    st.success(f"✅ File '{uploaded_file.name}' uploaded successfully!")

# --- Display chat history ---
for message in st.session_state.chat_session.history:
    with st.chat_message(translate_role_for_streamlit(message.role)):
        st.markdown(message.parts[0].text)

# --- Chat Input ---
user_prompt = st.chat_input("Ask Gemini-Pro...")
if user_prompt:
    st.chat_message("user").markdown(user_prompt)

    # If file uploaded, include its content in the prompt
    if st.session_state.file_content:
        prompt = f"File Content:\n{st.session_state.file_content}\n\nUser Question:\n{user_prompt}"
    else:
        prompt = user_prompt

    gemini_response = st.session_state.chat_session.send_message(prompt)

    with st.chat_message("assistant"):
        st.markdown(gemini_response.text)
