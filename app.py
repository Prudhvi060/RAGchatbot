import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import requests
import random
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
if not api_key or api_key.strip() == "":
    st.error("API key not found. Please set the GROQ_API_KEY in your .env file and restart the app.")
    st.stop()

# Exponential backoff request function for Groq API
def groq_request_with_retry(url, headers, payload, max_retries=5):
    wait = 1
    for attempt in range(max_retries):
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code == 429:
            if attempt == max_retries - 1:
                st.error("Too many requests (429). Please try again later.")
                return None
            delay = wait + random.uniform(0, 0.5)
            time.sleep(delay)
            wait *= 2
            continue
        elif response.status_code >= 400:
            st.error(f"Groq API error: {response.status_code} {response.text}")
            return None
        return response.json()
    return None

def get_pdf_text(pdf_docs):
    """Extract text from PDF documents."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        if not pdf_reader.pages:
            st.error(f"PDF file {pdf.name} is empty or corrupted.")
            continue
        for page in pdf_reader.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text
    return text

def get_text_chunks(text):
    """Split text into chunks for processing."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def ask_groq(context, question):
    url = "https://api.groq.com/openai/v1/chat/completions"  # OpenAI-compatible Groq endpoint
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Use the provided context to answer questions."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{question}"}
    ]
    payload = {
        "model": "gemma2-9b-it",  # Using the correct model name
        "messages": messages,
        "temperature": 0.3
    }
    result = groq_request_with_retry(url, headers, payload)
    if result and "choices" in result and len(result["choices"]) > 0:
        return result["choices"][0]["message"]["content"]
    else:
        return "No answer returned."

def user_input(user_question, context):
    """Handle user input and generate a response using Groq API."""
    try:
        answer = ask_groq(context, user_question)
        st.write("Reply:", answer)
    except Exception as e:
        st.error(f"An error occurred while processing your question: {str(e)}")

def main():
    """Main function to run the Streamlit app."""
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using Groq 💁")

    # Store context from PDF
    if 'context' not in st.session_state:
        st.session_state['context'] = ""

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question, st.session_state['context'])

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                if raw_text:
                    text_chunks = get_text_chunks(raw_text)
                    # For now, just join all chunks as context
                    st.session_state['context'] = "\n".join(text_chunks)
                    st.success(f"Processed {len(text_chunks)} chunks of text.")

if __name__ == "__main__":
    main()