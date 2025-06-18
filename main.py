# === Imports ===
# Old: CLI-based imports
# import os

# New: Streamlit + core logic
import streamlit as st
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOllama
from langchain.chains import RetrievalQA

# === Streamlit UI Setup ===
st.set_page_config(page_title="AskMyPDF", layout="centered")
st.title("📄 AskMyPDF - Chat with your PDF (Free & Local)")
st.markdown("Upload a PDF and ask questions powered by a local LLM via Ollama.")

# === File Upload ===
# Old: Command-line file path input
# pdf_path = input("Enter the path to your PDF file: ").strip()
# while not os.path.exists(pdf_path):
#     print("❌ File not found. Please enter a valid path.")
#     pdf_path = input("Enter the path to your PDF file: ").strip()

# New: Streamlit drag-and-drop upload
uploaded_file = st.file_uploader("📤 Upload a PDF", type=["pdf"])

if uploaded_file is not None:
    with st.spinner("Processing your PDF..."):
        # Save uploaded file to a temp location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            pdf_path = tmp_file.name

        # === Load and Chunk PDF ===
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = splitter.split_documents(pages)

        # === Embedding & Vector DB ===
        # Old: embeddings = OllamaEmbeddings(model="nomic-embed-text")
        embeddings = OllamaEmbeddings(model="all-minilm")  # ✅ locally available model

        db = FAISS.from_documents(docs, embeddings)

        # === LLM Setup and QA Chain ===
        llm = ChatOllama(model="llama3")
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever())

    st.success("✅ PDF loaded and ready!")

    # === User Query Input ===
    # Old: Infinite loop with input()
    # while True:
    #     query = input("\nAsk something about the PDF (or 'exit'): ")
    #     if query.lower() == "exit":
    #         break
    #     answer = qa_chain.invoke(query)
    #     print("\nAnswer:", answer)

    # New: Streamlit input + invoke
    query = st.text_input("💬 Ask a question about the PDF")

    if query:
        with st.spinner("Thinking..."):
            answer = qa_chain.invoke(query)
        st.success("💡 Answer:")
        st.write(answer["result"] if isinstance(answer, dict) and "result" in answer else answer)

    # Optional: Clean up temp file
    os.remove(pdf_path)