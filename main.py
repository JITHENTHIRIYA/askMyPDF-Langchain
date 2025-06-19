import streamlit as st
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOllama
from langchain.chains import RetrievalQA

st.set_page_config(page_title="AskMyPDF", layout="centered")
st.title("AskMyPDF - Chat with your PDF (Free & Local)")
st.markdown("Upload a PDF and ask questions powered by a local LLM via Ollama.")

uploaded_file = st.file_uploader("Upload or replace a PDF", type=["pdf"])

if "pdf_path" not in st.session_state:
    st.session_state.pdf_path = None

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if uploaded_file is not None:
    with st.spinner("Processing new PDF..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            new_pdf_path = tmp_file.name

        if new_pdf_path != st.session_state.pdf_path:
            loader = PyPDFLoader(new_pdf_path)
            pages = loader.load()

            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            docs = splitter.split_documents(pages)

            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            db = FAISS.from_documents(docs, embeddings)

            llm = ChatOllama(model="llama3")
            st.session_state.qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=db.as_retriever()
            )

            st.session_state.pdf_path = new_pdf_path
            st.success("PDF uploaded and ready!")

for q, a in st.session_state.chat_history:
    st.markdown(f"<div style='background-color:#f0f2f6;padding:10px;border-radius:10px;margin-bottom:5px'><b>You:</b> {q}</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='background-color:#e0ffe0;padding:10px;border-radius:10px;margin-bottom:15px'><b>Bot:</b> {a}</div>", unsafe_allow_html=True)

query = st.chat_input("Ask a question about the current PDF")

if query and st.session_state.qa_chain:
    with st.spinner("Thinking..."):
        response = st.session_state.qa_chain.invoke(query)

    if isinstance(response, dict) and "result" in response:
        answer = response["result"]
    else:
        answer = str(response)

    st.session_state.chat_history.append((query, answer))

    st.markdown(f"<div style='background-color:#f0f2f6;padding:10px;border-radius:10px;margin-bottom:5px'><b>You:</b> {query}</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='background-color:#e0ffe0;padding:10px;border-radius:10px;margin-bottom:15px'><b>Bot:</b> {answer}</div>", unsafe_allow_html=True)