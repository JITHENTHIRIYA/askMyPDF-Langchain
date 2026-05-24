import streamlit as st
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

st.set_page_config(page_title="AskMyPDF", layout="centered")
st.title("AskMyPDF - Chat with your PDF")
st.markdown("Upload a PDF and ask questions powered by HuggingFace.")

hf_token = st.secrets.get("HF_TOKEN", os.environ.get("HF_TOKEN", ""))
if not hf_token:
    st.warning("Set **HF_TOKEN** in Streamlit secrets or as an environment variable to enable the LLM.")

uploaded_file = st.file_uploader("Upload or replace a PDF", type=["pdf"])

if "pdf_path" not in st.session_state:
    st.session_state.pdf_path = None

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if uploaded_file is not None:
    with st.spinner("Processing PDF..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            new_pdf_path = tmp_file.name

        if new_pdf_path != st.session_state.pdf_path:
            loader = PyPDFLoader(new_pdf_path)
            pages = loader.load()

            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            docs = splitter.split_documents(pages)

            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={"device": "cpu"}
            )

            db = FAISS.from_documents(docs, embeddings)

            llm = HuggingFaceEndpoint(
                repo_id="mistralai/Mistral-7B-Instruct-v0.3",
                huggingfacehub_api_token=hf_token,
                temperature=0.1,
                max_new_tokens=512,
            )

            prompt = PromptTemplate.from_template(
                "Use the following context to answer the question.\n\n"
                "Context: {context}\n\nQuestion: {question}\n\nAnswer:"
            )

            retriever = db.as_retriever()
            st.session_state.qa_chain = (
                {"context": retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)),
                 "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )

            st.session_state.pdf_path = new_pdf_path
            st.success("✅ PDF uploaded and ready!")

for q, a in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(q)
    with st.chat_message("assistant"):
        st.write(a)

query = st.chat_input("Ask a question about the current PDF")

if query:
    if not st.session_state.qa_chain:
        st.warning("Please upload a PDF first.")
    else:
        with st.chat_message("user"):
            st.write(query)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = st.session_state.qa_chain.invoke(query)
            st.write(answer)
        st.session_state.chat_history.append((query, answer))