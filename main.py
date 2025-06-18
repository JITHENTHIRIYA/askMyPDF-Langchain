from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOllama
# Load your PDF
loader = PyPDFLoader("document.pdf")
pages = loader.load()

# Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = splitter.split_documents(pages)

# Create embeddings and vector store
embeddings = OllamaEmbeddings(model="nomic-embed-text")  # Or try "all-minilm"
db = FAISS.from_documents(docs, embeddings)

# Setup LLM
llm = ChatOllama(model="llama3")

# Build Retrieval QA chain
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever())

# Ask user
while True:
    query = input("\nAsk something about the PDF (or 'exit'): ")
    if query.lower() == "exit":
        break
    answer = qa_chain.run(query)
    print("\nAnswer:", answer)