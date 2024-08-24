import os
import logging
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from constants import CHROMA_SETTINGS

persist_directory = "db"

# Logging configuration
logging.basicConfig(level=logging.INFO)

def load_documents(directory):
    documents = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".pdf"):
                try:
                    loader = PyPDFLoader(os.path.join(root, file))
                    documents.extend(loader.load())
                    logging.info(f"Loaded {file}")
                except Exception as e:
                    logging.error(f"Failed to load document {file}: {str(e)}")
    return documents

def split_texts(documents, chunk_size=500, chunk_overlap=100):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(documents)

def create_vector_store(texts, persist_directory):
    logging.info("Loading sentence transformers model")
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    logging.info(f"Creating embeddings. This may take some time...")
    db = Chroma.from_documents(texts, embeddings, persist_directory=persist_directory, client_settings=CHROMA_SETTINGS)
    db.persist()
    return db

def main():
    logging.info("Starting document ingestion process...")
    documents = load_documents("docs")
    if not documents:
        logging.warning("No documents were loaded.")
        return
    
    logging.info("Splitting documents into chunks...")
    texts = split_texts(documents)
    
    logging.info("Creating embeddings and vector store...")
    create_vector_store(texts, persist_directory)
    
    logging.info("Ingestion complete! You can now run privateGPT.py to query your documents")

if __name__ == "__main__":
    main()
