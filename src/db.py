import os
from langchain_chroma import Chroma
from src.loader import get_documents
from src.model import get_embedding
from langchain_text_splitters import RecursiveCharacterTextSplitter

PERSIST_DIR = "chroma_db"

def load_vector_store(embeddings):
    if not os.path.exists(PERSIST_DIR):
        print("Vector DB not found. Building automatically...")

        docs = get_documents()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100
        )
        chunks = splitter.split_documents(docs)

        Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=PERSIST_DIR,
            collection_name="my_collection"
        )

    return Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings,
        collection_name="my_collection"
    )
