from src.loader import get_documents
from src.model import get_embedding
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
import os

PERSIST_DIR = "chroma_db"

def ingest():
    if os.path.exists(PERSIST_DIR):
        print("Vector DB already exists. Delete it to re-ingest.")
        return

    print("Building vector database...")

    docs = get_documents()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(docs)

    embeddings = get_embedding()

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR,
        collection_name="my_collection"
    )



    print("Ingestion complete.")

if __name__ == "__main__":
    ingest()
