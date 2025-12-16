import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, TextLoader

DATA_DIR = "Data"

def get_documents():
    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(
            "Data directory not found. Please upload documents or provide Data/ locally."
        )

    pdf_loader = DirectoryLoader(
        path=DATA_DIR,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    text_loader = DirectoryLoader(
        path=DATA_DIR,
        glob="*.txt",
        loader_cls=TextLoader
    )
    md_loader = DirectoryLoader(
        path=DATA_DIR,
        glob="*.md",
        loader_cls=TextLoader
    )

    return (
        pdf_loader.load()
        + text_loader.load()
        + md_loader.load()
    )
