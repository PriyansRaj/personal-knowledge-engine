from src.model import get_embedding
from src.db import load_vector_store

def get_retriever():
    embeddings = get_embedding()
    vectorstore = load_vector_store(embeddings)

    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 3, "lambda_mult": 0.7}
    )
