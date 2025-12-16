from langchain_chroma import Chroma

import os
PERSIST_DIR='chroma_db'
def load_vector_store(embeddings):
    if not os.path.exists(PERSIST_DIR):
        raise RuntimeError(
            "Vector DB not found. Run `python -m src.ingest` first."
        )

    return Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings,
        collection_name="my_collection"
    )

