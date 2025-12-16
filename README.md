# 🧠 Personal Knowledge Engine
Retrieval-Augmented Generation (RAG) system for grounded question answering over private documents

---

## Overview

Personal Knowledge Engine is a Retrieval-Augmented Generation (RAG) system that allows users to query unstructured documents such as PDFs, text files, and Markdown files.

Unlike typical “chat with PDF” demos, this system is designed to prioritize correctness over fluency. If the requested information is not present in the retrieved documents, the system explicitly refuses to answer instead of hallucinating.

---

## Key Features

- **Persistent Knowledge Base**  
  One-time ingestion of documents into a vector database for fast and reusable semantic retrieval.

- **Temporary (Session-Based) RAG**  
  Supports runtime file uploads and ad-hoc querying without contaminating the main knowledge base. Uploaded documents exist only for the active session.

- **Strict Hallucination Control**  
  Responses are generated strictly from retrieved context. When information is missing, the system responds with:  
  _"I don't know based on the provided documents."_

- **Efficient Architecture**  
  Clear separation between ingestion and query pipelines. Embeddings are computed once and reused. Local embeddings ensure low latency and reliability.

- **Interactive Streamlit Interface**  
  Allows querying both persistent documents and uploaded files with transparent system behavior.

---

## System Architecture

Documents  
→ Text Chunking  
→ Embeddings  
→ Vector Database (Chroma)  
→ Retriever (MMR)  
→ Context-Grounded Prompt  
→ LLM Response  

Two retrieval modes are supported:
- **Persistent RAG** for curated long-term knowledge  
- **Temporary RAG** for session-only uploads  

---

## Tech Stack

- **Language:** Python  
- **Framework:** LangChain  
- **Vector Database:** ChromaDB  
- **Embeddings:** Sentence-Transformers (all-MiniLM-L6-v2)  
- **LLM:** DeepSeek (via Hugging Face)  
- **Frontend:** Streamlit  

---

## Project Structure

Personal-Knowledge-Engine/
│
├── app.py
├── chroma_db/
├── src/
│ ├── agent.py
│ ├── retriever.py
│ ├── db.py
│ ├── ingest.py
│ ├── loader.py
│ ├── model.py
│ └── prompt.py
│
├── requirements.txt
└── README.md

## Setup Instructions

### 1. Clone the repository


git clone git@github.com:PriyansRaj/personal-knowledge-engine.git
cd personal-knowledge-engine
### 2. Create and activate a virtual environment
bash
Copy code
python -m venv myvenv
myvenv\Scripts\activate        # Windows
source myvenv/bin/activate    # Linux / Mac
### 3. Install dependencies
bash
Copy code
pip install -r requirements.txt
### 4. Configure environment variables
Create a .env file in the project root:
HUGGINGFACEHUB_API_TOKEN=your_huggingface_token
### 5. Ingest documents (one-time step)
Place documents inside the Data/ folder and run:
python -m src.ingest
This builds the persistent vector database.

### 6. Run the application

streamlit run app.py
Open the browser at:
http://localhost:8501
