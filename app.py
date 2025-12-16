import streamlit as st
import tempfile
import os

from src.agent import Agent
from src.model import get_embedding, get_llm
from src.prompt import prompt

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda,
)

# ---------------- Page setup ----------------

st.set_page_config(
    page_title="Personal Knowledge Engine",
    page_icon="🧠",
    layout="centered",
)

st.title("🧠 Personal Knowledge Engine")
st.caption("RAG-powered search over your private documents")

# ---------------- Persistent agent ----------------

@st.cache_resource
def load_agent():
    return Agent()

agent = load_agent()

# ---------------- Upload section ----------------

st.subheader("📂 Upload documents (PDF / TXT / MD)")

uploaded_files = st.file_uploader(
    "Upload files (select multiple files to simulate a folder)",
    type=["pdf", "txt", "md"],
    accept_multiple_files=True,
)

# ---------------- Build TEMP RAG from uploads ----------------

@st.cache_resource
def build_temp_vectorstore(files):
    if not files:
        return None

    with tempfile.TemporaryDirectory() as tmpdir:
        documents = []

        for file in files:
            file_path = os.path.join(tmpdir, file.name)

            with open(file_path, "wb") as f:
                f.write(file.read())

            if file.name.lower().endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            else:
                loader = TextLoader(file_path)

            documents.extend(loader.load())

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
        )

        chunks = splitter.split_documents(documents)

        embeddings = get_embedding()

        # TEMP vector DB (in-memory)
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
        )

        return vectorstore

temp_vectorstore = build_temp_vectorstore(uploaded_files)

# ---------------- Mode selection ----------------

st.subheader("🔍 Query mode")

mode = st.radio(
    "Choose what to query:",
    ["My Knowledge Base", "Uploaded Files"],
    horizontal=True,
)

# ---------------- Question input ----------------

question = st.text_input(
    "Ask a question:",
    placeholder="e.g. Explain transformers based on my notes",
)

# ---------------- RAG over uploaded files ----------------

def run_upload_rag(vectorstore, question):
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 3, "lambda_mult": 0.7},
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        RunnableParallel(
            {
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough(),
            }
        )
        | prompt()
        | get_llm()
        | StrOutputParser()
    )

    return chain.invoke(question)

# ---------------- Ask button ----------------

if st.button("Ask"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking..."):
            try:
                if mode == "My Knowledge Base":
                    answer = agent.invoke(question)

                else:
                    if not temp_vectorstore:
                        st.warning("Please upload files first.")
                        st.stop()

                    answer = run_upload_rag(temp_vectorstore, question)

                st.markdown("### 📌 Answer")
                st.write(answer)

            except Exception as e:
                st.error("Something went wrong.")
                st.exception(e)
