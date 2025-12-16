from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace,HuggingFaceEmbeddings
from dotenv import load_dotenv
import os

token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

def get_llm():
    llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V3.2",
    task="text-generation",
    huggingfacehub_api_token=token,
    )

    model = ChatHuggingFace(llm=llm)
    return model


def get_embedding():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
