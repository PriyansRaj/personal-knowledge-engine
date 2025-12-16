from langchain_community.document_loaders import PyPDFLoader,DirectoryLoader,TextLoader

def get_documents():
    pdf_loader = DirectoryLoader(
        path="./Data",
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    text_loader = DirectoryLoader(
        path="./Data",
        glob="*.txt",
        loader_cls=TextLoader
    )
    md_loader = DirectoryLoader(
        path="./Data",
        glob="*.md",
        loader_cls=TextLoader
    )
    documents = (
        pdf_loader.load() 
        + md_loader.load() 
        + text_loader.load()
    )
    return documents
