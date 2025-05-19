from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain.vectorstores import Chroma

def load_pdf(db_path, emb_model, pdf_path):
    # Cargar el PDF
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    # Dividir el texto en fragmentos
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    all_splits = text_splitter.split_documents(docs)

    # Crear embeddings con el modelo local
    local_embeddings = OllamaEmbeddings(model=emb_model)

    # Crear base vectorial persistente
    vectorstore = Chroma.from_documents(
        documents=all_splits,
        embedding=local_embeddings,
        persist_directory=db_path
    )
    #vectorstore.persist()  # Guarda los vectores en disco

    return vectorstore
