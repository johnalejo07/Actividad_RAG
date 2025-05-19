import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from rag import load_pdf
import tempfile

st.title("🤖 Chatbot IA con PDF + Ollama")
st.write("Consulta cualquier documento PDF usando un modelo local.")
USER_ICON = "🧑"
BOT_ICON = "🧠"

# Inicialización del estado del chat
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Inicialización del vectorstore en sesión (si no existe)
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# Inicialización del modelo LLM
llm = ChatOllama(model="llama3.1:8b", device="cpu")
# llm = ChatOllama(model="llama2:7b", device="cpu")



# Prompt con la pregunta incluida
prompt_template = ChatPromptTemplate.from_template("""
Eres un asistente que responde únicamente usando el contenido del documento proporcionado. 
Si no encuentras información suficiente, responde con sinceridad. Sé claro, directo y profesional.

DOCUMENTO:
{context}

PREGUNTA:
{pregunta}
""")

# Función para formatear los documentos
def construir_contexto(documentos):
    return "\n\n".join([doc.page_content for doc in documentos])


# Embeddings y ruta de la base de datos
emb_model = "nomic-embed-text"
db_path = "./chroma_db"

# Pipeline de RAG actualizado
rag_pipeline = (
    {
        "context": lambda x: construir_contexto(x["documentos"]),
        "pregunta": lambda x: x["pregunta"]
    }
    | prompt_template
    | llm
    | StrOutputParser()
)

# ----------------- Sidebar -----------------
with st.sidebar:
    st.header("⚙ Cargue su PDF")
    st.write("Sube un PDF para indexar y hacer preguntas sobre su contenido.")
    uploaded_file = st.file_uploader("Selecciona un PDF para indexar", type="pdf")

    if uploaded_file:
        # Solo indexa si el archivo es nuevo (no igual al anterior)
        if ("uploaded_file_name" not in st.session_state) or (st.session_state.uploaded_file_name != uploaded_file.name):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(uploaded_file.read())
                temp_path = temp_file.name

            with st.spinner("Indexando el PDF..."):
                vectorstore = load_pdf(db_path=db_path, emb_model=emb_model, pdf_path=temp_path)
                st.session_state.vectorstore = vectorstore
                st.session_state.uploaded_file_name = uploaded_file.name
                st.success("PDF indexado exitosamente")


# Chatbot
if pregunta := st.chat_input("Haz una pregunta sobre el documento..."):
    st.session_state.chat_history.append({"role": "user", "content": pregunta})

    with st.chat_message("user", avatar=USER_ICON):
        st.markdown(pregunta)

    with st.chat_message("assistant", avatar=BOT_ICON):
        placeholder = st.empty()

        if st.session_state.vectorstore is None:
            placeholder.markdown("Por favor, suba y indexe un PDF primero.")
            respuesta = "No hay índice disponible."
        else:
            vectorstore = st.session_state.vectorstore

            # Busca los documentos similares
            documentos_similares = vectorstore.similarity_search(pregunta, k=4)
            try:
                respuesta = rag_pipeline.invoke({
                    "documentos": documentos_similares,
                    "pregunta": pregunta
                })
                placeholder.markdown(respuesta)
            except Exception as error:
                respuesta = f"❌ Error generando respuesta: {error}"
                placeholder.markdown(respuesta)

        st.session_state.chat_history.append({"role": "assistant", "content": respuesta})
