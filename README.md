# REQUISITOS
- Python 3.10 o superior
- Ollama instalado y configurado
- Modelos Ollama descargados (ej. `llama2:7b`, `llama3.1:8b`)( Probé con estos dos porque en mi computadora no tengo suficiente RAM o GPU)
- Recomiendo hacerlo con CPU, en app.py lo tengo configurado como llm = ChatOllama(model="llama3.1:8b", device="cpu"), si no se requiere hacerlo con CPU, ajustalo a GPU o elimina ese fragmento de código.
- Recomendado: al menos 16 GB de RAM (o usar modelos más livianos)


# Ejecución

1. Se debe abrir la carpeta
cd CHATBOT_JOHNCHACUA

2. Se activa el entorno virtual
.\.venv\Scripts\activate

3. Accede a src

cd src

4. Ya que estamos trabajando con el entorno de desarrollo con "uv", agregaremos las dependencias:

uv add streamlit langchain pypdf langchain_ollama langchain-community chromadb

5. Teniendo installado ollama, ejecutamos:

ollama pull -r requirements.txt

6. Inicia el servicio Ollama (si no está activo)

ollama run llama3.1:8b
O el modelo que vayas a usar.

7. Ejecuta la aplicación

streamlit run app.py