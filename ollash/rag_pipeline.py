import os
import shutil
import time
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOllama

# Global
retriever = None
vectorstore = None
_last_call_time = 0
MIN_CALL_INTERVAL = 0.5  # seconds


def get_embeddings_model():
    return OllamaEmbeddings(model="nomic-embed-text")


def build_faiss_index(pdf_path: str, store_path: str) -> str:
    global retriever, vectorstore

    if os.path.exists(store_path):
        shutil.rmtree(store_path, ignore_errors=True)

    print(f"📄 Loading and chunking {pdf_path}...")
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    ).split_documents(docs)

    print(f"🔎 Generating embeddings and building FAISS index...")
    vectorstore = FAISS.from_documents(chunks, get_embeddings_model())
    retriever = vectorstore.as_retriever()

    os.makedirs(store_path, exist_ok=True)
    vectorstore.save_local(store_path)

    return f"✅ Indexed {len(chunks)} chunks into {store_path}"


def load_vectorstore(store_path: str):
    global retriever, vectorstore
    if retriever is not None:
        return

    if os.path.exists(store_path):
        vectorstore = FAISS.load_local(
            store_path,
            get_embeddings_model(),
            allow_dangerous_deserialization=True
        )
        retriever = vectorstore.as_retriever()


def rate_limit():
    global _last_call_time
    now = time.time()
    wait_for = MIN_CALL_INTERVAL - (now - _last_call_time)
    if wait_for > 0:
        time.sleep(wait_for)
    _last_call_time = time.time()


def get_contextual_command(query: str, store_path: str, model_name="gemma:2b", os_label="Linux") -> str:
    """
    Retrieves context from FAISS and returns a formatted prompt to be passed to `ollama run`.
    """
    load_vectorstore(store_path)

    if retriever is None:
        return "❌ No retriever available. You must build the datastore first."

    def join_chunks(docs):
        return "\n\n".join(d.page_content for d in docs)

    relevant_docs = retriever.get_relevant_documents(query)
    context = join_chunks(relevant_docs)

    full_prompt = (
        f"Translate the following instruction into a safe {os_label} terminal command. "
        "Respond ONLY with the command dont write bash or zsh just one line for the command, no explanation:\n\n"
        f"Context:\n{context}\n\nInstruction: {query}"
    )

    ollama_cmd = [
        "ollama", "run", model_name,
        full_prompt
    ]

    return ollama_cmd
