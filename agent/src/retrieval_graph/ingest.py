import json
import os
import uuid
import hashlib
from typing import List
import torch

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
#from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

DATA_PATH = "data/platon_analisis_nlp.json"
VECTORSTORE_PATH = "data/faiss"
CHUNKS_PATH = "data/platon_chunks.json"

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# -----------------------------
# EMBEDDINGS
# -----------------------------

def get_embeddings():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Usando embeddings en: {device}")

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )

# -----------------------------
# LOAD JSON
# -----------------------------
def load_analysis(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# -----------------------------
# SPLITTER
# -----------------------------
def build_splitter():
    return RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", "; ", " ", ""],
    )

# -----------------------------
# HELPERS
# -----------------------------
def stable_id(item: dict) -> str:
    base = f"{item.get('titulo','')}_{item.get('tipo','')}"
    return hashlib.sha1(base.encode("utf-8")).hexdigest()

def extract_concepts(item: dict) -> list[str]:
    return list({c["concepto"] for c in item.get("conceptos_filosoficos", [])})

def extract_entities(item: dict) -> list[str]:
    return list({
        ent[0]
        for ent in item.get("analisis_spacy", {})
                      .get("entidades_nombradas", [])
    })

# -----------------------------
# TO DOCUMENTS
# -----------------------------
def to_documents(data: list[dict]) -> List[Document]:
    splitter = build_splitter()
    documents: List[Document] = []

    for item in data:
        text = item.get("texto", "").strip()
        if not text:
            continue

        doc_id = stable_id(item)
        conceptos = extract_concepts(item)
        entidades = extract_entities(item)

        chunks = splitter.split_text(text)

        for i, chunk in enumerate(chunks):
            documents.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "doc_id": doc_id,
                        "chunk_id": i,
                        "tipo": item.get("tipo"),
                        "titulo": item.get("titulo"),
                        "conceptos": conceptos,
                        "entidades": entidades,
                        "source": "platon",
                    },
                )
            )

    return documents

# -----------------------------
# SAVE CHUNKS
# -----------------------------
def save_chunks(documents: list[Document]):
    os.makedirs("data", exist_ok=True)

    serializable = [
        {
            "text": doc.page_content,
            "metadata": doc.metadata,
        }
        for doc in documents
    ]

    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)

# -----------------------------
# VECTORSTORE
# -----------------------------
def build_vectorstore(documents: list[Document]):
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)

    os.makedirs(VECTORSTORE_PATH, exist_ok=True)
    vectorstore.save_local(VECTORSTORE_PATH)

# -----------------------------
# MAIN
# -----------------------------
def main():
    print("📖 Cargando análisis spaCy...")
    data = load_analysis(DATA_PATH)

    print("✂️  Creando chunks...")
    documents = to_documents(data)
    print(f"   → {len(documents)} chunks creados")

    print("💾 Guardando chunks...")
    save_chunks(documents)

    print("🧠 Construyendo FAISS...")
    build_vectorstore(documents)

    print("✅ Vectorstore creado en:", VECTORSTORE_PATH)

if __name__ == "__main__":
    main()
