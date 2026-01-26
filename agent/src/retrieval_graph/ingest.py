import json
import os
import uuid
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

DATA_PATH = "data/platon_analisis_nlp.json"
VECTORSTORE_PATH = "data/faiss"

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" # Comprobar

def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True},
    )

def load_analysis(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
    
def build_splitter():
    return RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " ", ""],
    )

def to_documents(data: list[dict]) -> List[Document]:
    splitter = build_splitter()
    documents: List[Document] = []

    for item in data:
        text = item.get("texto", "").strip()
        if not text:
            continue

        chunks = splitter.split_text(text)

        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,
                metadata={
                    "doc_id": item.get("id", str(uuid.uuid4())),
                    "chunk_id": i,
                    "tipo": item.get("tipo"),
                    "titulo": item.get("titulo"),
                    "conceptos": list(item.get("conceptos", {}).keys()),
                    "entidades": [
                        e["text"] for e in item.get("entidades", [])
                    ],
                },
            )
            documents.append(doc)

    return documents

def build_vectorstore(documents: list[Document]):
    embeddings = get_embeddings()

    vectorstore = FAISS.from_documents(
        documents,
        embeddings,
    )

    os.makedirs(VECTORSTORE_PATH, exist_ok=True)
    vectorstore.save_local(VECTORSTORE_PATH)

def main():
    print("📖 Cargando análisis spaCy...")
    data = load_analysis(DATA_PATH)

    print("✂️  Creando documentos...")
    documents = to_documents(data)
    print(f"   → {len(documents)} chunks creados")

    print("🧠 Construyendo FAISS...")
    build_vectorstore(documents)

    print("✅ Vectorstore creado en:", VECTORSTORE_PATH)

if __name__ == "__main__":
    main()
