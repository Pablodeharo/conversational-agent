#vectorstore.py
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from pathlib import Path
import pickle

def load_vectorstore(path: str):
    path = Path(path).expanduser().resolve()

    # Si es pickle, cargar directamente
    if path.suffix == ".pkl":
        with open(path, "rb") as f:
            return pickle.load(f)
    
    # Si es carpeta o .faiss
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    return FAISS.load_local(
        str(path),
        embeddings,
        allow_dangerous_deserialization=True,
    )