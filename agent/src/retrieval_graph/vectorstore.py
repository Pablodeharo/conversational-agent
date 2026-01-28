#vectorstore.py
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import pickle

def load_vectorstore(path: str):
    """
    Load a vector store from disk.

    This function supports two formats:
    1. A pickle file (.pkl) containing a pre-serialized vector store.
    2. A FAISS directory or .faiss file saved using LangChain's FAISS integration.

    Args:
        path (str): Path to the vector store file or directory.

    Returns:
        Any: A loaded vector store instance (typically FAISS).
    """
    path = Path(path).expanduser().resolve()

    
    if path.suffix == ".pkl":
        with open(path, "rb") as f:
            return pickle.load(f)
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    return FAISS.load_local(
        str(path),
        embeddings,
        allow_dangerous_deserialization=True,
    )