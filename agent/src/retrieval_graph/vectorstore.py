#vectorstore.py
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

def load_vectorstore(path: str):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    if not Path(path).is_absolute():
        # Si la ruta es relativa, hacerla relativa al directorio del módulo
        base_path = Path(__file__).parent
        path = str(base_path / path)
    
    return FAISS.load_local(
        path,
        embeddings,
        allow_dangerous_deserialization=True,
    )