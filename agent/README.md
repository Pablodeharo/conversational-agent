[![LangGraph](https://img.shields.io/badge/Built_with-LangGraph-00324d.svg)](https://github.com/langchain-ai/langgraph)
[![LangChain](https://img.shields.io/badge/Powered_by-LangChain-1c3c3c.svg)](https://github.com/langchain-ai/langchain)
[![Python](https://img.shields.io/badge/Python-Backend_AI-blue.svg)](https://www.python.org/)


<p align="center"> <img src="assets/Captura desde 2026-01-23 00-16-33.png" width="800"/> </p> 

🤖 Socratic Conversational Agent with Local LLM
A conversational AI agent that engages in Socratic dialogues, using semantic retrieval over Plato's complete works and powered by a local Mixtral model fine-tuned for Spanish.
✨ Features

🎭 Socratic Method: Engages users through questions rather than direct answers, guiding them to examine their own beliefs
📚 Semantic Retrieval: Uses sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 for multilingual document embeddings
🖥️ Local LLM Backend: Runs Mixtral model locally with custom ModelManager for efficient model loading and caching
⚡ FAISS Vector Store: Fast similarity search over Plato's complete works
🔄 LangGraph Workflow: Sophisticated multi-step reasoning with query generation, retrieval, reflection, and response
🌍 Multilingual Support: Optimized for Spanish with multilingual capabilities

🏗️ Architecture
Graph Flow
__start__
   ↓
generate_query  → Generates optimized search query
   ↓
retrieve        → Searches FAISS for relevant Plato texts
   ↓
reflect         → Analyzes question using Socratic method
   ↓
call_model      → Generates Socratic response with Mixtral
   ↓
__end__

Directory Structure

agent/
├── src/
│   └── retrieval_graph/
│       ├── Backend/              # Local model backends
│       │   ├── base.py          # Base backend interface
│       │   ├── llamacpp.py      # LlamaCpp backend
│       │   └── transformers.py  # HuggingFace Transformers backend
│       ├── config/
│       │   └── models.yaml      # Model configurations
│       ├── data/
│       │   ├── faiss/           # FAISS index files
│       │   │   ├── index.faiss
│       │   │   └── index.pkl
│       │   └── platon_analisis_nlp.json  # Source documents
│       ├── configuration.py     # Agent configuration
│       ├── graph.py            # Main conversation graph
│       ├── index_graph.py      # Document indexing graph
│       ├── ingest.py           # Document ingestion script
│       ├── model_manager.py    # Model loading and caching
│       ├── prompts.py          # System prompts
│       ├── retrieval.py        # Retrieval logic
│       ├── state.py            # State management
│       ├── tools.py            # Socratic reflection tools
│       ├── utils.py            # Utility functions
│       └── vectorstore.py      # FAISS wrapper
├── langgraph.json              # LangGraph configuration
├── pyproject.toml
├── requirements.txt
└── README.md

🚀 Getting Started
Prerequisites

Installation

Clone the repository

bashgit clone https://github.com/pablodeharo/conversational-agent.git
cd conversational-agent/agent

Create virtual environment

bashpython -m venv .venv
source .venv/bin/activate   # Linux/Mac
# or
.venv\Scripts\activate      # Windows

Install dependencies

hpip install -r requirements.txt

Configuration
1. Configure Models (config/models.yaml)
yamlmixtral:
  backend: llamacpp
  model_path: /path/to/mixtral_spanish_ft.Q4_0.gguf
  context_length: 8192
  n_gpu_layers: 35      # Set to 0 if no GPU
  n_threads: 8          # Adjust based on your CPU
  temperature: 0.7
  top_p: 0.9
  max_tokens: 512

2. Environment Variables (.env)

LANGSMITH_API_KEY=your_key_here

3. Update Configuration (configuration.py)
Ensure your paths are correct:
pythonresponse_model: str = "mixtral"
retriever_provider: str = "faiss-local"
index_path: str = "src/retrieval_graph/data/faiss"

# Document Ingestion

Ingest your documents into FAISS:
bash: python src/retrieval_graph/ingest.py --file data/platon_analisis_nlp.json

The ingestion script will:

Load documents from JSON
Generate embeddings using the multilingual model
Create/update FAISS index
Save to data/faiss/

# Running the Agent
Development Mode

bash: langgraph dev --allow-blocking
Access the UI at:

🎨 Studio UI: http://127.0.0.1:2024/studio
📚 API Docs: http://127.0.0.1:2024/docs




# 🔧 Technical Details

Embeddings

Uses multilingual semantic embeddings for cross-lingual retrieval:
pythonfrom langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
Model specs:

Dimension: 384
Languages: 50+ including Spanish, English, French, German
Performance: ~0.5s per query on CPU

Model Backend
Custom ModelManager with support for:

LlamaCpp: For GGUF quantized models (recommended)
Transformers: For HuggingFace models
Async generation
Model caching

Configurable parameters

pythonfrom retrieval_graph.model_manager import ModelManager

manager = ModelManager("config/models.yaml")
backend = await manager.get_backend("mixtral")

response = await backend.generate(
    prompt="Your prompt here",
    config=GenerationConfig(max_tokens=512, temperature=0.7)
)

🗺️ Roadmap

 Add voice input/output (Whisper + Bark/TTS)
 Support for multiple philosophical texts
 Fine-tune Mixtral specifically on Socratic dialogues
 Add conversation memory and context tracking
 Web interface for easier interaction
 Multi-turn conversation optimization


🙏 Acknowledgments

LangChain & LangGraph for the orchestration framework
Sentence Transformers for multilingual embeddings
FAISS for efficient vector search
Mixtral by Mistral AI
Plato's complete works for the knowledge base

Built with ❤️ using LangGraph and Local LLMs