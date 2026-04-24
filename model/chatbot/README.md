
# 🐾 Pet Care RAG Chatbot Backend

An advanced, production-ready Retrieval-Augmented Generation (RAG) chatbot backend designed for veterinary and pet care assistance.

Built with **FastAPI**, this system leverages a strict **Plugin/Abstraction Architecture (Clean Architecture)**. This design decouples the core business logic from specific technologies, allowing seamless swapping between Local/Cloud LLMs (vLLM, OpenAI) and Vector Databases (ChromaDB) without altering the core RAG engine.

## ✨ Key Features

-   **Decoupled Architecture:** Core logic depends on abstract interfaces (`BaseLLM`, `BaseVectorDB`, `BaseMemory`), making the system incredibly flexible and future-proof.
    
-   **Real-time Streaming:** Utilizes Server-Sent Events (SSE) to stream AI responses token-by-token, providing a ChatGPT-like user experience.
    
-   **Context-Aware Memory:** Built-in sliding-window memory management to maintain conversational context across user sessions without exceeding token limits.
    
-   **High-Performance Vector Search:** Integrates local `ChromaDB` paired with lightweight embedding models (e.g., `vietnamese-sbert`) optimized for CPU execution.
    
-   **Production-Ready Dockerization:** Fully containerized setup with GPU support (`nvidia-container-toolkit`) orchestrating both the FastAPI backend and a dedicated vLLM engine via `docker-compose`.
    

## 🏗️ System Architecture

The project follows the **Ports and Adapters (Onion) pattern**:

1.  **API Layer (`api/`):** Handles HTTP requests, SSE streaming, and routing.
    
2.  **Core Logic (`services/rag_engine.py`):** Pure functions that assemble the optimal ChatML prompt combining retrieved contexts, history, and user queries.
    
3.  **Interfaces (`core/interfaces.py`):** The "Contracts" defining how the system interacts with external tools.
    
4.  **Plugins (`plugins/`):** Concrete implementations of the interfaces (e.g., `OpenAIPlugin`, `ChromaDBPlugin`, `LocalMemoryPlugin`).
    

### Codebase Structure

```
.
├── backend/
│   ├── api/                # API Routes (SSE Streaming endpoint)
│   ├── core/               # App configs and Abstract Interfaces
│   ├── plugins/            # Concrete adapters (OpenAI, Chroma, Memory)
│   ├── schemas/            # Pydantic models for I/O validation
│   ├── services/           # Pure RAG logic (rag_engine.py)
│   ├── main.py             # App entry point & Dependency Injection
│   └── requirements.txt    # Python dependencies
├── vectordb-processing/    # Scripts for parsing PDFs and building ChromaDB
├── docs/                   # Internal documentation and guides
├── backend.Dockerfile      # Docker image builder for the API
└── docker-compose.yaml     # Microservices orchestration (API + vLLM)

```

## 🚀 Getting Started (Local Development)

### 1. Prerequisites

-   Python 3.10+
    
-   (Optional but recommended) Virtual Environment (`venv` or `conda`)
    

### 2. Installation

Clone the repository and install the backend dependencies:

```
cd backend
pip install -r requirements.txt

```

### 3. Environment Configuration

Create a `.env` file in the `backend/` directory:

```
PROJECT_NAME="Pet Care RAG API"
CHROMA_DB_PATH="../vectordb-processing/chroma_db"
COLLECTION_NAME="pet_medical_docs"
EMBEDDING_MODEL="keepitreal/vietnamese-sbert"
MAX_HISTORY_TOKENS=1500

# Plugin Configuration
ACTIVE_LLM="openai" # Switch to "vllm" when deploying local models
OPENAI_API_KEY="sk-your-openai-api-key-here"

```

### 4. Run the Server

Start the FastAPI application using Uvicorn:

```
python main.py

```

_The server will start on `http://0.0.0.0:8080`._

## 🐳 Deployment (Docker)

To deploy the full decoupled architecture (FastAPI Backend + vLLM Engine) on a GPU-enabled VPS:

1.  Ensure Docker and `nvidia-container-toolkit` are installed on your host.
    
2.  Place your fine-tuned GGUF model in the root directory (e.g., `finetune-model.gguf`).
    
3.  Build and start the services:
    

```
docker-compose up -d --build

```

Check the logs to ensure both services are running smoothly:

```
docker logs -f pet_rag_backend
docker logs -f pet_vllm_engine

```

## 📡 API Usage

### `POST /api/chat`

This endpoint receives the user's message and streams back the AI's response using Server-Sent Events (SSE).

**Request Payload (JSON):**

```
{
  "session_id": "user_12345",
  "message": "Chó nhà tôi bị nôn mửa liên tục, phải làm sao?",
  "top_k": 3
}

```

**Testing with cURL:**

```
curl -N -X POST http://localhost:8080/api/chat \
     -H "Content-Type: application/json" \
     -d '{"session_id": "test_01", "message": "Chó nhà tôi bị nôn mửa liên tục, phải làm sao?"}'

```

**Expected Streamed Response:**

```
data: {"content": "Chào "}
data: {"content": "bạn, "}
data: {"content": "dựa "}
...
data: [DONE]

```

## 🧩 Extending the System

Thanks to the Plugin Architecture, extending this system is trivial and requires zero changes to the core `rag_engine.py` or API routes.

**How to add a new LLM Provider (e.g., Anthropic Claude):**

1.  Create `backend/plugins/llm_anthropic.py`.
    
2.  Implement a class inheriting from `BaseLLM` and define the `stream_chat` generator.
    
3.  Update `backend/main.py` in the `lifespan` function to inject your new plugin based on the `.env` config.
    

**How to switch to a distributed VectorDB (e.g., Pinecone):**

1.  Create `backend/plugins/vectordb_pinecone.py` inheriting from `BaseVectorDB`.
    
2.  Implement the `search` method returning the expected `List[Dict]` format.
    
3.  Inject it in `main.py`.
    

_Built with ❤️ for better Pet Care through Artificial Intelligence._
