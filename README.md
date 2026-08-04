# DocQA Chatbot v3.0

A full-stack **Document Q&A Chatbot** powered by RAG (Retrieval-Augmented Generation). Upload documents, ask questions via text or voice, compare documents, visualize similarity scores, and get AI-powered answers with highlighted source citations.

> **Built by:** [Dinesh Raj Upadhya](https://linkedin.com/in/dinesh-raj-upadhya-920075206)

## Live Demo

**[dineshupadhya-docqa-chatbot.hf.space](https://dineshupadhya-docqa-chatbot.hf.space)**

---

## Features

### Core
- **RAG Pipeline** — Retrieves relevant document chunks before answering, ensuring grounded responses
- **Multi-Model Support** — Choose between Flan-T5 (free, fast) or GPT-3.5 Turbo (best quality)
- **Batch Upload** — Upload multiple PDF, DOCX, TXT, CSV, MD files at once
- **Streaming Answers** — Watch answers appear word-by-word with cursor animation

### Voice
- **Voice Input** — Record audio questions via browser microphone (Google Speech Recognition)
- **Voice Output** — Listen to AI answers with text-to-speech (gTTS)
- **Query Suggestions** — Auto-generated clickable questions after document upload

### Analysis
- **Document Comparison** — Side-by-side AI comparison of two documents with word overlap stats
- **Vector Similarity Visualization** — Interactive Plotly bar chart showing chunk relevance scores
- **Highlighted Sources** — Source text highlights keywords from your question in bold
- **Document Analytics** — Word count, sentence count, top words with interactive charts
- **Chunk Viewer** — Browse and filter all indexed text chunks

### Utility
- **URL Scraper** — Scrape any webpage and ask questions about its content
- **Chat Persistence** — Save/load/delete conversations via SQLite sessions
- **Export Chat** — Download conversation history as a formatted text file
- **Auto-Summarize** — Generate one-click summaries of uploaded documents
- **OpenAI Integration** — Optional API key input to use GPT-3.5 Turbo

---

## How It Works

```
User uploads document
        │
        ▼
┌──────────────────┐
│  Document Loader  │  ← PDF / TXT / DOCX / CSV / MD
└────────┬─────────┘
         ▼
┌──────────────────┐
│  Text Splitter    │  ← RecursiveCharacterTextSplitter (500 chars, 50 overlap)
└────────┬─────────┘
         ▼
┌──────────────────┐
│  Embeddings       │  ← HuggingFace all-MiniLM-L6-v2 (384-dim vectors)
└────────┬─────────┘
         ▼
┌──────────────────┐
│  ChromaDB         │  ← Persistent vector store on disk
└────────┬─────────┘
         ▼
User asks a question
         │
         ▼
┌──────────────────┐
│  Retriever        │  ← Cosine similarity, top 5 chunks
└────────┬─────────┘
         ▼
┌──────────────────┐
│  LLM              │  ← Flan-T5-small or GPT-3.5 Turbo
└────────┬─────────┘
         ▼
    Answer with sources
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **Frontend** | Streamlit, Plotly, gTTS |
| **Backend** | FastAPI, Uvicorn |
| **RAG** | LangChain, ChromaDB |
| **Embeddings** | HuggingFace all-MiniLM-L6-v2 |
| **LLM** | Flan-T5-small (free) / GPT-3.5 Turbo |
| **Voice Input** | SpeechRecognition + Google Web Speech API |
| **Voice Output** | gTTS (Google Text-to-Speech) |
| **Database** | SQLite (chat sessions) |
| **Deployment** | HuggingFace Spaces (Docker) |
| **Reverse Proxy** | Nginx |
| **Process Manager** | Supervisord |

---

## Quick Start

### Option 1: Local Setup

```bash
# Clone the repo
git clone https://github.com/dineshrajupadhya/ZeroToDev.git
cd ZeroToDev/Artificial_Intelligence/doc-qa-chatbot

# Install dependencies
pip install -r requirements.txt

# Start backend (Terminal 1)
cd backend
python run.py

# Start frontend (Terminal 2)
cd frontend
streamlit run app.py
```

Open **http://localhost:8501**

### Option 2: Docker

```bash
docker build -t docqa-chatbot .
docker run -p 7860:7860 docqa-chatbot
```

Open **http://localhost:7860**

### Option 3: Use Hosted Version

No setup needed — visit **[dineshupadhya-docqa-chatbot.hf.space](https://dineshupadhya-docqa-chatbot.hf.space)**

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | Health check |
| `GET` | `/api/health/llm` | LLM loading status |
| `GET` | `/api/models` | List available AI models |
| `POST` | `/api/upload` | Upload a single document |
| `POST` | `/api/upload/batch` | Upload multiple documents |
| `POST` | `/api/ask` | Ask a question (standard) |
| `POST` | `/api/ask/stream` | Ask a question (SSE streaming) |
| `GET` | `/api/suggest` | Get suggested questions for documents |
| `POST` | `/api/compare` | Compare two documents |
| `GET` | `/api/similarity` | Get vector similarity scores |
| `POST` | `/api/summarize` | Summarize text |
| `POST` | `/api/scrape` | Scrape a URL |
| `GET` | `/api/documents` | List all documents |
| `GET` | `/api/chunks` | Browse text chunks |
| `GET` | `/api/analytics` | Document analytics |
| `GET` | `/api/stats/{collection}` | Collection statistics |
| `POST` | `/api/delete` | Delete a document |
| `GET` | `/api/sessions` | List chat sessions |
| `POST` | `/api/sessions` | Create a session |
| `GET` | `/api/sessions/{id}/history` | Get session history |
| `DELETE` | `/api/sessions/{id}` | Delete a session |

---

## Project Structure

```
doc-qa-chatbot/
├── backend/
│   ├── main.py              # FastAPI server — 20+ endpoints
│   ├── rag.py               # RAG pipeline — LLM, embeddings, vector store
│   ├── config.py            # Environment configuration
│   └── run.py               # Development runner
├── frontend/
│   └── app.py               # Streamlit UI — 8 tabs, voice, streaming
├── Dockerfile               # Multi-service container (nginx + FastAPI + Streamlit)
├── nginx.conf               # Reverse proxy — routes /api/* to FastAPI
├── supervisord.conf         # Process manager — runs nginx, uvicorn, streamlit
├── requirements.txt         # All Python dependencies
└── README.md                # This file
```

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | `""` | OpenAI API key for GPT-3.5 (optional) |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | HuggingFace embedding model |
| `CHUNK_SIZE` | `500` | Text chunk size in characters |
| `CHUNK_OVERLAP` | `50` | Overlap between chunks |
| `API_URL` | `http://localhost:8000` | Backend API URL |

---

## License

MIT License — free to use, modify, and distribute.
