import os
import uuid
import re
import json
from collections import Counter
from urllib.parse import urlparse

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse
from pydantic import BaseModel

from rag import (
    ingest_documents, ask_question, ask_question_stream, get_collection_stats,
    delete_documents_by_source, list_documents, SUPPORTED_EXTENSIONS,
    get_vectorstore, get_llm_status, compare_documents, get_similarity_scores,
    save_message, get_conversation_history, list_sessions, create_session, delete_session,
)
from config import UPLOAD_DIR

app = FastAPI(title="DocQA Chatbot API", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str
    collection: str = "docs"
    chat_history: list = []
    model: str = "flan-t5"


class CompareRequest(BaseModel):
    source1: str
    source2: str
    collection: str = "docs"


class SessionRequest(BaseModel):
    name: str = ""
    model: str = "flan-t5"


class DeleteRequest(BaseModel):
    filename: str
    collection: str = "docs"


class SetKeyRequest(BaseModel):
    key: str


@app.post("/set-key")
def set_api_key(req: SetKeyRequest):
    global OPENAI_API_KEY
    if req.key.startswith("sk-"):
        OPENAI_API_KEY = req.key
        os.environ["OPENAI_API_KEY"] = req.key
        return {"message": "OpenAI API key set", "status": "ok"}
    return {"message": "Invalid key format", "status": "error"}


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.get("/health/llm")
def health_llm():
    return get_llm_status()


@app.get("/supported-types")
def supported_types():
    return {"types": SUPPORTED_EXTENSIONS}


@app.get("/models")
def list_models():
    models = [
        {"id": "flan-t5", "name": "Flan-T5 Small", "description": "Fast, free, runs on CPU", "free": True},
    ]
    if OPENAI_API_KEY:
        models.append({"id": "openai", "name": "GPT-3.5 Turbo", "description": "Best quality, requires API key", "free": False})
    return {"models": models}


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")


@app.post("/upload")
async def upload_document(file: UploadFile = File(...), collection: str = "docs"):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"File type {ext} not supported")

    file_id = uuid.uuid4().hex
    file_path = os.path.join(UPLOAD_DIR, f"{file_id}{ext}")
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    try:
        chunks = ingest_documents(file_path, collection)
    finally:
        os.remove(file_path)

    return {"filename": file.filename, "chunks": chunks, "message": f"Ingested {chunks} chunks"}


@app.post("/upload/batch")
async def upload_batch(files: list[UploadFile] = File(...), collection: str = "docs"):
    results = []
    total_chunks = 0
    for file in files:
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            results.append({"filename": file.filename, "status": "skipped", "reason": f"Unsupported: {ext}"})
            continue
        file_id = uuid.uuid4().hex
        file_path = os.path.join(UPLOAD_DIR, f"{file_id}{ext}")
        with open(file_path, "wb") as f:
            f.write(await file.read())
        try:
            chunks = ingest_documents(file_path, collection)
            total_chunks += chunks
            results.append({"filename": file.filename, "status": "ok", "chunks": chunks})
        except Exception as e:
            results.append({"filename": file.filename, "status": "error", "reason": str(e)})
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)
    return {"files": results, "total_chunks": total_chunks}


@app.post("/ask")
def ask(req: AskRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    stats = get_collection_stats(req.collection)
    if stats["total_chunks"] == 0:
        return {"answer": "No documents uploaded yet. Please upload a document first.", "sources": []}
    try:
        history_text = ""
        if req.chat_history:
            for msg in req.chat_history[-6:]:
                role = "Human" if msg.get("role") == "user" else "Assistant"
                history_text += f"{role}: {msg.get('content', '')}\n"
        result = ask_question(req.question, req.collection, history=history_text, model_name=req.model)
        return {"answer": result["answer"], "sources": result["sources"]}
    except Exception as e:
        return {"answer": f"Error: {str(e)}", "sources": []}


@app.post("/ask/stream")
async def ask_stream(req: AskRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    stats = get_collection_stats(req.collection)
    if stats["total_chunks"] == 0:
        async def empty():
            yield json.dumps({"type": "done", "data": "No documents uploaded yet."}) + "\n"
        return StreamingResponse(empty(), media_type="text/event-stream")

    def generate():
        for chunk in ask_question_stream(req.question, req.collection, model_name=req.model):
            yield chunk
    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/suggest")
def suggest_questions(collection: str = "docs"):
    stats = get_collection_stats(collection)
    if stats["total_chunks"] == 0:
        return {"suggestions": []}
    try:
        vectorstore = get_vectorstore(collection)
        collection_obj = vectorstore._collection
        all_docs = collection_obj.get(limit=5)
        sample_text = "\n".join(all_docs["documents"][:3])[:1500]
        llm = get_llm("flan-t5")
        prompt = f"""Based on this text, generate 3 short questions a user might ask:

{sample_text}

Questions:
1."""
        result = llm.invoke(prompt)
        answer = result if isinstance(result, str) else result.get("text", str(result))
        questions = [q.strip().lstrip("0123456789. ") for q in answer.split("\n") if q.strip() and "?" in q]
        if len(questions) < 3:
            questions.extend(["What is this document about?", "Summarize the key points", "What are the main topics?"])
        return {"suggestions": questions[:3]}
    except Exception:
        return {"suggestions": ["What is this document about?", "Summarize the key points", "What are the main topics?"]}


@app.post("/compare")
def compare(req: CompareRequest):
    stats = get_collection_stats(req.collection)
    if stats["total_chunks"] == 0:
        raise HTTPException(status_code=400, detail="No documents uploaded")
    try:
        return compare_documents(req.source1, req.source2, req.collection)
    except Exception as e:
        return {"comparison": f"Error: {str(e)}", "doc1_words": 0, "doc2_words": 0, "common_words": 0, "unique_to_doc1": 0, "unique_to_doc2": 0, "common_sample": []}


@app.get("/similarity")
def similarity(question: str, collection: str = "docs"):
    if not question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    return get_similarity_scores(question, collection)


@app.get("/stats/{collection}")
def stats(collection: str = "docs"):
    return get_collection_stats(collection)


@app.get("/documents")
def get_documents(collection: str = "docs"):
    return list_documents(collection)


@app.post("/delete")
def delete_document(req: DeleteRequest):
    deleted = delete_documents_by_source(req.filename, req.collection)
    if deleted == 0:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"message": f"Deleted {deleted} chunks", "deleted": deleted}


@app.get("/chunks")
def get_chunks(collection: str = "docs", source: str = "", limit: int = 50):
    vectorstore = get_vectorstore(collection)
    collection_obj = vectorstore._collection
    if source:
        all_docs = collection_obj.get(where={"source": source})
    else:
        all_docs = collection_obj.get()
    chunks = []
    for i in range(min(limit, len(all_docs["ids"]))):
        chunks.append({
            "id": all_docs["ids"][i],
            "content": all_docs["documents"][i],
            "source": os.path.basename(all_docs["metadatas"][i].get("source", "unknown")),
            "page": all_docs["metadatas"][i].get("page", 0),
        })
    return chunks


@app.post("/summarize")
def summarize(req: dict):
    text = req.get("text", "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    try:
        llm = get_llm("flan-t5")
        prompt = f"""summarize: {text[:4000]} summary:"""
        result = llm.invoke(prompt)
        answer = result if isinstance(result, str) else result.get("text", str(result))
        return {"summary": answer}
    except Exception as e:
        return {"summary": f"Error: {str(e)}"}


@app.post("/scrape")
def scrape_url(req: dict):
    url = req.get("url", "").strip()
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    try:
        import requests as req_lib
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        response = req_lib.get(url, headers=headers, timeout=15)
        response.raise_for_status()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch URL: {str(e)}")
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()
    title = soup.title.string if soup.title else "No title"
    text = soup.get_text(separator="\n", strip=True)
    text = re.sub(r'\n{3,}', '\n\n', text)
    links = []
    for a in soup.find_all("a", href=True)[:20]:
        links.append({"text": a.get_text(strip=True)[:100], "href": a["href"]})
    return {"title": title, "text": text[:50000], "text_length": len(text), "links": links}


@app.get("/analytics")
def get_analytics(collection: str = "docs"):
    vectorstore = get_vectorstore(collection)
    collection_obj = vectorstore._collection
    all_docs = collection_obj.get()
    if not all_docs["documents"]:
        return {"total_words": 0, "total_sentences": 0, "total_paragraphs": 0, "avg_word_length": 0, "top_words": [], "word_count_by_source": {}}
    full_text = " ".join(all_docs["documents"])
    words = re.findall(r'\b[a-zA-Z]+\b', full_text.lower())
    word_count = len(words)
    avg_word_length = sum(len(w) for w in words) / max(len(words), 1)
    sentences = len(re.split(r'[.!?]+', full_text))
    paragraphs = len(re.split(r'\n\s*\n', full_text))
    stopwords = {"the", "a", "an", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did", "will", "would", "could", "should", "may", "might", "shall", "can", "to", "of", "in", "for", "on", "with", "at", "by", "from", "as", "into", "through", "during", "before", "after", "above", "below", "between", "and", "but", "or", "not", "no", "nor", "so", "yet", "both", "either", "neither", "each", "every", "all", "any", "few", "more", "most", "other", "some", "such", "than", "too", "very", "just", "about", "also", "this", "that", "these", "those", "i", "me", "my", "we", "our", "you", "your", "he", "him", "his", "she", "her", "it", "its", "they", "them", "their", "what", "which", "who", "whom", "when", "where", "why", "how", "if", "then", "else", "new", "one", "two", "three", "first", "second", "use", "using", "used", "etc", "com"}
    filtered = [w for w in words if w not in stopwords and len(w) > 2]
    top_words = Counter(filtered).most_common(20)
    source_counts = {}
    for doc in all_docs["documents"]:
        src = "unknown"
        source_counts[src] = source_counts.get(src, 0) + len(doc.split())
    return {"total_words": word_count, "total_sentences": sentences, "total_paragraphs": paragraphs, "avg_word_length": round(avg_word_length, 1), "top_words": [{"word": w, "count": c} for w, c in top_words], "word_count_by_source": source_counts}


@app.get("/sessions")
def get_sessions():
    return list_sessions()


@app.post("/sessions")
def create_new_session(req: SessionRequest):
    session_id = uuid.uuid4().hex
    create_session(session_id, req.name, req.model)
    return {"session_id": session_id, "name": req.name, "model": req.model}


@app.get("/sessions/{session_id}/history")
def get_session_history(session_id: str):
    return get_conversation_history(session_id)


@app.delete("/sessions/{session_id}")
def remove_session(session_id: str):
    delete_session(session_id)
    return {"message": "Session deleted"}


@app.post("/sessions/{session_id}/save")
def save_to_session(session_id: str, req: dict):
    save_message(session_id, req.get("role", "user"), req.get("content", ""), req.get("sources"), req.get("model", "flan-t5"))
    return {"message": "Saved"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
