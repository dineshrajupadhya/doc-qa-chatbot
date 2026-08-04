import os
import re
import json
import sqlite3
import hashlib
from datetime import datetime
from collections import Counter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks import CallbackManagerForLLMRun
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from config import (
    OPENAI_API_KEY, EMBEDDING_MODEL, CHUNK_SIZE,
    CHUNK_OVERLAP, VECTORSTORE_DIR
)

_llm = None
_llm_name = None
_embedding_model = None
_llm_error = None

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "chat_history.db")


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            role TEXT,
            content TEXT,
            sources TEXT,
            model TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            name TEXT,
            model TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    return conn


class FlanT5LLM(LLM):
    model_id: str = "google/flan-t5-small"
    model: object = None
    tokenizer: object = None

    @property
    def _llm_type(self) -> str:
        return "flan-t5"

    def _call(self, prompt: str, stop=None, run_manager=None, **kwargs) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        outputs = self.model.generate(**inputs, max_new_tokens=256)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    @classmethod
    def from_pretrained(cls, model_id="google/flan-t5-small"):
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        return cls(model_id=model_id, model=model, tokenizer=tokenizer)


def get_embeddings():
    global _embedding_model
    if _embedding_model is not None:
        return _embedding_model
    try:
        if OPENAI_API_KEY:
            from langchain_openai import OpenAIEmbeddings
            _embedding_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        else:
            _embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        return _embedding_model
    except Exception as e:
        raise RuntimeError(f"Failed to load embeddings: {e}")


def get_llm(model_name="flan-t5"):
    global _llm, _llm_name, _llm_error
    if _llm is not None and _llm_name == model_name:
        return _llm
    if _llm_error is not None and _llm_name == model_name:
        raise RuntimeError(f"LLM previously failed to load: {_llm_error}")
    try:
        if model_name == "openai" and OPENAI_API_KEY:
            from langchain_openai import ChatOpenAI
            _llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, openai_api_key=OPENAI_API_KEY)
        elif model_name == "flan-t5":
            _llm = FlanT5LLM.from_pretrained("google/flan-t5-small")
        else:
            _llm = FlanT5LLM.from_pretrained("google/flan-t5-small")
        _llm_name = model_name
        _llm_error = None
        return _llm
    except Exception as e:
        _llm_error = str(e)
        _llm_name = model_name
        raise RuntimeError(f"Failed to load LLM: {e}")


def get_llm_status():
    return {"loaded": _llm is not None, "model": _llm_name, "error": _llm_error}


SUPPORTED_EXTENSIONS = {
    ".pdf": "PDF Document",
    ".txt": "Text File",
    ".docx": "Word Document",
    ".csv": "CSV File",
    ".md": "Markdown File",
}


def load_document(file_path: str):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return PyPDFLoader(file_path).load()
    elif ext == ".txt":
        return TextLoader(file_path).load()
    elif ext == ".docx":
        import docx2txt
        text = docx2txt.process(file_path)
        from langchain.schema import Document
        return [Document(page_content=text, metadata={"source": os.path.basename(file_path)})]
    elif ext == ".csv":
        return CSVLoader(file_path).load()
    elif ext == ".md":
        return TextLoader(file_path).load()
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    return splitter.split_documents(docs)


def get_vectorstore(collection_name: str = "docs") -> Chroma:
    embeddings = get_embeddings()
    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=VECTORSTORE_DIR,
    )


def ingest_documents(file_path: str, collection_name: str = "docs") -> int:
    docs = load_document(file_path)
    chunks = split_documents(docs)
    vectorstore = get_vectorstore(collection_name)
    vectorstore.add_documents(chunks)
    return len(chunks)


def delete_documents_by_source(source_name: str, collection_name: str = "docs") -> int:
    vectorstore = get_vectorstore(collection_name)
    collection = vectorstore._collection
    results = collection.get(where={"source": source_name})
    if results["ids"]:
        collection.delete(ids=results["ids"])
    return len(results["ids"])


def list_documents(collection_name: str = "docs") -> list:
    vectorstore = get_vectorstore(collection_name)
    collection = vectorstore._collection
    all_docs = collection.get()
    sources = {}
    for i, meta in enumerate(all_docs["metadatas"]):
        src = meta.get("source", "unknown")
        if src not in sources:
            sources[src] = 0
        sources[src] += 1
    return [{"name": os.path.basename(k), "full_path": k, "chunks": v} for k, v in sources.items()]


def _highlight_text(text: str, question: str) -> str:
    question_words = [w.lower() for w in re.findall(r'\b[a-zA-Z]{3,}\b', question)]
    highlighted = text
    for word in question_words:
        pattern = re.compile(re.escape(word), re.IGNORECASE)
        highlighted = pattern.sub(f"**{word}**", highlighted)
    return highlighted


def ask_question(question: str, collection_name: str = "docs", history: str = "", model_name: str = "flan-t5") -> dict:
    vectorstore = get_vectorstore(collection_name)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    context_docs = retriever.invoke(question)
    context_text = "\n\n".join([d.page_content for d in context_docs])

    prompt = f"""context: {context_text} question: {question} answer:"""

    llm = get_llm(model_name)
    result = llm.invoke(prompt)

    sources = []
    for doc in context_docs:
        highlighted = _highlight_text(doc.page_content, question)
        sources.append({
            "content": doc.page_content[:500],
            "highlighted": highlighted[:500],
            "metadata": doc.metadata,
        })

    answer = result if isinstance(result, str) else result.get("text", str(result))

    return {"answer": answer, "sources": sources}


def ask_question_stream(question: str, collection_name: str = "docs", history: str = "", model_name: str = "flan-t5"):
    vectorstore = get_vectorstore(collection_name)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    context_docs = retriever.invoke(question)
    context_text = "\n\n".join([d.page_content for d in context_docs])

    prompt = f"""context: {context_text} question: {question} answer:"""

    sources = []
    for doc in context_docs:
        highlighted = _highlight_text(doc.page_content, question)
        sources.append({
            "content": doc.page_content[:500],
            "highlighted": highlighted[:500],
            "metadata": doc.metadata,
        })

    yield json.dumps({"type": "sources", "data": sources}) + "\n"

    llm = get_llm(model_name)
    result = llm.invoke(prompt)
    answer = result if isinstance(result, str) else result.get("text", str(result))

    words = answer.split()
    for i in range(0, len(words), 3):
        chunk = " ".join(words[i:i+3])
        yield json.dumps({"type": "chunk", "data": chunk + " "}) + "\n"

    yield json.dumps({"type": "done", "data": answer}) + "\n"


def compare_documents(source1: str, source2: str, collection_name: str = "docs") -> dict:
    vectorstore = get_vectorstore(collection_name)
    collection = vectorstore._collection

    docs1 = collection.get(where={"source": source1})
    docs2 = collection.get(where={"source": source2})

    text1 = "\n".join(docs1["documents"]) if docs1["documents"] else ""
    text2 = "\n".join(docs2["documents"]) if docs2["documents"] else ""

    words1 = set(re.findall(r'\b[a-zA-Z]{3,}\b', text1.lower()))
    words2 = set(re.findall(r'\b[a-zA-Z]{3,}\b', text2.lower()))

    common = words1 & words2
    only1 = words1 - words2
    only2 = words2 - words1

    prompt = f"""Compare these two documents briefly.

Document 1 ({source1}):
{text1[:2000]}

Document 2 ({source2}):
{text2[:2000]}

Comparison:"""

    llm = get_llm("flan-t5")
    result = llm.invoke(prompt)
    comparison = result if isinstance(result, str) else result.get("text", str(result))

    return {
        "comparison": comparison,
        "doc1_words": len(text1.split()),
        "doc2_words": len(text2.split()),
        "common_words": len(common),
        "unique_to_doc1": len(only1),
        "unique_to_doc2": len(only2),
        "common_sample": list(common)[:20],
    }


def get_similarity_scores(question: str, collection_name: str = "docs") -> list:
    vectorstore = get_vectorstore(collection_name)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    docs = retriever.invoke(question)

    query_embedding = get_embeddings().embed_query(question)

    results = []
    for doc in docs:
        doc_embedding = get_embeddings().embed_documents([doc.page_content])[0]
        score = sum(a * b for a, b in zip(query_embedding, doc_embedding)) / (
            (sum(a**2 for a in query_embedding) ** 0.5) *
            (sum(b**2 for b in doc_embedding) ** 0.5)
        )
        results.append({
            "source": os.path.basename(doc.metadata.get("source", "unknown")),
            "content_preview": doc.page_content[:100],
            "score": round(score, 4),
        })

    return sorted(results, key=lambda x: x["score"], reverse=True)


def get_collection_stats(collection_name: str = "docs") -> dict:
    vectorstore = get_vectorstore(collection_name)
    collection = vectorstore._collection
    return {"total_chunks": collection.count(), "collection_name": collection_name}


def save_message(session_id: str, role: str, content: str, sources: list = None, model: str = "flan-t5"):
    conn = get_db()
    conn.execute(
        "INSERT INTO conversations (session_id, role, content, sources, model) VALUES (?, ?, ?, ?, ?)",
        (session_id, role, content, json.dumps(sources) if sources else None, model)
    )
    conn.commit()
    conn.close()


def get_conversation_history(session_id: str) -> list:
    conn = get_db()
    rows = conn.execute(
        "SELECT role, content, sources, model, timestamp FROM conversations WHERE session_id = ? ORDER BY id",
        (session_id,)
    ).fetchall()
    conn.close()
    result = []
    for row in rows:
        item = {"role": row[0], "content": row[1], "timestamp": row[4]}
        if row[2]:
            item["sources"] = json.loads(row[2])
        if row[3]:
            item["model"] = row[3]
        result.append(item)
    return result


def list_sessions() -> list:
    conn = get_db()
    rows = conn.execute(
        """SELECT s.id, s.name, s.model, s.created_at, COUNT(c.id) as message_count
           FROM sessions s LEFT JOIN conversations c ON s.id = c.session_id
           GROUP BY s.id ORDER BY s.created_at DESC"""
    ).fetchall()
    conn.close()
    return [{"id": r[0], "name": r[1], "model": r[2], "created_at": r[3], "message_count": r[4]} for r in rows]


def create_session(session_id: str, name: str = "", model: str = "flan-t5"):
    conn = get_db()
    conn.execute(
        "INSERT OR REPLACE INTO sessions (id, name, model) VALUES (?, ?, ?)",
        (session_id, name or f"Chat {datetime.now().strftime('%m/%d %H:%M')}", model)
    )
    conn.commit()
    conn.close()


def delete_session(session_id: str):
    conn = get_db()
    conn.execute("DELETE FROM conversations WHERE session_id = ?", (session_id,))
    conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
    conn.commit()
    conn.close()
