import streamlit as st
import requests
import json
import os
import tempfile
import uuid
from datetime import datetime

try:
    import speech_recognition as sr
    VOICE_AVAILABLE = True
except ImportError:
    VOICE_AVAILABLE = False

try:
    from gtts import gTTS
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

API_URL = os.environ.get("API_URL", "http://localhost:8000")
TIMEOUT = 300


def stream_answer(question, model="flan-t5"):
    try:
        res = requests.post(f"{API_URL}/ask/stream", json={
            "question": question, "collection": "docs",
            "chat_history": [], "model": model,
        }, timeout=TIMEOUT, stream=True)
        full_answer = ""
        sources = []
        placeholder = st.empty()
        for line in res.iter_lines(decode_unicode=True):
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event["type"] == "chunk":
                full_answer += event["data"]
                placeholder.markdown(full_answer + "▌")
            elif event["type"] == "sources":
                sources = event["data"]
            elif event["type"] == "done":
                full_answer = event["data"]
                placeholder.markdown(full_answer)
        return full_answer, sources
    except requests.exceptions.Timeout:
        st.error("Timed out. Try again.")
        return "", []
    except Exception as e:
        st.error(f"Error: {e}")
        return "", []

st.set_page_config(page_title="DocQA Chatbot", page_icon=":books:", layout="wide")

st.markdown("""
<style>
.highlighted { background: #ffeb3b; padding: 1px 3px; border-radius: 2px; font-weight: bold; }
.source-card { background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 8px; padding: 12px; margin: 8px 0; }
div[data-testid="stExpander"] p { font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)

st.title("DocQA Chatbot v3.0")
st.caption("Upload documents, ask questions, compare docs, visualize similarity — all with voice")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "selected_model" not in st.session_state:
    st.session_state.selected_model = "flan-t5"
if "current_session" not in st.session_state:
    st.session_state.current_session = None
if "uploaded_files_list" not in st.session_state:
    st.session_state.uploaded_files_list = []

try:
    _stats = requests.get(f"{API_URL}/stats/docs", timeout=5).json()
    _chunk_count = _stats.get("total_chunks", 0)
    if _chunk_count == 0:
        st.warning("No documents uploaded yet. Upload a file to get started.")
    else:
        st.success(f"Ready — {_chunk_count} chunks indexed")
except Exception:
    pass

tab_chat, tab_docs, tab_chunks, tab_analytics, tab_scrape, tab_compare, tab_similarity, tab_sessions = st.tabs(
    ["Chat", "Documents", "Chunks", "Analytics", "URL Scraper", "Compare", "Similarity", "Sessions"]
)

# ── SIDEBAR: Model Selection ──────────────────────────────────
with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("OpenAI API Key (optional)", type="password", key="api_key_input")
    if api_key:
        try:
            requests.post(f"{API_URL}/set-key", json={"key": api_key}, timeout=5)
        except Exception:
            pass
    try:
        models_resp = requests.get(f"{API_URL}/models", timeout=5).json()
        model_options = {m["name"]: m["id"] for m in models_resp["models"]}
        selected = st.selectbox("AI Model", list(model_options.keys()), key="model_selector")
        st.session_state.selected_model = model_options[selected]
    except Exception:
        st.session_state.selected_model = "flan-t5"

    st.divider()
    if st.button("New Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.current_session = None
        st.rerun()

    if st.session_state.messages:
        chat_text = ""
        for msg in st.session_state.messages:
            role = "You" if msg["role"] == "user" else "AI"
            ts = msg.get("timestamp", "")
            chat_text += f"[{ts}] {role}: {msg['content']}\n\n"
        st.download_button("Export Chat (.txt)", chat_text,
            file_name=f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain", use_container_width=True)

        if TTS_AVAILABLE:
            ai_messages = [m["content"] for m in st.session_state.messages if m["role"] == "assistant"]
            if ai_messages:
                if st.button("Read Last Answer", use_container_width=True):
                    tts = gTTS(ai_messages[-1], lang="en")
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
                        tts.save(f.name)
                        st.audio(f.name, format="audio/mp3")

# ── TAB: Chat ──────────────────────────────────────────────────
with tab_chat:
    c1, c2 = st.columns([1, 2])

    with c1:
        st.subheader("Upload")
        uploaded_files = st.file_uploader("Choose files", type=["pdf", "txt", "docx", "csv", "md"], accept_multiple_files=True)
        if uploaded_files and st.button("Upload All", use_container_width=True):
            with st.spinner("Processing..."):
                try:
                    files = [("files", (f.name, f.getvalue())) for f in uploaded_files]
                    res = requests.post(f"{API_URL}/upload/batch", files=files, timeout=TIMEOUT)
                    if res.status_code == 200:
                        data = res.json()
                        for f in data["files"]:
                            if f["status"] == "ok":
                                st.success(f"{f['filename']}: {f['chunks']} chunks")
                            else:
                                st.warning(f"{f['filename']}: {f.get('reason', 'error')}")
                        st.info(f"Total chunks: {data['total_chunks']}")
                        try:
                            sug_res = requests.get(f"{API_URL}/suggest", timeout=30)
                            if sug_res.status_code == 200:
                                suggestions = sug_res.json().get("suggestions", [])
                                if suggestions:
                                    st.divider()
                                    st.subheader("Suggested Questions")
                                    for sq in suggestions:
                                        if st.button(sq, key=f"sug_{sq}", use_container_width=True):
                                            st.session_state["voice_query"] = sq
                                            st.rerun()
                        except Exception:
                            pass
                    else:
                        st.error("Upload failed")
                except Exception as e:
                    st.error(f"Error: {e}")

        st.divider()
        st.subheader("Voice Input")
        if VOICE_AVAILABLE:
            audio_file = st.audio_input("Record your question", key="voice_recorder")
            if audio_file:
                with st.spinner("Transcribing..."):
                    try:
                        recognizer = sr.Recognizer()
                        with sr.AudioFile(audio_file) as source:
                            audio_data = recognizer.record(source)
                        text = recognizer.recognize_google(audio_data)
                        if text.strip():
                            st.success(f"Transcribed: {text}")
                            st.session_state["voice_query"] = text
                    except sr.UnknownValueError:
                        st.error("Could not understand the audio.")
                    except sr.RequestError as e:
                        st.error(f"Speech service error: {e}")
                    except Exception as e:
                        st.error(f"Error: {e}")
        else:
            st.info("Voice input requires SpeechRecognition package.")

    voice_q = st.session_state.pop("voice_query", None)
    if voice_q:
        st.session_state.messages.append({"role": "user", "content": voice_q, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
        with st.chat_message("user"):
            st.write(voice_q)
            st.caption(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        with st.chat_message("assistant"):
            answer, sources = stream_answer(voice_q, st.session_state.selected_model)
            if answer:
                if sources:
                    with st.expander("Sources"):
                        for src in sources:
                            st.markdown(f"**[{src['metadata'].get('source', 'unknown')}]**")
                            st.markdown(src.get("highlighted", src["content"]), unsafe_allow_html=True)
                st.session_state.messages.append({
                    "role": "assistant", "content": answer,
                    "sources": sources,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                })

    with c2:
        st.subheader("Chat")
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
                if "timestamp" in msg:
                    st.caption(f"{msg['timestamp']} | {msg.get('model', st.session_state.selected_model)}")
                if "sources" in msg and msg["sources"]:
                    with st.expander("Sources"):
                        for src in msg["sources"]:
                            st.markdown(f"**[{src['metadata'].get('source', 'unknown')}]**")
                            highlighted = src.get("highlighted", src["content"])
                            st.markdown(highlighted, unsafe_allow_html=True)

        if not st.session_state.messages:
            try:
                sug_res = requests.get(f"{API_URL}/suggest", timeout=10)
                if sug_res.status_code == 200:
                    suggestions = sug_res.json().get("suggestions", [])
                    if suggestions:
                        st.info("Try asking:")
                        for sq in suggestions:
                            if st.button(sq, key=f"chat_sug_{sq}", use_container_width=True):
                                st.session_state["voice_query"] = sq
                                st.rerun()
            except Exception:
                pass

        question = st.chat_input("Ask something...")
        if question:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.session_state.messages.append({"role": "user", "content": question, "timestamp": timestamp})
            with st.chat_message("user"):
                st.write(question)
                st.caption(timestamp)

            with st.chat_message("assistant"):
                answer, sources = stream_answer(question, st.session_state.selected_model)
                if answer:
                    if sources:
                        with st.expander("Sources"):
                            for src in sources:
                                st.markdown(f"**[{src['metadata'].get('source', 'unknown')}]**")
                                highlighted = src.get("highlighted", src["content"])
                                st.markdown(highlighted, unsafe_allow_html=True)
                    st.session_state.messages.append({
                        "role": "assistant", "content": answer,
                        "sources": sources,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "model": st.session_state.selected_model,
                    })

        st.divider()
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

# ── TAB: Documents ─────────────────────────────────────────────
with tab_docs:
    st.subheader("Uploaded Documents")
    if st.button("Refresh", key="refresh_docs"):
        try:
            res = requests.get(f"{API_URL}/documents", timeout=10)
            if res.status_code == 200:
                st.session_state.uploaded_files_list = res.json()
        except Exception:
            pass
    docs = st.session_state.uploaded_files_list
    if docs:
        for doc in docs:
            col1, col2, col3 = st.columns([4, 1, 1])
            with col1:
                st.write(f"**{doc['name']}**")
            with col2:
                st.caption(f"{doc['chunks']} chunks")
            with col3:
                if st.button("Delete", key=f"del_{doc['name']}"):
                    try:
                        requests.post(f"{API_URL}/delete", json={"filename": doc["full_path"]}, timeout=10)
                        st.session_state.uploaded_files_list = []
                        st.rerun()
                    except Exception:
                        st.error("Delete failed")
    else:
        st.info("No documents uploaded.")

# ── TAB: Chunks ────────────────────────────────────────────────
with tab_chunks:
    st.subheader("Document Chunks")
    source_filter = st.text_input("Filter by source filename (optional)")
    limit = st.slider("Max chunks to show", 10, 200, 50)
    if st.button("Load Chunks"):
        with st.spinner("Loading..."):
            try:
                params = f"?limit={limit}"
                if source_filter:
                    params += f"&source={source_filter}"
                res = requests.get(f"{API_URL}/chunks{params}", timeout=30)
                if res.status_code == 200:
                    chunks = res.json()
                    st.info(f"Showing {len(chunks)} chunks")
                    for i, chunk in enumerate(chunks):
                        with st.expander(f"Chunk {i+1} | {chunk['source']} | Page {chunk['page']}"):
                            st.text(chunk["content"])
            except Exception as e:
                st.error(f"Error: {e}")

# ── TAB: Analytics ─────────────────────────────────────────────
with tab_analytics:
    st.subheader("Document Analytics")
    if st.button("Generate Analytics"):
        with st.spinner("Analyzing..."):
            try:
                res = requests.get(f"{API_URL}/analytics", timeout=30)
                if res.status_code == 200:
                    data = res.json()
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Words", f"{data['total_words']:,}")
                    col2.metric("Sentences", f"{data['total_sentences']:,}")
                    col3.metric("Paragraphs", f"{data['total_paragraphs']:,}")
                    col4.metric("Avg Word Length", data['avg_word_length'])
                    if data["top_words"]:
                        st.subheader("Top Words")
                        if PLOTLY_AVAILABLE:
                            fig = go.Figure(data=[go.Bar(x=[w["word"] for w in data["top_words"]],
                                                        y=[w["count"] for w in data["top_words"]])])
                            fig.update_layout(xaxis_title="Word", yaxis_title="Count")
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.bar_chart([{"Word": w["word"], "Count": w["count"]} for w in data["top_words"]], x="Word", y="Count")
            except Exception as e:
                st.error(f"Error: {e}")

# ── TAB: URL Scraper ──────────────────────────────────────────
with tab_scrape:
    st.subheader("URL Scraper & Analyzer")
    url = st.text_input("Enter URL to scrape")
    if url and st.button("Scrape & Analyze"):
        with st.spinner("Scraping..."):
            try:
                res = requests.post(f"{API_URL}/scrape", json={"url": url}, timeout=30)
                if res.status_code == 200:
                    data = res.json()
                    st.success(f"Scraped: {data['title']} ({data['text_length']:,} chars)")
                    with st.expander("Page Content"):
                        st.text(data["text"][:5000])
                    if data["links"]:
                        with st.expander(f"Links ({len(data['links'])})"):
                            for link in data["links"]:
                                st.markdown(f"[{link['text'][:60]}]({link['href']})")
                    st.divider()
                    if st.button("Summarize Scraped Content"):
                        with st.spinner("Summarizing..."):
                            try:
                                res2 = requests.post(f"{API_URL}/summarize", json={"text": data["text"][:4000]}, timeout=TIMEOUT)
                                if res2.status_code == 200:
                                    st.write(res2.json()["summary"])
                            except Exception as e:
                                st.error(f"Summarize error: {e}")
            except Exception as e:
                st.error(f"Error: {e}")

# ── TAB: Compare ──────────────────────────────────────────────
with tab_compare:
    st.subheader("Document Comparison")
    try:
        docs_res = requests.get(f"{API_URL}/documents", timeout=10).json()
        doc_names = [d["name"] for d in docs_res]
    except Exception:
        doc_names = []

    if len(doc_names) >= 2:
        col1, col2 = st.columns(2)
        with col1:
            source1 = st.selectbox("Document 1", doc_names, key="cmp1")
        with col2:
            source2 = st.selectbox("Document 2", doc_names, key="cmp2")

        if source1 != source2 and st.button("Compare Documents", use_container_width=True):
            with st.spinner("Comparing..."):
                try:
                    res = requests.post(f"{API_URL}/compare", json={
                        "source1": source1, "source2": source2
                    }, timeout=TIMEOUT)
                    if res.status_code == 200:
                        data = res.json()
                        st.write(data["comparison"])
                        st.divider()
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric(f"{source1} Words", f"{data['doc1_words']:,}")
                        col2.metric(f"{source2} Words", f"{data['doc2_words']:,}")
                        col3.metric("Common Words", f"{data['common_words']:,}")
                        col4.metric("Unique Words", f"{data['unique_to_doc1'] + data['unique_to_doc2']:,}")
                        if data["common_sample"]:
                            with st.expander("Common Terms Sample"):
                                st.write(", ".join(data["common_sample"]))
                except Exception as e:
                    st.error(f"Error: {e}")
    else:
        st.info("Upload at least 2 documents to compare.")

# ── TAB: Similarity ───────────────────────────────────────────
with tab_similarity:
    st.subheader("Vector Similarity Visualization")
    sim_question = st.text_input("Enter a question to visualize similarity", key="sim_q")
    if sim_question and st.button("Visualize Similarity", use_container_width=True):
        with st.spinner("Computing similarity scores..."):
            try:
                res = requests.get(f"{API_URL}/similarity", params={"question": sim_question}, timeout=30)
                if res.status_code == 200:
                    scores = res.json()
                    if scores:
                        if PLOTLY_AVAILABLE:
                            fig = go.Figure(data=[go.Bar(
                                x=[s["source"] for s in scores],
                                y=[s["score"] for s in scores],
                                text=[f'{s["score"]:.3f}' for s in scores],
                                textposition='auto',
                            )])
                            fig.update_layout(xaxis_title="Document Chunk", yaxis_title="Similarity Score", title=f"Similarity to: {sim_question}")
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.bar_chart([{"Source": s["source"], "Score": s["score"]} for s in scores], x="Source", y="Score")

                        st.subheader("Top Matching Chunks")
                        for i, s in enumerate(scores[:5]):
                            with st.expander(f"{i+1}. {s['source']} (score: {s['score']:.4f})"):
                                st.text(s["content_preview"])
                    else:
                        st.warning("No chunks found.")
            except Exception as e:
                st.error(f"Error: {e}")

# ── TAB: Sessions ─────────────────────────────────────────────
with tab_sessions:
    st.subheader("Chat Sessions")
    if st.button("Refresh Sessions"):
        st.rerun()

    try:
        sessions = requests.get(f"{API_URL}/sessions", timeout=10).json()
        if sessions:
            for session in sessions:
                with st.expander(f"{session['name']} — {session['message_count']} messages ({session['model']})"):
                    st.caption(f"Created: {session['created_at']}")
                    if st.button("Load", key=f"load_{session['id']}"):
                        history = requests.get(f"{API_URL}/sessions/{session['id']}/history", timeout=10).json()
                        st.session_state.messages = [{"role": h["role"], "content": h["content"], "timestamp": h.get("timestamp", "")} for h in history]
                        st.session_state.current_session = session["id"]
                        st.rerun()
                    if st.button("Delete", key=f"del_sess_{session['id']}"):
                        requests.delete(f"{API_URL}/sessions/{session['id']}", timeout=10)
                        st.rerun()
        else:
            st.info("No saved sessions yet.")
    except Exception as e:
        st.error(f"Error loading sessions: {e}")

    if st.session_state.messages:
        st.divider()
        session_name = st.text_input("Session name", value=f"Chat {datetime.now().strftime('%m/%d %H:%M')}")
        if st.button("Save Current Chat", use_container_width=True):
            try:
                sid = st.session_state.current_session or uuid.uuid4().hex
                if not st.session_state.current_session:
                    requests.post(f"{API_URL}/sessions", json={"name": session_name, "model": st.session_state.selected_model}, timeout=10)
                for msg in st.session_state.messages:
                    requests.post(f"{API_URL}/sessions/{sid}/save", json={
                        "role": msg["role"], "content": msg["content"],
                        "sources": msg.get("sources"), "model": st.session_state.selected_model,
                    }, timeout=10)
                st.session_state.current_session = sid
                st.success("Chat saved!")
            except Exception as e:
                st.error(f"Save failed: {e}")
