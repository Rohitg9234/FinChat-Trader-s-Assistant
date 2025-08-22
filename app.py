# app.py
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st
from dotenv import load_dotenv

# ---- your modules ----
from llm import answer_with_tone
from embed import EmbeddingStore
from clean import clean_query
from video import list_channel_videos
from extract_sub import get_subtitle_whisper

# ------------- setup -------------
load_dotenv()
st.set_page_config(page_title="RAG Chat + Ingest", page_icon="💬")

@st.cache_resource(show_spinner=True)
def get_store():
    return EmbeddingStore()

store = get_store()

# ------------- session state -------------
if "view" not in st.session_state:
    st.session_state.view = "chat"   # "chat" or "ingest"
if "messages" not in st.session_state:
    st.session_state.messages = []
if "yt_videos" not in st.session_state:
    st.session_state.yt_videos = []

# ------------- shared header with view switch -------------
def header(title_left: str):
    top_l, top_r = st.columns([4, 1], vertical_alignment="top")
    with top_l:
        st.title(title_left)
    with top_r:
        if st.session_state.view == "chat":
            if st.button("➡️ Ingest", use_container_width=True):
                st.session_state.view = "ingest"
                st.rerun()
        else:
            if st.button("⬅️ Back to Chat", use_container_width=True):
                st.session_state.view = "chat"
                st.rerun()

# ------------- sidebar (applies to both views) -------------
st.sidebar.header("Settings")
tone = st.sidebar.text_input("Tone", value="concise, friendly")
temperature = st.sidebar.slider("Temperature", 0.0, 1.5, 0.6, 0.1)
max_tokens = st.sidebar.number_input("Max tokens", min_value=64, max_value=4096, value=1024, step=64)
top_k = st.sidebar.slider("Top-K chunks", 1, 10, 3, 1)
show_chunks = st.sidebar.checkbox("Show retrieved chunks (in Chat view)", value=True)
if st.sidebar.button("Clear chat history"):
    st.session_state.messages = []
    st.sidebar.success("Cleared!")

# ------------- chat view -------------
def render_chat():
    header("💬 RAG Chat")
    st.caption("Ask anything. I’ll retrieve top chunks and answer with your LLM wrapper.")

    # history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if show_chunks and msg["role"] == "assistant" and msg.get("chunks"):
                with st.expander("Retrieved chunks"):
                    for i, ch in enumerate(msg["chunks"], start=1):
                        text = ch.get("text", str(ch))
                        score = ch.get("score")
                        meta = []
                        if score is not None:
                            meta.append(f"**score:** {score:.4f}" if isinstance(score, (int, float)) else f"**score:** {score}")
                        if ch.get("id") is not None:
                            meta.append(f"**id:** {ch['id']}")
                        st.markdown(f"**Chunk {i}**  " + ("• " + " | ".join(meta) if meta else ""))
                        st.code(text, language="markdown")

    # input
    user_query = st.chat_input("Type a question")
    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                filtered_query = clean_query(user_query)
                chunks = store.query(filtered_query, top_k=int(top_k))
                reply = answer_with_tone(
                    query=user_query,
                    chunks=[r.get("text", str(r)) for r in chunks],
                    tone=tone,
                    temperature=float(temperature),
                    max_tokens=int(max_tokens),
                    stream=False,
                )
            st.markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply, "chunks": chunks})
        st.rerun()

# ------------- ingest view -------------
def render_ingest():
    header("▶️ YouTube Ingest")
    st.caption("Paste a channel / playlist / video URL. We'll fetch videos, transcribe with Whisper, and add to your EmbeddingStore.")

    with st.form("yt_ingest_form", clear_on_submit=False):
        url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/@channel ... or playlist/video URL")
        limit = st.number_input("Number of videos to ingest", min_value=1, max_value=200, value=5, step=1)
        col_a, col_b = st.columns(2)
        with col_a:
            model_name = st.selectbox("Whisper model", ["tiny", "base", "small", "medium", "large"], index=1)
        with col_b:
            language = st.text_input("Language (ISO 639-1)", value="en")
        show_embeds = st.checkbox("Preview embeds while ingesting", value=True)
        submitted = st.form_submit_button("Fetch & Ingest")

    if submitted:
        if not url.strip():
            st.warning("Please provide a valid YouTube URL.")
            return

        with st.status("Fetching video list...", expanded=True) as status:
            try:
                videos = list_channel_videos(url.strip(), limit=int(limit))
                st.write(f"Found **{len(videos)}** video(s).")
                st.session_state.yt_videos = videos

                if not videos:
                    status.update(label="No videos found.", state="error")
                    return

                status.update(label="Transcribing & indexing...", state="running")
                prog = st.progress(0)
                for i, v in enumerate(videos, start=1):
                    title = v.get("title", "(untitled)")
                    st.write(f"**{title}** — processing")
                    try:
                        transcript = get_subtitle_whisper(
                            v["url"],
                            model_name=model_name,
                            language=language,
                        )
                        # If your EmbeddingStore supports metadata, attach it:
                        # store.add_text(transcript, metadata={"source": "youtube", "title": title, "url": v["url"]})
                        store.add_text(transcript)
                        st.write(f"✅ Added: **{title}**")
                    except Exception as e:
                        st.write(f"⚠️ Skipped **{title}** — {e}")
                    prog.progress(i / len(videos))
                    if show_embeds:
                        st.video(v["url"])
                status.update(label="Done ingesting videos.", state="complete")
                st.success("Embeddings updated with YouTube transcripts.")
            except Exception as e:
                st.error(f"Failed: {e}")

    if st.session_state.yt_videos:
        with st.expander("Browse fetched videos"):
            titles = [v.get("title", "(untitled)") for v in st.session_state.yt_videos]
            idx = st.selectbox("Preview video", range(len(titles)), format_func=lambda i: titles[i])
            st.video(st.session_state.yt_videos[idx]["url"])

# ------------- router -------------
if st.session_state.view == "chat":
    render_chat()
else:
    render_ingest()
