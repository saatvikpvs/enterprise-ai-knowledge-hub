"""
app.py
======
This is the front door of the application — the Streamlit web interface
the user actually interacts with in their browser.

WHY STREAMLIT?
---------------
Streamlit lets us build a web UI using only Python (no HTML/CSS/JS
needed). Every time the user interacts with a widget (clicks a button,
types a question), Streamlit RE-RUNS this entire script top to bottom.
That's an important mental model: app.py is not a long-running server
loop — it's a script that gets replayed on every interaction. This is why
we use `st.session_state` (a persistent dictionary that survives across
re-runs) to remember things like chat history, instead of ordinary Python
variables (which would reset to empty on every re-run).

HOW THIS FILE CONNECTS TO EVERYTHING ELSE
--------------------------------------------
    User clicks "Process Documents"
        -> utils.save_uploaded_file()      (save bytes to disk)
        -> document_loader.load_single_document() (extract text)
        -> text_splitter.split_documents() (break into chunks)
        -> vector_store.add_chunks_to_store() (embed + save to ChromaDB)

    User types a question and hits Enter
        -> qa_chain.answer_question()
             -> vector_store.similarity_search() (retrieve relevant chunks)
             -> Gemini LLM (generate the answer using those chunks)
        -> app.py renders the answer + source citations in the chat UI
"""

import streamlit as st

from config import GOOGLE_API_KEY
from src.document_loader import load_single_document
from src.text_splitter import split_documents
from src.vector_store import (
    add_chunks_to_store,
    get_document_count,
    list_unique_sources,
    delete_document,
)
from src.qa_chain import answer_question
from src.utils import save_uploaded_file, is_supported_file, format_source_label


# ---------------------------------------------------------------------------
# PAGE SETUP
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Enterprise AI Knowledge Hub",
    page_icon="📚",
    layout="wide",
)

# ---------------------------------------------------------------------------
# SESSION STATE INITIALIZATION
# ---------------------------------------------------------------------------
# st.session_state is a dictionary-like object that Streamlit preserves
# across script re-runs FOR THE SAME BROWSER TAB / SESSION. We use it to
# store the chat history, since a normal Python list would be wiped out
# and recreated empty every time the script re-runs (which happens on
# every click).
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of {"role", "content", "sources"}

if "processing_log" not in st.session_state:
    st.session_state.processing_log = []


def render_sources(sources):
    """
    Render retrieved source chunks without showing the same citation text
    repeatedly when overlapping chunks return near-identical previews.
    """
    seen = set()
    for chunk in sources:
        label = format_source_label(chunk)
        preview = chunk.page_content[:300]
        dedupe_key = (label, " ".join(preview.split()).lower())

        if dedupe_key in seen:
            continue

        seen.add(dedupe_key)
        st.markdown(f"**{label}**")
        st.caption(preview + ("..." if len(chunk.page_content) > 300 else ""))


# ---------------------------------------------------------------------------
# SIDEBAR: DOCUMENT UPLOAD + MANAGEMENT
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("📁 Document Management")

    if not GOOGLE_API_KEY:
        st.error(
            "GOOGLE_API_KEY is not set. Copy `.env.example` to `.env` and "
            "add your Gemini API key before uploading documents or asking "
            "questions."
        )

    st.subheader("Upload documents")
    uploaded_files = st.file_uploader(
        "Supported formats: PDF, DOCX, TXT, MD, HTML",
        type=["pdf", "docx", "txt", "md", "html", "htm"],
        accept_multiple_files=True,
    )

    if st.button("Process Documents", disabled=not uploaded_files):
        with st.spinner("Reading, chunking, and embedding your documents..."):
            total_chunks_added = 0
            for uploaded_file in uploaded_files:
                if not is_supported_file(uploaded_file.name):
                    st.warning(f"Skipping unsupported file: {uploaded_file.name}")
                    continue

                # 1. Save the uploaded bytes to disk so our file-path-based
                #    loaders can read them.
                file_path = save_uploaded_file(uploaded_file)

                # 2. LOAD: extract raw text (+ metadata) from the file.
                raw_documents = load_single_document(file_path)

                # 3. SPLIT: break the raw text into small, retrieval-sized
                #    chunks.
                chunks = split_documents(raw_documents)

                # 4. EMBED + STORE: turn each chunk into a vector and save
                #    it into ChromaDB.
                added = add_chunks_to_store(chunks)
                total_chunks_added += added

                st.session_state.processing_log.append(
                    f"✅ {uploaded_file.name}: {added} chunks indexed"
                )

            st.success(
                f"Done! Indexed {total_chunks_added} new chunks. "
                f"You can now ask questions about these documents."
            )

    if st.session_state.processing_log:
        with st.expander("Processing log"):
            for line in st.session_state.processing_log:
                st.write(line)

    st.divider()

    st.subheader("📊 Knowledge base status")
    try:
        chunk_count = get_document_count()
        st.metric("Chunks indexed", chunk_count)
    except Exception:
        st.info("No documents indexed yet.")
        chunk_count = 0

    st.subheader("🗂️ Manage indexed documents")
    try:
        sources = list_unique_sources()
    except Exception:
        sources = []

    if sources:
        for source in sources:
            col1, col2 = st.columns([3, 1])
            col1.write(source)
            if col2.button("🗑️", key=f"delete_{source}"):
                delete_document(source)
                st.rerun()
    else:
        st.caption("No documents indexed yet.")

    st.divider()
    if st.button("Clear conversation"):
        st.session_state.chat_history = []
        st.rerun()


# ---------------------------------------------------------------------------
# MAIN PANEL: CHAT INTERFACE
# ---------------------------------------------------------------------------
st.title("📚 Enterprise AI Knowledge Hub")
st.caption(
    "Ask natural-language questions about your uploaded documents. "
    "Answers are generated using Retrieval-Augmented Generation (RAG) and "
    "include source citations."
)

# Render the existing conversation. st.chat_message() draws a chat-style
# bubble; we loop through everything stored in session_state so the full
# conversation re-appears after every Streamlit re-run.
for turn in st.session_state.chat_history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        # If this turn is an assistant answer that used retrieved sources,
        # show them in a collapsible section underneath the answer.
        if turn["role"] == "assistant" and turn.get("sources"):
            with st.expander("📎 Sources"):
                render_sources(turn["sources"])

# st.chat_input renders a text box pinned to the bottom of the page. It
# returns None on every re-run EXCEPT the one right after the user submits
# a message, in which case it returns the typed string.
question = st.chat_input("Ask a question about your documents...")

if question:
    if chunk_count == 0:
        st.warning("Please upload and process at least one document first.")
    elif not GOOGLE_API_KEY:
        st.warning("Please set GOOGLE_API_KEY in your .env file first.")
    else:
        # 1. Record and display the user's message immediately.
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        # 2. Run the full RAG pipeline: retrieve relevant chunks, then
        #    generate a grounded answer using the LLM.
        with st.chat_message("assistant"):
            with st.spinner("Searching documents and generating answer..."):
                result = answer_question(question, st.session_state.chat_history[:-1])
                st.markdown(result["answer"])

                if result["sources"]:
                    with st.expander("📎 Sources"):
                        render_sources(result["sources"])

        # 3. Save the assistant's turn (including sources) into history so
        #    it persists across re-runs and future follow-up questions.
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"],
        })
