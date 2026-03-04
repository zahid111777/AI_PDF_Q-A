"""
frontend/frontend.py
AI PDF Q&A — Streamlit UI
Hybrid (BM25 + Semantic) retrieval powered by LangChain & OpenAI
"""

import os
import sys
import tempfile

import streamlit as st
from dotenv import load_dotenv

# ── Path setup ─────────────────────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "backend", "src"))

from main import build_qa_pipeline  # noqa: E402

# ── Load environment ───────────────────────────────────────────────────────
load_dotenv(os.path.join(_ROOT, "keys", ".env"))

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI PDF Q&A",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
        .main-header {
            font-size: 2.4rem;
            font-weight: 800;
            background: linear-gradient(90deg, #1f77b4, #17becf);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .sub-text { color: #6c757d; font-size: 0.95rem; margin-bottom: 1rem; }
        .source-card {
            background: #f8f9ff;
            border-left: 4px solid #1f77b4;
            padding: 0.6rem 1rem;
            border-radius: 6px;
            margin: 6px 0;
            font-size: 0.85rem;
        }
        .badge {
            display: inline-block;
            padding: 2px 10px;
            border-radius: 12px;
            font-size: 0.72rem;
            font-weight: 600;
        }
        .badge-hybrid { background: #e8f4fd; color: #1565c0; }
        .retriever-info {
            background: #f0fff4;
            border: 1px solid #c3e6cb;
            border-radius: 8px;
            padding: 0.6rem 1rem;
            font-size: 0.82rem;
            color: #155724;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Session state init ─────────────────────────────────────────────────────
for key, default in [
    ("qa_engine", None),
    ("chat_history", []),       # list of dicts: {q, a, sources}
    ("pdf_processed", False),
    ("pdf_name", ""),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    model = st.selectbox(
        "LLM Model",
        ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
        index=0,
        help="Which OpenAI model to use for answering.",
    )

    st.markdown("---")
    st.markdown("### 🔍 Hybrid Retrieval")
    st.markdown(
        '<div class="retriever-info">'
        "Hybrid = <b>BM25</b> (keyword) + <b>Qdrant</b> (semantic).<br>"
        "Adjust weights to balance precision vs. recall."
        "</div>",
        unsafe_allow_html=True,
    )

    bm25_weight = st.slider(
        "BM25 (Keyword) Weight",
        min_value=0.0,
        max_value=1.0,
        value=0.4,
        step=0.05,
        help="Higher → more keyword matching.",
    )
    qdrant_weight = round(1.0 - bm25_weight, 2)
    st.caption(f"Qdrant (Semantic) Weight: **{qdrant_weight}**")

    k = st.slider("Top-K Chunks per Query", 2, 10, 4)

    st.markdown("---")
    st.markdown("### 📄 Chunking")
    chunk_size = st.slider("Chunk Size (chars)", 300, 2000, 1000, 50)
    chunk_overlap = st.slider("Chunk Overlap (chars)", 0, 500, 200, 25)

    st.markdown("---")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        if st.session_state.qa_engine:
            st.session_state.qa_engine.reset_memory()
        st.rerun()

    if st.session_state.pdf_processed:
        st.success(f"✅ **{st.session_state.pdf_name}**\nloaded & indexed")

# ── Main area ──────────────────────────────────────────────────────────────
st.markdown('<p class="main-header">📄 AI PDF Q&A</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-text">Upload a PDF, then ask questions. '
    'Uses <b>Hybrid Retrieval</b>: BM25 keyword search + Qdrant semantic search '
    '→ LangChain ConversationalRetrievalChain → OpenAI.</p>',
    unsafe_allow_html=True,
)

# ── PDF Upload section ─────────────────────────────────────────────────────
with st.container(border=True):
    st.markdown("#### 1️⃣  Upload your PDF")
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        label_visibility="collapsed",
    )

    if uploaded_file:
        col_info, col_btn = st.columns([4, 1])
        with col_info:
            st.info(
                f"📎  **{uploaded_file.name}**  •  "
                f"{uploaded_file.size / 1024:.1f} KB"
            )
        with col_btn:
            process_btn = st.button(
                "⚡ Process", type="primary", use_container_width=True
            )

        if process_btn:
            if not api_key:
                st.error("⚠️ Enter your OpenAI API key in the sidebar first.")
            else:
                with st.spinner(
                    "Extracting text → chunking → embedding → building hybrid index…"
                ):
                    try:
                        # Write to temp file
                        with tempfile.NamedTemporaryFile(
                            delete=False, suffix=".pdf"
                        ) as tmp:
                            tmp.write(uploaded_file.read())
                            tmp_path = tmp.name

                        engine = build_qa_pipeline(
                            pdf_path=tmp_path,
                            api_key=api_key,
                            model=model,
                            chunk_size=chunk_size,
                            chunk_overlap=chunk_overlap,
                            bm25_weight=bm25_weight,
                            qdrant_weight=qdrant_weight,
                            k=k,
                        )
                        os.unlink(tmp_path)

                        st.session_state.qa_engine = engine
                        st.session_state.pdf_processed = True
                        st.session_state.pdf_name = uploaded_file.name
                        st.session_state.chat_history = []
                        st.success(
                            f"✅ **{uploaded_file.name}** processed! "
                            "Scroll down to ask questions."
                        )
                        st.rerun()

                    except Exception as exc:
                        st.error(f"❌ Processing failed: {exc}")

# ── Helper ────────────────────────────────────────────────────────────────
def _render_sources(sources):
    """Render source document chunks in an expandable section."""
    if not sources:
        return
    with st.expander(f"📚 Source chunks used ({len(sources)})"):
        for doc in sources:
            page_num = doc.metadata.get("page", None)
            page_label = f"Page {page_num + 1}" if page_num is not None else "Unknown page"
            preview = doc.page_content[:350]
            if len(doc.page_content) > 350:
                preview += "…"
            st.markdown(
                f'<div class="source-card">'
                f'<span class="badge badge-hybrid">Hybrid</span> '
                f"&nbsp;<b>{page_label}</b><br><br>{preview}"
                f"</div>",
                unsafe_allow_html=True,
            )


# ── Chat section ───────────────────────────────────────────────────────────
if st.session_state.pdf_processed and st.session_state.qa_engine:
    st.markdown("---")
    st.markdown("#### 2️⃣  Ask questions")

    # ── Render existing chat history ───────────────────────────────────────
    for turn in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(turn["q"])
        with st.chat_message("assistant"):
            st.write(turn["a"])
            _render_sources(turn["sources"])

    # ── New question input ─────────────────────────────────────────────────
    question = st.chat_input(
        f"Ask something about {st.session_state.pdf_name}…"
    )
    if question:
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving context and generating answer…"):
                try:
                    answer, sources = st.session_state.qa_engine.ask(question)
                    st.write(answer)
                    _render_sources(sources)
                    st.session_state.chat_history.append(
                        {"q": question, "a": answer, "sources": sources}
                    )
                except Exception as exc:
                    st.error(f"❌ Error: {exc}")

elif not uploaded_file:
    st.markdown("---")
    st.info(
        "👆 Upload a PDF above to get started. "
        "Once processed, a chat box will appear here."
    )
