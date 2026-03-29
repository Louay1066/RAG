import html
import json
import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from rag import (
    clear_vectorstore,
    evaluate_single,
    get_collection_count,
    ingest_documents,
    query_rag,
)

load_dotenv()

BASE_DIR = Path(__file__).parent
CHROMA_DIR = str(BASE_DIR / "chroma_db")
DOCS_DIR = str(BASE_DIR / "docs")

FILE_ICONS = {".pdf": "PDF", ".txt": "TXT", ".md": "MD"}

# ── Page config & CSS ───────────────────────────────────────────────────────

st.set_page_config(page_title="RAG Pipeline", layout="wide")

st.markdown(
    """
<style>
    /* layout */
    .block-container { padding-top: 1.5rem; max-width: 1100px; }

    /* answer highlight */
    .answer-card {
        background: #1e293b;
        color: #e2e8f0;
        border-left: 4px solid #3b82f6;
        padding: 1.1rem 1.4rem;
        border-radius: 0 10px 10px 0;
        margin: 0.6rem 0 1rem 0;
        line-height: 1.7;
        font-size: 0.97rem;
    }

    /* score pills */
    .pill { display: inline-block; padding: 3px 12px; border-radius: 14px;
            font-weight: 600; font-size: 0.85rem; margin-right: 4px; }
    .pill-green  { background: #d1e7dd; color: #0a3622; }
    .pill-yellow { background: #fff3cd; color: #664d03; }
    .pill-red    { background: #f8d7da; color: #58151c; }

    /* file row */
    .file-row {
        display: flex; align-items: center; gap: 10px;
        padding: 0.55rem 0.8rem; border-radius: 8px;
        background: #fafafa; border: 1px solid #eee;
        margin-bottom: 6px; font-size: 0.9rem;
    }
    .file-row .badge {
        background: #e9ecef; color: #495057; padding: 2px 8px;
        border-radius: 4px; font-weight: 700; font-size: 0.75rem;
        font-family: monospace;
    }
    .file-row .fname { font-weight: 600; flex: 1; }
    .file-row .fsize { color: #888; font-size: 0.82rem; }

    /* stat card */
    .stat-card {
        text-align: center; padding: 1rem 0.5rem;
        background: #f8f9fa; border-radius: 10px;
        border: 1px solid #e9ecef;
    }
    .stat-card .num  { font-size: 2rem; font-weight: 700; color: #1a1a2e; }
    .stat-card .lab  { font-size: 0.82rem; color: #666; margin-top: 2px; }

    /* section divider */
    .section-gap { margin-top: 1.5rem; }

    /* hide streamlit chrome */
    #MainMenu, footer, header { visibility: hidden; }
</style>
""",
    unsafe_allow_html=True,
)

# ── Header ──────────────────────────────────────────────────────────────────

st.markdown("# RAG Pipeline")
st.caption(
    "Upload your documents, ask questions against them, "
    "and evaluate retrieval quality — all in one place."
)

# ── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Settings")
    api_key = st.text_input(
        "OpenAI API Key",
        value=os.getenv("OPENAI_API_KEY", ""),
        type="password",
        help="Stored in session only — never persisted.",
    )

    st.divider()
    with st.expander("Model & Retrieval", expanded=False):
        embedding_model = st.selectbox(
            "Embedding model",
            ["text-embedding-3-small", "text-embedding-3-large"],
        )
        llm_model = st.selectbox("LLM model", ["gpt-4o-mini", "gpt-4o"])
        top_k = st.slider("Top K chunks", 1, 10, 4)
        rerank = st.toggle("Cross-encoder reranking", value=True,
                           help="Re-rank bi-encoder results with a cross-encoder for higher relevance.")

    with st.expander("Chunking", expanded=False):
        chunk_size = st.slider("Chunk size (chars)", 200, 2000, 1000, step=100)
        chunk_overlap = st.slider("Chunk overlap (chars)", 0, 500, 200, step=50)

if not api_key:
    st.info("Paste your OpenAI API key in the sidebar to get started.")
    st.stop()

# ── Tabs ────────────────────────────────────────────────────────────────────

tab_docs, tab_query, tab_eval = st.tabs(["Documents", "Query", "Evaluate"])


# ── Helpers ─────────────────────────────────────────────────────────────────


def _score_pill(score: float) -> str:
    cls = "pill-green" if score >= 0.7 else ("pill-yellow" if score >= 0.4 else "pill-red")
    return f'<span class="pill {cls}">{score:.2f}</span>'


def _file_row_html(name: str, ext: str, size_kb: float) -> str:
    badge = FILE_ICONS.get(ext, ext.lstrip(".").upper())
    return (
        f'<div class="file-row">'
        f'  <span class="badge">{badge}</span>'
        f'  <span class="fname">{html.escape(name)}</span>'
        f'  <span class="fsize">{size_kb:.1f} KB</span>'
        f"</div>"
    )


def _stat_card(value, label: str) -> str:
    return (
        f'<div class="stat-card">'
        f'  <div class="num">{value}</div>'
        f'  <div class="lab">{label}</div>'
        f"</div>"
    )


# ═══════════════════════════════════════════════════════════════════════════
#  DOCUMENTS TAB
# ═══════════════════════════════════════════════════════════════════════════

with tab_docs:
    upload_col, spacer, status_col = st.columns([5, 0.3, 2])

    # ── Upload column ───────────────────────────────────────────────────
    with upload_col:
        st.subheader("Upload Documents")
        uploaded_files = st.file_uploader(
            "Drag and drop or browse — PDF, TXT, Markdown",
            accept_multiple_files=True,
            type=["pdf", "txt", "md"],
        )

        if uploaded_files:
            file_html = ""
            for f in uploaded_files:
                ext = Path(f.name).suffix.lower()
                file_html += _file_row_html(f.name, ext, len(f.getvalue()) / 1024)
            st.markdown(file_html, unsafe_allow_html=True)

            st.markdown("")  # spacing
            if st.button("Ingest into Vector Store", type="primary", use_container_width=True):
                os.makedirs(DOCS_DIR, exist_ok=True)
                file_paths = []
                for f in uploaded_files:
                    path = os.path.join(DOCS_DIR, f.name)
                    with open(path, "wb") as fp:
                        fp.write(f.getbuffer())
                    file_paths.append(path)

                with st.status("Ingesting documents...", expanded=True) as status:
                    st.write("Parsing files...")
                    try:
                        num_docs, num_chunks = ingest_documents(
                            file_paths=file_paths,
                            chroma_dir=CHROMA_DIR,
                            api_key=api_key,
                            embedding_model=embedding_model,
                            chunk_size=chunk_size,
                            chunk_overlap=chunk_overlap,
                        )
                        st.write(f"Split {num_docs} page(s) into {num_chunks} chunks")
                        st.write("Embedded and stored in ChromaDB")
                        status.update(label="Ingestion complete!", state="complete")
                    except Exception as e:
                        status.update(label="Ingestion failed", state="error")
                        st.error(str(e))
        else:
            st.markdown(
                '<div style="color:#999; padding:1rem 0; font-size:0.92rem;">'
                "Select files above to get started.</div>",
                unsafe_allow_html=True,
            )

    # ── Status column ───────────────────────────────────────────────────
    with status_col:
        st.subheader("Vector Store")

        chunk_count = get_collection_count(CHROMA_DIR)
        doc_files = sorted(Path(DOCS_DIR).glob("*")) if os.path.exists(DOCS_DIR) else []

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(_stat_card(chunk_count, "Chunks"), unsafe_allow_html=True)
        with c2:
            st.markdown(_stat_card(len(doc_files), "Files"), unsafe_allow_html=True)

        st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)

        if doc_files:
            with st.expander("Stored files", expanded=False):
                for f in doc_files:
                    ext = f.suffix.lower()
                    st.markdown(
                        _file_row_html(f.name, ext, f.stat().st_size / 1024),
                        unsafe_allow_html=True,
                    )

        if st.button("Clear Vector Store", use_container_width=True):
            clear_vectorstore(CHROMA_DIR)
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════
#  QUERY TAB
# ═══════════════════════════════════════════════════════════════════════════

with tab_query:
    st.subheader("Ask a Question")

    query_col, _ = st.columns([4, 1])
    with query_col:
        question = st.text_input(
            "Question",
            placeholder="e.g. What are the key findings in the report?",
            label_visibility="collapsed",
        )

    if st.button("Ask", disabled=not question, type="primary"):
        if not os.path.exists(CHROMA_DIR) or get_collection_count(CHROMA_DIR) == 0:
            st.warning("No documents ingested yet. Go to the Documents tab first.")
        else:
            with st.spinner("Retrieving and generating..."):
                try:
                    result = query_rag(
                        question=question,
                        chroma_dir=CHROMA_DIR,
                        api_key=api_key,
                        embedding_model=embedding_model,
                        llm_model=llm_model,
                        top_k=top_k,
                        rerank=rerank,
                    )

                    # ── Answer ──────────────────────────────────────────
                    st.markdown("**Answer**")
                    st.markdown(
                        f'<div class="answer-card">{html.escape(result["answer"])}</div>',
                        unsafe_allow_html=True,
                    )

                    # ── Retrieved contexts ──────────────────────────────
                    st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
                    st.markdown(f"**Retrieved Contexts** ({len(result['contexts'])} chunks)")

                    for i, (ctx, score, src) in enumerate(
                        zip(result["contexts"], result["scores"], result["sources"])
                    ):
                        score_html = _score_pill(score)
                        with st.expander(f"Chunk {i + 1}  —  relevance {score:.3f}"):
                            st.markdown(ctx)
                            src_str = src.get("source", str(src))
                            page = src.get("page")
                            loc = f"{Path(src_str).name}"
                            if page is not None:
                                loc += f", p.{page}"
                            st.caption(f"Source: {loc}")

                    # Save to history
                    if "query_history" not in st.session_state:
                        st.session_state.query_history = []
                    st.session_state.query_history.append({"question": question, **result})

                except Exception as e:
                    st.error(f"Query failed: {e}")


# ═══════════════════════════════════════════════════════════════════════════
#  EVALUATE TAB
# ═══════════════════════════════════════════════════════════════════════════

with tab_eval:
    st.subheader("Evaluate Pipeline")
    st.markdown(
        "Provide test questions (with optional ground-truth answers) to measure "
        "**faithfulness**, **answer relevancy**, **context relevancy**, and "
        "**correctness**."
    )

    eval_method = st.radio(
        "Input method", ["Manual entry", "Upload JSON"], horizontal=True
    )

    test_data: list[dict] = []

    if eval_method == "Manual entry":
        num_q = st.number_input("Number of test questions", 1, 20, 3)
        for i in range(num_q):
            with st.expander(f"Question {i + 1}", expanded=(i == 0)):
                q = st.text_input("Question", key=f"eq_{i}")
                gt = st.text_input("Ground truth (optional)", key=f"egt_{i}")
                if q:
                    test_data.append({"question": q, "ground_truth": gt or None})
    else:
        json_input = st.text_area(
            "Paste or edit JSON directly",
            value='[\n  {"question": "What is X?", "ground_truth": "X is..."},\n'
            '  {"question": "How does Y work?"}\n]',
            height=200,
        )
        json_file = st.file_uploader("Or upload a JSON file", type=["json"])

        raw = None
        if json_file:
            raw = json_file.read().decode("utf-8")
        elif json_input.strip():
            raw = json_input

        if raw:
            try:
                test_data = json.loads(raw)
            except json.JSONDecodeError:
                st.error("Invalid JSON. Check your syntax.")

    ready = len(test_data) > 0
    if st.button("Run Evaluation", disabled=not ready, type="primary"):
        if not os.path.exists(CHROMA_DIR) or get_collection_count(CHROMA_DIR) == 0:
            st.warning("No documents ingested yet.")
        else:
            questions = [d["question"] for d in test_data]
            ground_truths = [d.get("ground_truth") for d in test_data]

            progress = st.progress(0, text="Starting evaluation...")
            rag_results: list[dict] = []
            eval_results: list[dict] = []

            try:
                total = len(questions)
                for idx, (q, gt) in enumerate(zip(questions, ground_truths)):
                    progress.progress(
                        idx / total,
                        text=f"Evaluating question {idx + 1} / {total}...",
                    )

                    rag_r = query_rag(
                        question=q,
                        chroma_dir=CHROMA_DIR,
                        api_key=api_key,
                        embedding_model=embedding_model,
                        llm_model=llm_model,
                        top_k=top_k,
                        rerank=rerank,
                    )
                    rag_results.append(rag_r)

                    eval_r = evaluate_single(
                        question=q,
                        answer=rag_r["answer"],
                        contexts=rag_r["contexts"],
                        ground_truth=gt,
                        api_key=api_key,
                        llm_model=llm_model,
                    )
                    eval_results.append(eval_r)

                progress.progress(1.0, text="Evaluation complete!")

                # ── Build dataframe ─────────────────────────────────
                rows = []
                for q, rag_r, eval_r in zip(questions, rag_results, eval_results):
                    row = {
                        "Question": q,
                        "Answer": (
                            rag_r["answer"][:100] + "..."
                            if len(rag_r["answer"]) > 100
                            else rag_r["answer"]
                        ),
                        "Faithfulness": eval_r["faithfulness"]["score"],
                        "Relevancy": eval_r["answer_relevancy"]["score"],
                        "Context": eval_r["context_relevancy"]["score"],
                    }
                    if "correctness" in eval_r:
                        row["Correctness"] = eval_r["correctness"]["score"]
                    rows.append(row)

                df = pd.DataFrame(rows)

                # ── Summary cards ───────────────────────────────────
                st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
                st.markdown("### Summary")

                metric_map = {
                    "Faithfulness": "Faithfulness",
                    "Answer Relevancy": "Relevancy",
                    "Context Relevancy": "Context",
                }
                if "Correctness" in df.columns:
                    metric_map["Correctness"] = "Correctness"

                cols = st.columns(len(metric_map))
                for col, (display_name, col_name) in zip(cols, metric_map.items()):
                    avg = df[col_name].mean()
                    pill = _score_pill(avg)
                    col.markdown(
                        f'<div class="stat-card">'
                        f'<div style="margin-bottom:4px">{pill}</div>'
                        f'<div class="lab">{display_name}</div>'
                        f"</div>",
                        unsafe_allow_html=True,
                    )

                # ── Results table ───────────────────────────────────
                st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
                st.markdown("### Detailed Results")
                st.dataframe(
                    df.style.format(
                        {
                            c: "{:.2f}"
                            for c in ["Faithfulness", "Relevancy", "Context", "Correctness"]
                            if c in df.columns
                        }
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

                # ── Per-question breakdown ──────────────────────────
                st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
                st.markdown("### Per-Question Breakdown")

                for i, (q, rag_r, eval_r) in enumerate(
                    zip(questions, rag_results, eval_results)
                ):
                    with st.expander(f"Q{i + 1}: {q}"):
                        st.markdown(
                            f'<div class="answer-card">{html.escape(rag_r["answer"])}</div>',
                            unsafe_allow_html=True,
                        )
                        metric_html = ""
                        for name, data in eval_r.items():
                            label = name.replace("_", " ").title()
                            pill = _score_pill(data["score"])
                            reason = html.escape(data.get("reason", ""))
                            metric_html += (
                                f"<p>{pill} <strong>{label}</strong>"
                                f' <span style="color:#666">— {reason}</span></p>'
                            )
                        st.markdown(metric_html, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Evaluation failed: {e}")
