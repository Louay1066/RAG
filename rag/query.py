from sentence_transformers import CrossEncoder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

RAG_PROMPT = """\
Answer the question based only on the following context.
If the context doesn't contain enough information, say so clearly.

Context:
{context}

Question: {question}"""

CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_cross_encoder_cache: CrossEncoder | None = None


def _get_cross_encoder() -> CrossEncoder:
    """Load the cross-encoder once and cache it."""
    global _cross_encoder_cache
    if _cross_encoder_cache is None:
        _cross_encoder_cache = CrossEncoder(CROSS_ENCODER_MODEL)
    return _cross_encoder_cache


def _rerank(question: str, docs, top_k: int):
    """Re-rank documents using a cross-encoder and return the top_k results."""
    cross_encoder = _get_cross_encoder()
    pairs = [[question, doc.page_content] for doc in docs]
    cross_scores = cross_encoder.predict(pairs)

    ranked = sorted(
        zip(docs, cross_scores),
        key=lambda x: x[1],
        reverse=True,
    )[:top_k]

    reranked_docs, reranked_scores = zip(*ranked)
    return list(reranked_docs), [float(s) for s in reranked_scores]


def query_rag(
    question: str,
    chroma_dir: str,
    api_key: str,
    embedding_model: str = "text-embedding-3-large",
    llm_model: str = "gpt-4o-mini",
    top_k: int = 4,
    rerank: bool = True,
    rerank_candidates_factor: int = 3,
) -> dict:
    """Retrieve relevant chunks and generate an answer.

    When rerank=True, the bi-encoder retrieves top_k * rerank_candidates_factor
    candidates, then a cross-encoder reranks them down to top_k.
    """
    embeddings = OpenAIEmbeddings(model=embedding_model, api_key=api_key)
    vectorstore = Chroma(persist_directory=chroma_dir, embedding_function=embeddings)

    # Fetch more candidates when reranking so the cross-encoder has a richer pool
    fetch_k = top_k * rerank_candidates_factor if rerank else top_k
    results = vectorstore.similarity_search_with_relevance_scores(question, k=fetch_k)

    if not results:
        return {"answer": "No relevant documents found.", "contexts": [], "sources": [], "scores": []}

    docs, bi_scores = zip(*results)

    if rerank:
        docs, scores = _rerank(question, docs, top_k)
    else:
        docs, scores = list(docs[:top_k]), [float(s) for s in bi_scores[:top_k]]

    context = "\n\n---\n\n".join(doc.page_content for doc in docs)

    llm = ChatOpenAI(model=llm_model, api_key=api_key, temperature=0)
    chain = ChatPromptTemplate.from_template(RAG_PROMPT) | llm
    response = chain.invoke({"context": context, "question": question})

    return {
        "answer": response.content,
        "contexts": [doc.page_content for doc in docs],
        "sources": [doc.metadata for doc in docs],
        "scores": scores,
    }
