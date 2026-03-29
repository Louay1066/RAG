from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

RAG_PROMPT = """\
Answer the question based only on the following context.
If the context doesn't contain enough information, say so clearly.

Context:
{context}

Question: {question}"""


def query_rag(
    question: str,
    chroma_dir: str,
    api_key: str,
    embedding_model: str = "text-embedding-3-small",
    llm_model: str = "gpt-4o-mini",
    top_k: int = 4,
) -> dict:
    """Retrieve relevant chunks and generate an answer."""
    embeddings = OpenAIEmbeddings(model=embedding_model, api_key=api_key)
    vectorstore = Chroma(persist_directory=chroma_dir, embedding_function=embeddings)

    results = vectorstore.similarity_search_with_relevance_scores(question, k=top_k)

    if not results:
        return {"answer": "No relevant documents found.", "contexts": [], "sources": [], "scores": []}

    docs, scores = zip(*results)
    context = "\n\n---\n\n".join(doc.page_content for doc in docs)

    llm = ChatOpenAI(model=llm_model, api_key=api_key, temperature=0)
    chain = ChatPromptTemplate.from_template(RAG_PROMPT) | llm
    response = chain.invoke({"context": context, "question": question})

    return {
        "answer": response.content,
        "contexts": [doc.page_content for doc in docs],
        "sources": [doc.metadata for doc in docs],
        "scores": [float(s) for s in scores],
    }
