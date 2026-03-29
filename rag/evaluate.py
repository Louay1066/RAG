import json

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# ── Evaluation prompts (LLM-as-judge) ──────────────────────────────────────

FAITHFULNESS_PROMPT = """\
Given the retrieved context and the generated answer, score whether the answer \
is grounded in (supported by) the context.

Retrieved Context:
{context}

Generated Answer:
{answer}

Score 0.0-1.0 (1 = fully supported, 0 = not supported).
Respond with ONLY JSON: {{"score": <float>, "reason": "<brief>"}}"""

ANSWER_RELEVANCY_PROMPT = """\
Given the question and answer, score whether the answer addresses the question.

Question:
{question}

Generated Answer:
{answer}

Score 0.0-1.0 (1 = fully addresses, 0 = irrelevant).
Respond with ONLY JSON: {{"score": <float>, "reason": "<brief>"}}"""

CONTEXT_RELEVANCY_PROMPT = """\
Given the question and retrieved contexts, score whether the contexts are \
relevant for answering the question.

Question:
{question}

Retrieved Contexts:
{context}

Score 0.0-1.0 (1 = all relevant, 0 = none relevant).
Respond with ONLY JSON: {{"score": <float>, "reason": "<brief>"}}"""

CORRECTNESS_PROMPT = """\
Given the question, generated answer, and ground-truth answer, score the \
correctness of the generated answer.

Question:
{question}

Generated Answer:
{answer}

Ground Truth:
{ground_truth}

Score 0.0-1.0 (1 = same key info, 0 = incorrect).
Respond with ONLY JSON: {{"score": <float>, "reason": "<brief>"}}"""


def _parse_score(text: str) -> dict:
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(text[start:end])
    except (json.JSONDecodeError, ValueError):
        pass
    return {"score": 0.0, "reason": "Failed to parse evaluation response"}


def evaluate_single(
    question: str,
    answer: str,
    contexts: list[str],
    ground_truth: str | None,
    api_key: str,
    llm_model: str = "gpt-4o-mini",
) -> dict:
    """Evaluate one RAG response on faithfulness, relevancy, and context quality."""
    llm = ChatOpenAI(model=llm_model, api_key=api_key, temperature=0)
    ctx = "\n\n---\n\n".join(contexts)
    metrics = {}

    for name, tmpl, variables in [
        ("faithfulness", FAITHFULNESS_PROMPT, {"context": ctx, "answer": answer}),
        ("answer_relevancy", ANSWER_RELEVANCY_PROMPT, {"question": question, "answer": answer}),
        ("context_relevancy", CONTEXT_RELEVANCY_PROMPT, {"question": question, "context": ctx}),
    ]:
        chain = ChatPromptTemplate.from_template(tmpl) | llm
        resp = chain.invoke(variables)
        metrics[name] = _parse_score(resp.content)

    if ground_truth:
        chain = ChatPromptTemplate.from_template(CORRECTNESS_PROMPT) | llm
        resp = chain.invoke({"question": question, "answer": answer, "ground_truth": ground_truth})
        metrics["correctness"] = _parse_score(resp.content)

    return metrics
