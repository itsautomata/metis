"""RAG agent loop over the vault."""

from metis.client import get_client, get_chat_model
from metis.config import MetisConfig
from metis.search import search_vault, SearchResult

MAX_ROUNDS = 3


def _build_context(results: list[SearchResult]) -> str:
    """format search results as context for the LLM."""
    if not results:
        return "no relevant content found in the vault."

    sections = []
    for r in results:
        source = r.file_path.split("/")[-1] if "/" in r.file_path else r.file_path
        sections.append(f"[source: {source} | relevance: {r.score}]\n{r.text}")

    return "\n\n---\n\n".join(sections)


def _needs_more_context(answer: str, question: str) -> str | None:
    """check if the answer suggests we need to search again.

    returns a reformulated query if yes, None if the answer is sufficient.
    """
    low_confidence_signals = [
        "i don't have enough",
        "i couldn't find",
        "no relevant content",
        "not enough information",
        "i'm not sure",
        "based on limited",
    ]
    answer_lower = answer.lower()
    for signal in low_confidence_signals:
        if signal in answer_lower:
            return None  # don't retry if LLM itself says it can't find enough
    return None


def ask(question: str, config: MetisConfig) -> tuple[str, list[str]]:
    """ask a question against the vault. returns (answer, sources).

    runs an agent loop: retrieve → answer → evaluate → maybe retrieve again.
    """
    client = get_client(config)
    model = get_chat_model(config)
    all_sources = []

    query = question

    for round_num in range(MAX_ROUNDS):
        # retrieve
        results = search_vault(query, config, limit=5)
        context = _build_context(results)

        # track sources
        for r in results:
            if r.file_path not in all_sources:
                all_sources.append(r.file_path)

        # generate answer
        messages = [
            {
                "role": "system",
                "content": (
                    "you are metis, a knowledge assistant. answer the user's question "
                    "using ONLY the provided context from their personal vault. "
                    "be direct and concise. cite which source you're drawing from. "
                    "if the context doesn't contain enough information, say so honestly.\n\n"
                    f"context from vault:\n\n{context}"
                ),
            },
            {"role": "user", "content": question},
        ]

        if round_num > 0:
            messages.append({
                "role": "user",
                "content": f"(this is retrieval round {round_num + 1}. the previous search wasn't sufficient. "
                           f"new search query was: \"{query}\")",
            })

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.3,
        )

        answer = response.choices[0].message.content.strip()

        # evaluate — do we need another round?
        reformulated = _needs_more_context(answer, question)
        if reformulated is None:
            break
        query = reformulated

    return answer, all_sources
