"""RAG agent loop over the vault."""

import re
from datetime import date
from pathlib import Path

from metis.client import get_client, get_chat_model
from metis.config import MetisConfig
from metis.search import search_vault, SearchResult

MAX_ROUNDS = 3
LOW_CONFIDENCE_THRESHOLD = 0.7


def reformulate_question(question: str, config: MetisConfig) -> str:
    """clean up the question: fix grammar, remove ambiguity, improve clarity."""
    client = get_client(config)
    model = get_chat_model(config)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "you are a query reformulator. take the user's messy question "
                    "and rewrite it as a clear, grammatically correct search query. "
                    "preserve the original intent. do not answer the question. "
                    "return ONLY the reformulated question, nothing else."
                ),
            },
            {"role": "user", "content": question},
        ],
        temperature=0.1,
    )

    return response.choices[0].message.content.strip()


def _build_context(results: list[SearchResult]) -> str:
    """format search results as context for the LLM."""
    if not results:
        return "no relevant content found in the vault."

    sections = []
    for r in results:
        source = r.file_path.split("/")[-1] if "/" in r.file_path else r.file_path
        sections.append(f"[source: {source} | relevance: {r.score}]\n{r.text}")

    return "\n\n---\n\n".join(sections)


def _avg_score(results: list[SearchResult]) -> float:
    """average similarity score of results."""
    if not results:
        return 0.0
    return sum(r.score for r in results) / len(results)


def ask(
    question: str,
    config: MetisConfig,
    note_path: str | None = None,
) -> tuple[str, list[str], float, str]:
    """ask a question against the vault (or a specific note).

    returns (answer, sources, avg_confidence, reformulated_query)
    """
    client = get_client(config)
    model = get_chat_model(config)

    # reformulate the question
    clean_question = reformulate_question(question, config)

    all_sources = []
    query = clean_question

    for round_num in range(MAX_ROUNDS):
        # retrieve — scoped to note if provided
        results = search_vault(query, config, limit=5, note_path=note_path)
        context = _build_context(results)
        confidence = _avg_score(results)

        # track sources
        for r in results:
            if r.file_path not in all_sources:
                all_sources.append(r.file_path)

        # build system prompt — quote-first when scoped to a note
        if note_path:
            system_prompt = (
                "you are metis, a knowledge assistant. answer the user's question "
                "using ONLY the provided context from their note. "
                "FIRST, quote the exact relevant passages from the context using > blockquotes. "
                "THEN, provide a concise answer based on those quotes. "
                "if the context doesn't contain enough information, say so honestly. "
                "do not invent or infer beyond what the text says.\n\n"
                f"context from note:\n\n{context}"
            )
        else:
            system_prompt = (
                "you are metis, a knowledge assistant. answer the user's question "
                "using ONLY the provided context from their personal vault. "
                "be direct and concise. cite which source you're drawing from. "
                "if the context doesn't contain enough information, say so honestly.\n\n"
                f"context from vault:\n\n{context}"
            )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": clean_question},
        ]

        if round_num > 0:
            messages.append({
                "role": "user",
                "content": (
                    f"(retrieval round {round_num + 1}. "
                    f"previous search wasn't sufficient. "
                    f"new search query: \"{query}\")"
                ),
            })

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.3,
        )

        answer = response.choices[0].message.content.strip()

        # evaluate — check if answer indicates insufficient context
        insufficient_signals = [
            "i don't have enough",
            "i couldn't find",
            "no relevant content",
            "not enough information",
            "doesn't contain",
            "does not contain",
            "not mentioned",
        ]
        answer_lower = answer.lower()
        needs_retry = any(signal in answer_lower for signal in insufficient_signals)

        if not needs_retry or round_num == MAX_ROUNDS - 1:
            break

        # reformulate for next round
        query = f"{clean_question} (alternative phrasing)"

    return answer, all_sources, confidence, clean_question


def format_qa_entry(question: str, answer: str) -> str:
    """format a Q&A entry for saving to a note."""
    today = date.today().isoformat()
    return f"\n**{question}** *(metis, {today})*\n{answer}\n"


def save_qa_to_note(note_path: str, question: str, answer: str) -> None:
    """insert Q&A into the note, before the Transcript/Content section."""
    path = Path(note_path)
    text = path.read_text(encoding="utf-8")
    entry = format_qa_entry(question, answer)

    # find where to insert — before ## Transcript or ## Content
    insert_patterns = [r"\n## Transcript\b", r"\n## Content\b"]
    insert_pos = None

    for pattern in insert_patterns:
        match = re.search(pattern, text)
        if match:
            insert_pos = match.start()
            break

    if "## Q&A" in text:
        # append to existing Q&A section
        qa_match = re.search(r"(## Q&A\n)", text)
        if qa_match:
            insert_after = qa_match.end()
            # find the end of existing Q&A content (next ## heading or the insert_pos)
            next_heading = re.search(r"\n## (?!Q&A)", text[insert_after:])
            if next_heading:
                qa_end = insert_after + next_heading.start()
            else:
                qa_end = len(text)
            text = text[:qa_end] + entry + text[qa_end:]
    elif insert_pos is not None:
        # create new Q&A section before Transcript/Content
        qa_section = f"\n## Q&A\n{entry}"
        text = text[:insert_pos] + qa_section + text[insert_pos:]
    else:
        # no Transcript/Content section — append at end
        text += f"\n## Q&A\n{entry}"

    path.write_text(text, encoding="utf-8")
