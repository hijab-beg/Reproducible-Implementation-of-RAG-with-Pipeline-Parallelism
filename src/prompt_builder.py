def build_augmented_prompt(user_query: str, retrieved_chunks: list[str], partial_answer: str = "") -> str:
    context_text = "\n\n".join(
        [f"[Context {i+1}]\n{chunk}" for i, chunk in enumerate(retrieved_chunks)]
    )

    if not context_text.strip():
        context_text = "[No retrieved context]"

    if partial_answer.strip():
        return f"""
You must answer using only the Retrieved Context.
Do not use external or general knowledge.
If the Retrieved Context does not contain enough information, answer exactly: "Insufficient context to answer."

Question:
{user_query}

Retrieved Context:
{context_text}

Current Partial Answer:
{partial_answer}

Continue the answer:
""".strip()

    return f"""
You must answer using only the Retrieved Context.
Do not use external or general knowledge.
If the Retrieved Context does not contain enough information, answer exactly: "Insufficient context to answer."

Question:
{user_query}

Retrieved Context:
{context_text}

Answer:
""".strip()