def build_augmented_prompt(user_query: str, retrieved_chunks: list[str], partial_answer: str = "") -> str:
    context_text = "\n\n".join(
        [f"[Context {i+1}]\n{chunk}" for i, chunk in enumerate(retrieved_chunks)]
    )

    if partial_answer.strip():
        return f"""
Use the retrieved context if relevant, but you can also rely on general knowledge when the context is incomplete.

Question:
{user_query}

Retrieved Context:
{context_text}

Current Partial Answer:
{partial_answer}

Continue the answer:
""".strip()

    return f"""
Use the retrieved context if relevant, but you can also rely on general knowledge when the context is incomplete.

Question:
{user_query}

Retrieved Context:
{context_text}

Answer:
""".strip()