def build_augmented_prompt(
    user_query: str,
    retrieved_chunks: list[str],
    partial_answer: str = "",
) -> str:
    context_text = "\n\n".join(
        [f"[Context {i+1}]\n{chunk}" for i, chunk in enumerate(retrieved_chunks)]
    )

    if not context_text.strip():
        context_text = "[No retrieved context]"

    instruction_block = """
You are a helpful, conversational assistant.
Use the Retrieved Context to answer the user's request.

Keep the response brief, focused, and complete.
Do not repeat the same sentence or idea.
For greetings, thanks, farewells, and similar social prompts, respond naturally and briefly.
For follow-up prompts such as 'tell me more about it', use the immediate prior conversation if available.
If the context is insufficient, give the best supported answer and, if needed, ask one short clarifying question.
If a chunk is incomplete, finish the thought naturally when the surrounding context supports it.
""".strip()

    if partial_answer.strip():
        return f"""
{instruction_block}

Question:
{user_query}

Retrieved Context:
{context_text}

Current Partial Answer:
{partial_answer}

Continue the answer:
""".strip()

    return f"""
{instruction_block}

Question:
{user_query}

Retrieved Context:
{context_text}

Answer:
""".strip()