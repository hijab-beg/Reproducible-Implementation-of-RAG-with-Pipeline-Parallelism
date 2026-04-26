def _format_history(conversation_history: list[tuple[str, str]] | None) -> str:
    if not conversation_history:
        return ""

    recent_turns = conversation_history[-6:]
    lines = []
    for role, text in recent_turns:
        cleaned_text = " ".join(text.split()).strip()
        if cleaned_text:
            lines.append(f"{role.title()}: {cleaned_text}")

    return "\n".join(lines)


def build_augmented_prompt(
    user_query: str,
    retrieved_chunks: list[str],
    partial_answer: str = "",
    conversation_history: list[tuple[str, str]] | None = None,
) -> str:
    context_text = "\n\n".join(
        [f"[Context {i+1}]\n{chunk}" for i, chunk in enumerate(retrieved_chunks)]
    )

    if not context_text.strip():
        context_text = "[No retrieved context]"

    history_text = _format_history(conversation_history)
    history_block = f"\nRecent Conversation:\n{history_text}\n" if history_text else ""

    instruction_block = """
You are a helpful, conversational assistant.
Use the Retrieved Context and the recent conversation history to answer the user's request.

For factual or technical questions, do not invent details that are not supported by the retrieved context.
For follow-up prompts such as 'tell me more about it', use the most recent substantive topic from the conversation history.
For greetings, thanks, farewells, and similar social prompts, respond naturally and briefly.

Write one coherent response.
Do not repeat the same sentence, clause, or paragraph.
Do not restate the same idea in different words.
Prefer a concise answer, but make sure it is complete and ends cleanly.
If the context is insufficient, give the best supported response you can and ask one short clarifying question if needed.
If a chunk is incomplete, finish the thought naturally when the surrounding context supports it.
""".strip()

    if partial_answer.strip():
        return f"""
{instruction_block}

Question:
{user_query}

{history_block}

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

{history_block}

Retrieved Context:
{context_text}

Answer:
""".strip()