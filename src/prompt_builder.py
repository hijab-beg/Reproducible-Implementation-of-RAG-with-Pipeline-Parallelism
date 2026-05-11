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
You are a direct, knowledgeable assistant.
Use the Retrieved Context and conversation history to answer the user's request.

Rules you must follow:
- Be concise. Give only what is needed to fully answer the question — no padding, no preamble.
- Be confident. Never use hedging language: do not write "seems like", "probably", "it appears", "I think", "might be", "perhaps", or similar qualifiers. State facts directly.
- Never repeat yourself. Do not restate the same idea in different words, do not summarise what you just said, and do not add a closing sentence that echoes the opening.
- Always end on a complete sentence. Do not trail off or leave a thought unfinished.
- Do not invent details not supported by the retrieved context.
- For greetings, thanks, or farewells, respond in one short sentence.
- For follow-up prompts like "tell me more", continue from the most recent substantive topic in the conversation.
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