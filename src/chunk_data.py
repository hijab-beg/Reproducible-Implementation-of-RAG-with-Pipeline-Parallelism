import json
from transformers import AutoTokenizer
from tqdm import tqdm

INPUT_PATH = "../data/raw_docs.json"
OUTPUT_PATH = "../data/chunks.json"

TOKENIZER_NAME = "gpt2"
CHUNK_SIZE = 64
STRIDE = 32 # Overlap of 32 tokens between chunks
# Target corpus size for paper-style larger-scale retrieval tests.
TARGET_CHUNKS = 600000


def load_docs(path: str):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_chunks(chunks, path: str):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(chunks, file, ensure_ascii=False, indent=2)


def chunk_document(tokenizer, doc_id: str, text: str, chunk_size: int, stride: int):
    token_ids = tokenizer.encode(
        text,
        add_special_tokens=False,
        truncation=True,
        max_length=4096
    )

    chunks = []
    chunk_index = 0

    for start in range(0, len(token_ids), stride):
        end = start + chunk_size
        chunk_token_ids = token_ids[start:end]

        if len(chunk_token_ids) < chunk_size:
            break

        chunk_text = tokenizer.decode(chunk_token_ids)

        chunks.append({
            "chunk_id": f"{doc_id}_chunk_{chunk_index}",
            "doc_id": doc_id,
            "start_token": start,
            "end_token": end,
            "text": chunk_text
        })

        chunk_index += 1

    return chunks


def _prepare_document_text(doc: dict) -> str:
    if doc.get("source") == "wikipedia":
        title = doc.get("title", "").strip()
        text = doc.get("text", "").strip()

        if title and text.startswith(title):
            return text
        if title:
            return f"{title}\n\n{text}"
        return text

    title = doc.get("title", "").strip()
    text = doc.get("text", "").strip()

    if title and text.startswith(title):
        return text
    if title:
        return f"{title}\n\n{text}"
    return text


def main():
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    docs = load_docs(INPUT_PATH)

    all_chunks = []

    for doc in tqdm(docs, desc="Chunking documents"):
        document_text = _prepare_document_text(doc)
        doc_chunks = chunk_document(
            tokenizer=tokenizer,
            doc_id=doc["doc_id"],
            text=document_text,
            chunk_size=CHUNK_SIZE,
            stride=STRIDE
        )
        all_chunks.extend(doc_chunks)

        if len(all_chunks) >= TARGET_CHUNKS:
            all_chunks = all_chunks[:TARGET_CHUNKS]
            break

    save_chunks(all_chunks, OUTPUT_PATH)
    print(f"Saved {len(all_chunks)} chunks to {OUTPUT_PATH} (target={TARGET_CHUNKS})")


if __name__ == "__main__":
    main()