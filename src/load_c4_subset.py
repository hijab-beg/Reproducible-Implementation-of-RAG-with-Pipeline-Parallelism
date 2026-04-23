import json
import os
from datasets import load_dataset
from tqdm import tqdm

OUTPUT_PATH = "../data/raw_docs.json"

# Increased to support larger chunk corpora (e.g., ~600k chunks after chunking).
TARGET_DOCS = 30000
MIN_CHAR_LENGTH = 1200


def load_c4_subset(target_docs: int, min_char_length: int):
    dataset = load_dataset(
        "c4",
        "en",
        split="train",
        streaming=True,
        trust_remote_code=True
    )

    docs = []

    for sample in tqdm(dataset, desc="Loading C4 samples"):
        text = sample.get("text", "").strip()

        if len(text) < min_char_length:
            continue

        docs.append({
            "doc_id": f"doc_{len(docs)}",
            "text": text
        })

        if len(docs) >= target_docs:
            break

    return docs


def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    docs = load_c4_subset(TARGET_DOCS, MIN_CHAR_LENGTH)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
        json.dump(docs, file, ensure_ascii=False, indent=2)

    print(f"Saved {len(docs)} documents to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()