import json
import os
from datasets import load_dataset
from tqdm import tqdm

OUTPUT_PATH = "../data/raw_docs.json"

# Increased to support larger chunk corpora (e.g., ~600k chunks after chunking).
TARGET_DOCS = 30000
MIN_CHAR_LENGTH = 1200
PRIMARY_DATASET_NAME = "c4"
PRIMARY_DATASET_CONFIG = "en"
SECONDARY_DATASET_NAME = "wikipedia"
SECONDARY_DATASET_CONFIG = "20220301.en"
PRIMARY_TARGET_DOCS = TARGET_DOCS // 2
SECONDARY_TARGET_DOCS = TARGET_DOCS - PRIMARY_TARGET_DOCS


def _normalize_text(text: str) -> str:
    return " ".join(text.split()).strip()


def _load_streaming_subset(dataset_name: str, dataset_config: str, target_docs: int, min_char_length: int, desc: str, include_title: bool, docs_offset: int = 0):
    dataset = load_dataset(
        dataset_name,
        dataset_config,
        split="train",
        streaming=True,
        trust_remote_code=True
    )

    docs = []

    for sample in tqdm(dataset, desc=desc):
        title = _normalize_text(sample.get("title", "")) if include_title else ""
        text = _normalize_text(sample.get("text", ""))

        if title:
            text = f"{title}\n\n{text}"

        if len(text) < min_char_length:
            continue

        docs.append({
            "doc_id": f"doc_{docs_offset + len(docs)}",
            "source": dataset_name,
            "title": title,
            "text": text
        })

        if len(docs) >= target_docs:
            break

    return docs


def load_c4_subset(target_docs: int, min_char_length: int):
    return _load_streaming_subset(
        PRIMARY_DATASET_NAME,
        PRIMARY_DATASET_CONFIG,
        target_docs,
        min_char_length,
        desc="Loading C4 samples",
        include_title=False,
    )


def load_wikipedia_subset(target_docs: int, min_char_length: int, docs_offset: int = 0):
    return _load_streaming_subset(
        SECONDARY_DATASET_NAME,
        SECONDARY_DATASET_CONFIG,
        target_docs,
        min_char_length,
        desc="Loading Wikipedia samples",
        include_title=True,
        docs_offset=docs_offset,
    )


def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    c4_docs = load_c4_subset(PRIMARY_TARGET_DOCS, MIN_CHAR_LENGTH)
    wikipedia_docs = load_wikipedia_subset(
        SECONDARY_TARGET_DOCS,
        MIN_CHAR_LENGTH,
        docs_offset=len(c4_docs),
    )
    docs = c4_docs + wikipedia_docs

    with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
        json.dump(docs, file, ensure_ascii=False, indent=2)

    print(f"Saved {len(docs)} documents to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()