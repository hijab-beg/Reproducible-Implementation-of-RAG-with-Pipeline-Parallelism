import json
import random

CHUNKS_PATH = "../data/chunks.json"
OUTPUT_PATH = "../data/validation_queries.json"
VALIDATION_SIZE = 1000
RANDOM_SEED = 42


def main():
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    if len(chunks) < VALIDATION_SIZE:
        raise ValueError(
            f"Not enough chunks for validation set: have {len(chunks)}, need {VALIDATION_SIZE}."
        )

    rng = random.Random(RANDOM_SEED)
    selected_indices = rng.sample(range(len(chunks)), VALIDATION_SIZE)

    validation = []
    for i, idx in enumerate(selected_indices):
        chunk = chunks[idx]
        validation.append(
            {
                "query_id": f"val_{i}",
                "chunk_id": chunk.get("chunk_id"),
                "doc_id": chunk.get("doc_id"),
                "query": chunk.get("text", "").strip(),
            }
        )

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(validation, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(validation)} validation queries to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
