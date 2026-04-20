import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class TfidfRetriever:
    def __init__(self, chunks_path: str):
        with open(chunks_path, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

        self.texts = [chunk["text"] for chunk in self.chunks]

        print("Building TF-IDF matrix...")
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=5000
        )
        self.matrix = self.vectorizer.fit_transform(self.texts)

    def search(self, query: str, top_k: int = 3):
        query_vec = self.vectorizer.transform([query])
        scores = (self.matrix @ query_vec.T).toarray().flatten()

        top_indices = np.argsort(scores)[-top_k:][::-1]

        results = []
        for index in top_indices:
            results.append({
                "chunk_id": self.chunks[index]["chunk_id"],
                "doc_id": self.chunks[index]["doc_id"],
                "score": float(scores[index]),
                "text": self.chunks[index]["text"]
            })

        return results