import json
import time
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

class FaissRetriever:
    def __init__(
        self,
        chunks_path="../data/chunks.json",
        index_path="../data/faiss_index.bin",
        modelName="all-MiniLM-L6-v2",
    ):
        with open(chunks_path, "r", encoding="utf-8") as fileObj:
            self.chunks = json.load(fileObj)
        print(f"[FaissRetriever] Loaded {len(self.chunks)} chunks")
        
        self.index = faiss.read_index(index_path)
        print(f"[FaissRetriever] Loaded Faiss index  (ntotal={self.index.ntotal})")
        
        self.model = SentenceTransformer(modelName)
        print(f"[FaissRetriever] Embedding model ready ({modelName})")

    def retrieve(self, queryText, k=2, nProbe=10, return_metadata=False):
        self.index.nprobe = nProbe
        
        queryVec = self.model.encode([queryText], convert_to_numpy=True).astype(np.float32)
        
        startTime = time.perf_counter()
        distances, indices = self.index.search(queryVec, k)
        latencyMs = (time.perf_counter() - startTime) * 1000
        
        print(f"[FaissRetriever] Search done  (nprobe={nProbe}, k={k}, latency={latencyMs:.2f} ms)")
        
        results = []
        for rank, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx == -1:
                continue
            chunk = self.chunks[idx]
            results.append({
                "chunk_id": chunk["chunk_id"],
                "doc_id": chunk["doc_id"],
                "score": float(dist),
                "text": chunk["text"],
            })
            
        if return_metadata:
            return results, {
                "latency_ms": latencyMs,
                "nprobe": nProbe,
                "k": k,
                "query": queryText,
            }

        return results

    def search(self, query, top_k=2, nProbe=10, return_metadata=False):
        return self.retrieve(query, k=top_k, nProbe=nProbe, return_metadata=return_metadata)