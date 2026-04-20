import json
import os
import numpy as np
import faiss

# Keep sentence-transformers on PyTorch path only.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

from sentence_transformers import SentenceTransformer
from tqdm import tqdm

chunksPath = "../data/chunks.json"
indexPath = "../data/faiss_index.bin"
embeddingsPath = "../data/embeddings.npy"
modelName = "all-MiniLM-L6-v2"
nList = 16384
batchSize = 256

def loadChunks(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def encodeChunks(model, texts):
    print(f"Encoding {len(texts)} chunks with {modelName} ...")
    embeddings = model.encode(
        texts,
        batch_size=batchSize,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    return embeddings.astype(np.float32)

def buildIvfIndex(embeddings, nListParam):
    dim = embeddings.shape[1]
    nVectors = embeddings.shape[0]
    actualNList = min(nListParam, nVectors)
    if actualNList != nListParam:
        print(f"[WARN] Only {nVectors} vectors available; reducing nlist from {nListParam} to {actualNList}")
    
    quantizer = faiss.IndexFlatL2(dim)
    m = 8 # Number of subquantizers for Product Quantization
    nbits = 8 # bits per subvector
    index = faiss.IndexIVFPQ(quantizer, dim, actualNList, m, nbits)
    
    print(f"Training IVF-PQ index  (nlist={actualNList}, dim={dim}, m={m}) ...")
    index.train(embeddings)
    
    print("Adding vectors to the index ...")
    index.add(embeddings)
    
    print(f"Index built -> total vectors: {index.ntotal}")
    return index

def main():
    chunks = loadChunks(chunksPath)
    texts = [c["text"] for c in chunks]
    print(f"Loaded {len(texts)} chunks from {chunksPath}")
    
    model = SentenceTransformer(modelName)
    embeddings = encodeChunks(model, texts)
    
    index = buildIvfIndex(embeddings, nList)
    
    os.makedirs(os.path.dirname(indexPath), exist_ok=True)
    
    faiss.write_index(index, indexPath)
    print(f"Saved Faiss index  ->  {indexPath}")
    
    np.save(embeddingsPath, embeddings)
    print(f"Saved embeddings   ->  {embeddingsPath}")

if __name__ == "__main__":
    main()
