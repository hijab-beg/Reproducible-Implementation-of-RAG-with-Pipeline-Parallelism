from faiss_retriever import FaissRetriever

def printDivider(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")

def printResults(resultList):
    for rank, resultItem in enumerate(resultList, start=1):
        print(f"  [{rank}]  chunk_id : {resultItem['chunk_id']}")
        print(f"       doc_id   : {resultItem['doc_id']}")
        print(f"       score    : {resultItem['score']:.4f}")
        print(f"       text     : {resultItem['text'][:120]}...")
        print()

def main():
    testRetriever = FaissRetriever(
        chunks_path="../data/chunks.json",
        index_path="../data/faiss_index.bin",
    )
    
    testQuery = "When was the first barbie movie released?"
    
    printDivider("nprobe = 1   (fast / low quality)")
    resultsLow = testRetriever.retrieve(testQuery, k=2, nProbe=1)
    printResults(resultsLow)
    
    printDivider("nprobe = 10  (balanced)")
    resultsMed = testRetriever.retrieve(testQuery, k=2, nProbe=10)
    printResults(resultsMed)
    
    printDivider("nprobe = 64  (thorough / high quality)")
    resultsHigh = testRetriever.retrieve(testQuery, k=2, nProbe=64)
    printResults(resultsHigh)

if __name__ == "__main__":
    main()
