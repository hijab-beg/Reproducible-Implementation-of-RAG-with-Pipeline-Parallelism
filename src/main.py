from faiss_retriever import FaissRetriever
from llm_client import LLMClient
from piperag_generator import PipeRAGGenerator
from piperag_pipeline_engine import PipeRAGConfig, PipeRAGPipelineEngine


def print_retrieved_chunks(title: str, chunks: list[dict]):
    print(title)
    for i, chunk in enumerate(chunks, start=1):
        print(f"\n--- Chunk {i} ---")
        print(f"chunk_id: {chunk['chunk_id']}")
        print(f"doc_id:   {chunk['doc_id']}")
        print(f"score:    {chunk['score']:.4f}")
        print(chunk["text"][:200] + "...")


def is_chat_exit_command(user_text: str) -> bool:
    return user_text.strip().lower() in {"bye", "exit", "quit"}


def run_query(generator, pipeline_engine, user_query: str):
    print("\n=== INITIAL GENERATION ===\n")
    initial_result = generator.initial_generation(user_query)

    print_retrieved_chunks("Retrieved Chunks:", initial_result["retrieved_chunks"])
    print("Initial Answer:")
    print(initial_result["answer"])

    print("\n=== STALE QUERY CONTINUATION ===\n")
    continuation_result = generator.continue_generation_with_stale_retrieval(
        user_query=user_query,
        partial_answer=initial_result["answer"],
        step_number=2
    )

    print("Stale Query Used:")
    print(continuation_result["stale_query"])

    print()
    print_retrieved_chunks("Retrieved Chunks for Continuation:", continuation_result["retrieved_chunks"])

    print("Continuation:")
    print(continuation_result["continuation"])

    print("\n=== PIPELINE MODE (S1 + S2 + S3) ===\n")
    adaptive_model = generator.build_adaptive_model(
        sample_queries=[
            user_query,
            "Who wrote the first Harry Potter book?",
            "When was Python first released?",
        ],
        nprobe_values=[1, 5, 10, 20, 32],
        min_nprobe=1,
        max_nprobe=64,
    )

    pipeline_result = pipeline_engine.run(
        user_query=user_query,
        cfg=PipeRAGConfig(
            m_prime=32,
            max_total_tokens=180,
            top_k=3,
            default_nprobe=10,
            enable_s1_pipeline=True,
            enable_s2_flexible_interval=True,
            enable_s3_adaptive_nprobe=True,
        ),
        retrieval_model=adaptive_model,
    )

    print("Pipeline Answer:")
    print(pipeline_result["answer"])

    print("\nPipeline Timeline:")
    for event in pipeline_result["timeline"]:
        print(
            f"step={event['step']} "
            f"source={event['retrieval_source']} nprobe={event['nprobe']} "
            f"gen_tokens={event['generated_tokens']}"
        )


def main():
    retriever = FaissRetriever(
        chunks_path="../data/chunks.json",
        index_path="../data/faiss_index.bin"
    )
    llm_client = LLMClient()
    generator = PipeRAGGenerator(
        retriever=retriever,
        llm_client=llm_client,
        top_k=3,
        retrieval_interval=1
    )
    pipeline_engine = PipeRAGPipelineEngine(retriever=retriever, llm_client=llm_client)

    print("\nPipeRAG Chatbot Mode")
    print("Type your question and press Enter.")
    print("Type 'bye' (or 'exit'/'quit') to end.\n")

    while True:
        try:
            user_query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSession ended.")
            break

        if not user_query:
            continue
        if is_chat_exit_command(user_query):
            print("Goodbye!")
            break

        run_query(generator, pipeline_engine, user_query)


if __name__ == "__main__":
    main()