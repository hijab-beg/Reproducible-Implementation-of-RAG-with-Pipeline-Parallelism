import argparse
import os
from pathlib import Path

# Prevent transformers from trying to import TensorFlow/Keras.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

from benchmark_piperag import run_baseline, run_pipeline_mode
from faiss_retriever import FaissRetriever
from llm_client import LLMClient
from piperag_generator import PipeRAGGenerator
from piperag_pipeline_engine import PipeRAGConfig, PipeRAGPipelineEngine


def _resolve_default_paths() -> tuple[Path, Path]:
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data"
    return data_dir / "chunks.json", data_dir / "faiss_index.bin"


def _print_retrieved_chunks(title: str, chunks: list[dict]):
    print(title)
    for i, chunk in enumerate(chunks, start=1):
        print(f"\n--- Chunk {i} ---")
        print(f"chunk_id: {chunk['chunk_id']}")
        print(f"doc_id:   {chunk['doc_id']}")
        print(f"score:    {chunk['score']:.4f}")
        print(chunk["text"][:200] + "...")


def _run_demo_for_query(generator, pipeline_engine, args, user_query: str):
    print(f"\n{'#' * 12} Query: {user_query} {'#' * 12}\n")

    print("=== INITIAL GENERATION ===\n")
    initial_result = generator.initial_generation(user_query)

    _print_retrieved_chunks("Retrieved Chunks:", initial_result["retrieved_chunks"])
    print("Initial Answer:")
    print(initial_result["answer"])

    print("\n=== STALE QUERY CONTINUATION ===\n")
    continuation_result = generator.continue_generation_with_stale_retrieval(
        user_query=user_query,
        partial_answer=initial_result["answer"],
        step_number=2,
    )

    print("Stale Query Used:")
    print(continuation_result["stale_query"])

    print()
    _print_retrieved_chunks("Retrieved Chunks for Continuation:", continuation_result["retrieved_chunks"])

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
            max_total_tokens=args.max_total_tokens,
            top_k=args.top_k,
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


def _build_components(args):
    if args.backend:
        os.environ["LLM_BACKEND"] = args.backend
    if args.groq_api_key:
        os.environ["GROQ_API_KEY"] = args.groq_api_key
    if args.groq_base_url:
        os.environ["GROQ_BASE_URL"] = args.groq_base_url
    if args.groq_model:
        os.environ["GROQ_MODEL"] = args.groq_model
    if args.gemini_api_key:
        os.environ["GEMINI_API_KEY"] = args.gemini_api_key
    if args.ollama_base_url:
        os.environ["OLLAMA_BASE_URL"] = args.ollama_base_url
    if args.ollama_model:
        os.environ["OLLAMA_MODEL"] = args.ollama_model
    if args.fallback_ollama:
        os.environ["GEMINI_FALLBACK_TO_OLLAMA"] = "1"
    if args.max_rpm is not None:
        os.environ["LLM_MAX_RPM"] = str(args.max_rpm)

    chunks_path = Path(args.chunks_path)
    index_path = Path(args.index_path)

    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunks file not found: {chunks_path}")
    if not index_path.exists():
        raise FileNotFoundError(f"FAISS index file not found: {index_path}")

    retriever = FaissRetriever(
        chunks_path=str(chunks_path),
        index_path=str(index_path),
    )
    llm_client = LLMClient(model_name=args.model, backend=args.backend)
    generator = PipeRAGGenerator(
        retriever=retriever,
        llm_client=llm_client,
        top_k=args.top_k,
        retrieval_interval=1,
    )
    pipeline_engine = PipeRAGPipelineEngine(retriever=retriever, llm_client=llm_client)

    return generator, pipeline_engine


def run_demo(args):
    generator, pipeline_engine = _build_components(args)
    demo_queries = [
        args.query,
        "Who developed the C programming language?",
        "What is retrieval-augmented generation?",
    ]

    for demo_query in demo_queries:
        _run_demo_for_query(generator, pipeline_engine, args, demo_query)


def run_retriever_test(args):
    generator, _ = _build_components(args)
    retriever = generator.retriever

    print("\n" + "=" * 60)
    print("  nprobe = 1   (fast / low quality)")
    print("=" * 60 + "\n")
    results_low = retriever.retrieve(args.query, k=2, nProbe=1)
    _print_retrieved_chunks("Results:", results_low)

    print("\n" + "=" * 60)
    print("  nprobe = 10  (balanced)")
    print("=" * 60 + "\n")
    results_med = retriever.retrieve(args.query, k=2, nProbe=10)
    _print_retrieved_chunks("Results:", results_med)

    print("\n" + "=" * 60)
    print("  nprobe = 64  (thorough / high quality)")
    print("=" * 60 + "\n")
    results_high = retriever.retrieve(args.query, k=2, nProbe=64)
    _print_retrieved_chunks("Results:", results_high)


def run_benchmark(args):
    generator, pipeline_engine = _build_components(args)

    queries = [
        "When was the first barbie movie released?",
        "Who developed the C programming language?",
        "What is retrieval-augmented generation?",
    ]

    adaptive_model = generator.build_adaptive_model(
        sample_queries=queries,
        nprobe_values=[1, 5, 10, 20, 32],
        min_nprobe=1,
        max_nprobe=64,
    )

    ablations = [
        (
            "s1_only",
            PipeRAGConfig(
                m_prime=64,
                max_total_tokens=args.max_total_tokens,
                top_k=args.top_k,
                default_nprobe=10,
                enable_s1_pipeline=True,
                enable_s2_flexible_interval=False,
                enable_s3_adaptive_nprobe=False,
            ),
        ),
        (
            "s1_s2",
            PipeRAGConfig(
                m_prime=32,
                max_total_tokens=args.max_total_tokens,
                top_k=args.top_k,
                default_nprobe=10,
                enable_s1_pipeline=True,
                enable_s2_flexible_interval=True,
                enable_s3_adaptive_nprobe=False,
            ),
        ),
        (
            "s1_s2_s3",
            PipeRAGConfig(
                m_prime=32,
                max_total_tokens=args.max_total_tokens,
                top_k=args.top_k,
                default_nprobe=10,
                enable_s1_pipeline=True,
                enable_s2_flexible_interval=True,
                enable_s3_adaptive_nprobe=True,
                budget_safety_factor=0.9,
            ),
        ),
    ]

    results_by_mode: dict[str, list[dict]] = {
        "baseline": [],
        "s1_only": [],
        "s1_s2": [],
        "s1_s2_s3": [],
    }

    for query in queries:
        baseline = run_baseline(generator, query)
        results_by_mode["baseline"].append(baseline)

        for mode_name, cfg in ablations:
            run = run_pipeline_mode(
                pipeline_engine,
                query,
                retrieval_model=adaptive_model,
                cfg=cfg,
                mode_name=mode_name,
            )
            results_by_mode[mode_name].append(run)

    print("Query\tBaseline (ms)\tS1 Only (ms)\tS1+S2 (ms)\tS1+S2+S3 (ms)\tS1 Overlap\tS1+S2 Overlap\tS1+S2+S3 Overlap")
    for i, query in enumerate(queries):
        baseline = results_by_mode["baseline"][i]
        s1 = results_by_mode["s1_only"][i]
        s1s2 = results_by_mode["s1_s2"][i]
        s1s2s3 = results_by_mode["s1_s2_s3"][i]
        print(
            f"{query}\t"
            f"{baseline['latency_ms']:.2f}\t"
            f"{s1['latency_ms']:.2f}\t"
            f"{s1s2['latency_ms']:.2f}\t"
            f"{s1s2s3['latency_ms']:.2f}\t"
            f"{s1['overlap_ratio']:.2f}\t"
            f"{s1s2['overlap_ratio']:.2f}\t"
            f"{s1s2s3['overlap_ratio']:.2f}"
        )

    def avg_latency(mode_name: str) -> float:
        runs = results_by_mode[mode_name]
        return sum(run["latency_ms"] for run in runs) / len(runs)

    def avg_overlap(mode_name: str) -> float:
        runs = results_by_mode[mode_name]
        return sum(run.get("overlap_ratio", 0.0) for run in runs) / len(runs)

    avg_baseline = avg_latency("baseline")
    avg_s1 = avg_latency("s1_only")
    avg_s1s2 = avg_latency("s1_s2")
    avg_s1s2s3 = avg_latency("s1_s2_s3")

    print("\nAverages")
    print("Metric\tValue")
    print(f"Average Baseline (ms)\t{avg_baseline:.2f}")
    print(f"Average S1 Only (ms)\t{avg_s1:.2f}")
    print(f"Average S1+S2 (ms)\t{avg_s1s2:.2f}")
    print(f"Average S1+S2+S3 (ms)\t{avg_s1s2s3:.2f}")
    print(f"Average S1 Overlap\t{avg_overlap('s1_only'):.2f}")
    print(f"Average S1+S2 Overlap\t{avg_overlap('s1_s2'):.2f}")
    print(f"Average S1+S2+S3 Overlap\t{avg_overlap('s1_s2_s3'):.2f}")

    if avg_s1 > 0:
        print(f"Speedup Baseline / S1 Only\t{avg_baseline / avg_s1:.2f}x")
    if avg_s1s2 > 0:
        print(f"Speedup Baseline / S1+S2\t{avg_baseline / avg_s1s2:.2f}x")
    if avg_s1s2s3 > 0:
        print(f"Speedup Baseline / S1+S2+S3\t{avg_baseline / avg_s1s2s3:.2f}x")


def parse_args():
    default_chunks, default_index = _resolve_default_paths()

    parser = argparse.ArgumentParser(
        description="Single launcher for PipeRAG demo, retriever test, and benchmark.",
    )
    parser.add_argument(
        "--task",
        choices=["demo", "retriever-test", "benchmark", "all"],
        default="demo",
        help="Task to run from one entry point.",
    )
    parser.add_argument(
        "--query",
        default="When was the first barbie movie released?",
        help="User query for demo and retriever-test tasks.",
    )
    parser.add_argument(
        "--backend",
        choices=["groq", "gemini", "ollama"],
        default=None,
        help="LLM backend. If omitted, existing env behavior is used.",
    )
    parser.add_argument(
        "--groq-api-key",
        default=None,
        help="Optional Groq API key for this run only.",
    )
    parser.add_argument(
        "--groq-base-url",
        default=None,
        help="Optional Groq base URL override.",
    )
    parser.add_argument(
        "--groq-model",
        default=None,
        help="Optional Groq model override.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name override for selected backend.",
    )
    parser.add_argument(
        "--gemini-api-key",
        default=None,
        help="Optional Gemini API key for this run only.",
    )
    parser.add_argument(
        "--ollama-base-url",
        default=None,
        help="Optional Ollama base URL override.",
    )
    parser.add_argument(
        "--ollama-model",
        default=None,
        help="Optional Ollama model override.",
    )
    parser.add_argument(
        "--fallback-ollama",
        action="store_true",
        help="When using Gemini, auto-fallback to Ollama on quota errors (429).",
    )
    parser.add_argument(
        "--max-rpm",
        type=int,
        default=None,
        help="Maximum LLM requests per minute (Gemini throttling). Default is 8 on Gemini.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Retriever top-k.",
    )
    parser.add_argument(
        "--max-total-tokens",
        type=int,
        default=180,
        help="Max total generation tokens for pipeline mode.",
    )
    parser.add_argument(
        "--chunks-path",
        default=str(default_chunks),
        help="Path to chunks.json.",
    )
    parser.add_argument(
        "--index-path",
        default=str(default_index),
        help="Path to faiss_index.bin.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.task == "demo":
        run_demo(args)
    elif args.task == "retriever-test":
        run_retriever_test(args)
    elif args.task == "benchmark":
        run_benchmark(args)
    else:
        run_demo(args)
        run_benchmark(args)


if __name__ == "__main__":
    main()
