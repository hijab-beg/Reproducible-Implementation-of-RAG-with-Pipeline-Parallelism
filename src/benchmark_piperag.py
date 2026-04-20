import time

from faiss_retriever import FaissRetriever
from llm_client import LLMClient
from piperag_generator import PipeRAGGenerator
from piperag_pipeline_engine import PipeRAGConfig, PipeRAGPipelineEngine


def run_baseline(generator, query: str):
    start = time.perf_counter()
    initial = generator.initial_generation(query)
    continuation = generator.continue_generation_with_stale_retrieval(
        user_query=query,
        partial_answer=initial["answer"],
        step_number=2,
    )
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return {
        "mode": "baseline",
        "latency_ms": elapsed_ms,
        "answer": (initial["answer"] + " " + continuation["continuation"]).strip(),
        "overlap_ratio": 0.0,
    }


def run_pipeline_mode(engine, query: str, retrieval_model, cfg: PipeRAGConfig, mode_name: str):
    result = engine.run(user_query=query, cfg=cfg, retrieval_model=retrieval_model)
    return {
        "mode": mode_name,
        "latency_ms": result["latency_ms"],
        "answer": result["answer"],
        "overlap_ratio": result["overlap_ratio"],
        "prefetch_wait_ratio": result["prefetch_wait_ratio"],
        "retrieval_within_budget_ratio": result["retrieval_within_budget_ratio"],
        "average_nprobe": result["average_nprobe"],
    }


def main():
    retriever = FaissRetriever(
        chunks_path="../data/chunks.json",
        index_path="../data/faiss_index.bin",
    )
    llm_client = LLMClient()
    generator = PipeRAGGenerator(
        retriever=retriever,
        llm_client=llm_client,
        top_k=3,
        retrieval_interval=1,
    )
    pipeline_engine = PipeRAGPipelineEngine(retriever=retriever, llm_client=llm_client)

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
                max_total_tokens=180,
                top_k=3,
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
                max_total_tokens=180,
                top_k=3,
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
                max_total_tokens=180,
                top_k=3,
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

        print("=" * 60)
        print(f"Query: {query}")
        print(f"Baseline latency: {baseline['latency_ms']:.2f} ms")
        for mode_name, _ in ablations:
            current = results_by_mode[mode_name][-1]
            print(
                f"{mode_name} latency: {current['latency_ms']:.2f} ms | "
                f"overlap={current['overlap_ratio']:.2f} | "
                f"budget_ok={current['retrieval_within_budget_ratio']:.2f} | "
                f"avg_nprobe={current['average_nprobe']:.1f}"
            )

    def avg_latency(mode_name: str) -> float:
        runs = results_by_mode[mode_name]
        return sum(run["latency_ms"] for run in runs) / len(runs)

    def avg_overlap(mode_name: str) -> float:
        runs = results_by_mode[mode_name]
        return sum(run.get("overlap_ratio", 0.0) for run in runs) / len(runs)

    def avg_budget_ok(mode_name: str) -> float:
        runs = results_by_mode[mode_name]
        return sum(run.get("retrieval_within_budget_ratio", 0.0) for run in runs) / len(runs)

    def avg_nprobe(mode_name: str) -> float:
        runs = results_by_mode[mode_name]
        return sum(run.get("average_nprobe", 0.0) for run in runs) / len(runs)

    avg_baseline = avg_latency("baseline")
    avg_s1 = avg_latency("s1_only")
    avg_s1s2 = avg_latency("s1_s2")
    avg_s1s2s3 = avg_latency("s1_s2_s3")

    print("\n" + "#" * 60)
    print("Paper-style Person 3 Ablation Summary")
    print(f"Average baseline latency: {avg_baseline:.2f} ms")
    print(
        f"Average s1_only latency: {avg_s1:.2f} ms | overlap={avg_overlap('s1_only'):.2f} | "
        f"budget_ok={avg_budget_ok('s1_only'):.2f} | avg_nprobe={avg_nprobe('s1_only'):.1f}"
    )
    print(
        f"Average s1_s2 latency: {avg_s1s2:.2f} ms | overlap={avg_overlap('s1_s2'):.2f} | "
        f"budget_ok={avg_budget_ok('s1_s2'):.2f} | avg_nprobe={avg_nprobe('s1_s2'):.1f}"
    )
    print(
        f"Average s1_s2_s3 latency: {avg_s1s2s3:.2f} ms | overlap={avg_overlap('s1_s2_s3'):.2f} | "
        f"budget_ok={avg_budget_ok('s1_s2_s3'):.2f} | avg_nprobe={avg_nprobe('s1_s2_s3'):.1f}"
    )

    if avg_s1 > 0:
        print(f"Speedup baseline/s1_only: {avg_baseline / avg_s1:.2f}x")
    if avg_s1s2 > 0:
        print(f"Speedup baseline/s1_s2: {avg_baseline / avg_s1s2:.2f}x")
    if avg_s1s2s3 > 0:
        print(f"Speedup baseline/s1_s2_s3: {avg_baseline / avg_s1s2s3:.2f}x")


if __name__ == "__main__":
    main()
