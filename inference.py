import argparse
import os
import sys
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _resolve_path(root: Path, maybe_relative: str | None) -> str | None:
    if maybe_relative is None:
        return None
    p = Path(maybe_relative)
    if p.is_absolute():
        return str(p)
    return str((root / p).resolve())


def _coerce_scalar(value: str) -> Any:
    lowered = value.strip().lower()
    if lowered in {"null", "none", "~", ""}:
        return None
    if lowered in {"true", "yes", "on"}:
        return True
    if lowered in {"false", "no", "off"}:
        return False

    try:
        if "." in lowered:
            return float(lowered)
        return int(lowered)
    except ValueError:
        pass

    if (value.startswith("\"") and value.endswith("\"")) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    return value


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}

    try:
        import yaml  # type: ignore

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise TypeError("config.yaml must parse to a dict")
        return data
    except ModuleNotFoundError:
        cfg: dict[str, Any] = {}
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("-"):
                continue
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            if value and not value.startswith(("\"", "'")):
                value = value.split("#", 1)[0].strip()
            cfg[key] = _coerce_scalar(value)
        return cfg


def main() -> int:
    parser = argparse.ArgumentParser(description="Root-level inference wrapper (does not modify src/)")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--query", default=None, help="Override query from config")
    parser.add_argument("--print-timeline", action="store_true", help="Print per-step pipeline timeline")
    args = parser.parse_args()

    root = _repo_root()
    cfg = load_config(root / args.config)

    # Match src/run.py environment behavior for TF/Keras imports.
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

    # Apply backend/model settings (if provided) to env for LLMClient.
    if cfg.get("backend"):
        os.environ["LLM_BACKEND"] = str(cfg["backend"])

    if cfg.get("max_rpm") is not None:
        os.environ["LLM_MAX_RPM"] = str(cfg["max_rpm"])
    if bool(cfg.get("fallback_ollama")):
        os.environ["GEMINI_FALLBACK_TO_OLLAMA"] = "1"

    # Make src/ importable without touching it.
    sys.path.insert(0, str((root / "src").resolve()))

    from faiss_retriever import FaissRetriever  # noqa: E402
    from llm_client import LLMClient  # noqa: E402
    from piperag_generator import PipeRAGGenerator  # noqa: E402
    from piperag_pipeline_engine import PipeRAGConfig, PipeRAGPipelineEngine  # noqa: E402

    chunks_path = _resolve_path(root, cfg.get("chunks_path")) or str((root / "data/chunks.json").resolve())
    index_path = _resolve_path(root, cfg.get("index_path")) or str((root / "data/faiss_index.bin").resolve())

    query = args.query or cfg.get("query")
    if not query:
        raise SystemExit("No query provided. Set 'query' in config.yaml or pass --query.")

    top_k = int(cfg.get("top_k") or 3)

    retriever = FaissRetriever(chunks_path=chunks_path, index_path=index_path)
    llm_client = LLMClient(model_name=cfg.get("model"), backend=cfg.get("backend"))

    pipeline_engine = PipeRAGPipelineEngine(retriever=retriever, llm_client=llm_client)
    generator = PipeRAGGenerator(retriever=retriever, llm_client=llm_client, top_k=top_k, retrieval_interval=1)

    # Adaptive nprobe model is optional.
    retrieval_model = None
    if bool(cfg.get("enable_s3_adaptive_nprobe", True)):
        retrieval_model = generator.build_adaptive_model(
            sample_queries=[
                str(query),
                "Who wrote the first Harry Potter book?",
                "When was Python first released?",
            ],
            nprobe_values=[1, 5, 10, 20, 32],
            min_nprobe=1,
            max_nprobe=64,
        )

    pipe_cfg = PipeRAGConfig(
        m_prime=int(cfg.get("m_prime") or 32),
        max_total_tokens=int(cfg.get("max_total_tokens") or 180),
        top_k=top_k,
        default_nprobe=int(cfg.get("default_nprobe") or 10),
        enable_s1_pipeline=bool(cfg.get("enable_s1_pipeline", True)),
        enable_s2_flexible_interval=bool(cfg.get("enable_s2_flexible_interval", True)),
        enable_s3_adaptive_nprobe=bool(cfg.get("enable_s3_adaptive_nprobe", True)),
        enable_s4_uncertainty_gating=bool(cfg.get("enable_s4_uncertainty_gating", False)),
        uncertainty_threshold=float(cfg.get("uncertainty_threshold") or 0.5),
        budget_safety_factor=float(cfg.get("budget_safety_factor") or 0.9),
    )

    result = pipeline_engine.run(user_query=str(query), cfg=pipe_cfg, retrieval_model=retrieval_model)

    print(result["answer"])
    if args.print_timeline:
        print("\n--- timeline ---")
        for event in result.get("timeline", []):
            print(
                f"step={event['step']} source={event['retrieval_source']} "
                f"nprobe={event['nprobe']} gen_tokens={event['generated_tokens']}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
