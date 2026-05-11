import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
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
        # Minimal fallback: top-level key: value pairs only (no nesting).
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


def _run(cmd: list[str], *, cwd: Path, dry_run: bool) -> int:
    printable = " ".join(json.dumps(part) for part in cmd)
    print(f"[train] $ {printable}")
    if dry_run:
        return 0
    return subprocess.call(cmd, cwd=str(cwd))


def main() -> int:
    parser = argparse.ArgumentParser(description="Root-level training wrapper (does not modify src/)")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument(
        "--mode",
        choices=["index", "benchmark", "all"],
        default=None,
        help="What to run (default comes from config or 'all')",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    parser.add_argument(
        "--write-metrics",
        action="store_true",
        help="Optionally write simple JSON summaries into results/*.json (overwrites).",
    )

    args = parser.parse_args()

    root = _repo_root()
    cfg = load_config(root / args.config)

    mode = args.mode or str(cfg.get("train_mode") or "all")
    if mode not in {"index", "benchmark", "all"}:
        mode = "all"

    python = sys.executable
    src_dir = root / "src"

    # Ensure results directory exists.
    results_dir = root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    chunks_path = _resolve_path(root, cfg.get("chunks_path"))
    index_path = _resolve_path(root, cfg.get("index_path"))

    rc = 0

    if mode in {"index", "all"}:
        # These scripts already use ../data/* paths internally.
        rc = _run([python, "chunk_data.py"], cwd=src_dir, dry_run=args.dry_run)
        if rc != 0:
            return rc
        rc = _run([python, "build_index.py"], cwd=src_dir, dry_run=args.dry_run)
        if rc != 0:
            return rc

    if mode in {"benchmark", "all"}:
        cmd = [python, "run.py", "--task", "benchmark"]

        # Optional config-driven arguments.
        if cfg.get("backend"):
            cmd += ["--backend", str(cfg["backend"])]
        if cfg.get("model"):
            cmd += ["--model", str(cfg["model"])]
        if cfg.get("max_rpm") is not None:
            cmd += ["--max-rpm", str(cfg["max_rpm"])]
        if bool(cfg.get("fallback_ollama")):
            cmd += ["--fallback-ollama"]

        if cfg.get("top_k") is not None:
            cmd += ["--top-k", str(cfg["top_k"])]
        if cfg.get("max_total_tokens") is not None:
            cmd += ["--max-total-tokens", str(cfg["max_total_tokens"])]
        if cfg.get("val_queries") is not None:
            cmd += ["--val-queries", str(cfg["val_queries"])]
        if cfg.get("val_seed") is not None:
            cmd += ["--val-seed", str(cfg["val_seed"])]

        if chunks_path:
            cmd += ["--chunks-path", chunks_path]
        if index_path:
            cmd += ["--index-path", index_path]

        rc = _run(cmd, cwd=src_dir, dry_run=args.dry_run)
        if rc != 0:
            return rc

    if args.write_metrics and not args.dry_run:
        # Placeholder structure for later population.
        baseline_path = results_dir / "baseline_metrics.json"
        improved_path = results_dir / "improved_metrics.json"
        stamp = datetime.utcnow().isoformat() + "Z"

        baseline_path.write_text(json.dumps({"generated_at": stamp, "notes": "populate later"}, indent=2), encoding="utf-8")
        improved_path.write_text(json.dumps({"generated_at": stamp, "notes": "populate later"}, indent=2), encoding="utf-8")

    print("[train] Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
