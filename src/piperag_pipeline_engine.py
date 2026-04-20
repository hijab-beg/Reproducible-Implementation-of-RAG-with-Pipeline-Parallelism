from __future__ import annotations

import math
import time
from dataclasses import dataclass

from prompt_builder import build_augmented_prompt
from transformers import AutoTokenizer

from pipeline_controller import PipelineController
from performance_model import GenerationLatencyEMA, RetrievalLatencyModel


@dataclass
class PipeRAGConfig:
    # S2: Flexible retrieval interval m -> m' (RETRO baseline uses 64)
    m_prime: int = 64
    query_window_tokens: int = 64
    stale_offset_tokens: int | None = None

    # Generation and retrieval parameters
    max_total_tokens: int = 256
    top_k: int = 3
    default_nprobe: int = 10

    # Ablation toggles
    enable_s1_pipeline: bool = True
    enable_s2_flexible_interval: bool = True
    enable_s3_adaptive_nprobe: bool = True

    # S3 budget control
    budget_safety_factor: float = 0.9

    # Paper-inspired stale retrieval shift: drop stale prefix from retrieved chunks.
    apply_stale_shift_to_chunks: bool = True


class PipeRAGPipelineEngine:
    """Paper-oriented pipeline engine (S1/S2/S3) built on existing retriever + LLM APIs."""

    def __init__(self, retriever, llm_client):
        self.retriever = retriever
        self.llm_client = llm_client
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")

    def _stale_query_from_partial(
        self,
        partial_answer: str,
        fallback_query: str,
        query_window_tokens: int,
        stale_offset_tokens: int,
    ) -> str:
        token_ids = self.tokenizer.encode(partial_answer, add_special_tokens=False)
        if not token_ids:
            return fallback_query

        end_index = max(0, len(token_ids) - stale_offset_tokens)
        start_index = max(0, end_index - query_window_tokens)

        selected = token_ids[start_index:end_index]
        if not selected:
            selected = token_ids[-query_window_tokens:]

        text = self.tokenizer.decode(selected).strip()
        return text or fallback_query

    def _choose_nprobe(
        self,
        cfg: PipeRAGConfig,
        retrieval_model: RetrievalLatencyModel | None,
        generation_ema: GenerationLatencyEMA,
    ) -> int:
        if not cfg.enable_s3_adaptive_nprobe or retrieval_model is None:
            return cfg.default_nprobe

        budget_ms = generation_ema.budget(safety_factor=cfg.budget_safety_factor)
        return retrieval_model.pick_nprobe(budget_ms)

    def _shift_retrieved_chunks(self, chunks: list[dict], shift_tokens: int) -> list[dict]:
        if shift_tokens <= 0:
            return chunks

        shifted = []
        for chunk in chunks:
            text = chunk.get("text", "")
            token_ids = self.tokenizer.encode(text, add_special_tokens=False)
            trimmed = token_ids[shift_tokens:] if len(token_ids) > shift_tokens else []
            new_text = self.tokenizer.decode(trimmed).strip() if trimmed else ""

            updated = dict(chunk)
            updated["text"] = new_text
            shifted.append(updated)

        return shifted

    def run(
        self,
        user_query: str,
        cfg: PipeRAGConfig,
        retrieval_model: RetrievalLatencyModel | None = None,
    ) -> dict:
        effective_m = cfg.m_prime if cfg.enable_s2_flexible_interval else 64
        stale_offset = cfg.stale_offset_tokens
        if stale_offset is None:
            stale_offset = effective_m

        segments = max(1, math.ceil(cfg.max_total_tokens / effective_m))

        generation_ema = GenerationLatencyEMA()
        controller = PipelineController(self.retriever) if cfg.enable_s1_pipeline else None

        partial_answer = ""
        timeline = []

        t0 = time.perf_counter()
        try:
            for seg_idx in range(segments):
                step = seg_idx + 1
                step_budget_ms = generation_ema.budget(safety_factor=cfg.budget_safety_factor)

                # 1) Consume retrieval for this segment.
                if step == 1:
                    query_for_step = user_query
                    nprobe_used = self._choose_nprobe(cfg, retrieval_model, generation_ema)
                    chunks, ret_meta = self.retriever.search(
                        query_for_step,
                        top_k=cfg.top_k,
                        nProbe=nprobe_used,
                        return_metadata=True,
                    )
                    retrieval_source = "sync-initial"
                else:
                    prefetched_payload = None
                    retrieval_source = "sync"

                    if controller is not None:
                        prefetched_payload, status = controller.consume_prefetched_with_status(step)
                        if prefetched_payload is not None:
                            chunks, ret_meta = prefetched_payload
                            retrieval_source = status
                        else:
                            status = "none"

                    if prefetched_payload is None:
                        query_for_step = self._stale_query_from_partial(
                            partial_answer,
                            user_query,
                            query_window_tokens=cfg.query_window_tokens,
                            stale_offset_tokens=stale_offset,
                        )
                        nprobe_used = self._choose_nprobe(cfg, retrieval_model, generation_ema)
                        chunks, ret_meta = self.retriever.search(
                            query_for_step,
                            top_k=cfg.top_k,
                            nProbe=nprobe_used,
                            return_metadata=True,
                        )
                        retrieval_source = "sync-fallback" if controller is not None else "sync"

                if cfg.apply_stale_shift_to_chunks and step > 1:
                    chunks = self._shift_retrieved_chunks(chunks, stale_offset)

                # 2) Start prefetch for next segment (S1 overlap).
                next_step = step + 1
                if controller is not None and next_step <= segments:
                    query_for_next = self._stale_query_from_partial(
                        partial_answer,
                        user_query,
                        query_window_tokens=cfg.query_window_tokens,
                        stale_offset_tokens=stale_offset,
                    )
                    nprobe_next = self._choose_nprobe(cfg, retrieval_model, generation_ema)
                    controller.schedule_prefetch(
                        step=next_step,
                        query=query_for_next,
                        top_k=cfg.top_k,
                        nprobe=nprobe_next,
                    )

                # 3) Generate this segment.
                prompt = build_augmented_prompt(
                    user_query=user_query,
                    retrieved_chunks=[chunk["text"] for chunk in chunks],
                    partial_answer=partial_answer,
                )

                gen_start = time.perf_counter()
                continuation = self.llm_client.generate(prompt, max_tokens=effective_m)
                gen_ms = (time.perf_counter() - gen_start) * 1000.0
                generation_ema.observe(gen_ms)

                partial_answer = (partial_answer + " " + continuation).strip()

                generated_tokens = len(self.tokenizer.encode(continuation, add_special_tokens=False))

                timeline.append(
                    {
                        "step": step,
                        "segment_tokens_target": effective_m,
                        "generated_tokens": generated_tokens,
                        "retrieval_source": retrieval_source,
                        "retrieval_latency_ms": float(ret_meta["latency_ms"]),
                        "retrieval_within_budget": float(ret_meta["latency_ms"]) <= float(step_budget_ms),
                        "retrieval_budget_ms": float(step_budget_ms),
                        "generation_latency_ms": float(gen_ms),
                        "nprobe": int(ret_meta["nprobe"]),
                        "stale_query": ret_meta["query"],
                        "m_prime": effective_m,
                        "stale_offset_tokens": stale_offset,
                    }
                )
        finally:
            if controller is not None:
                controller.shutdown()

        total_ms = (time.perf_counter() - t0) * 1000.0
        retrieval_steps = len(timeline)
        prefetch_hits = sum(1 for event in timeline if event["retrieval_source"] == "prefetch-hit")
        prefetch_waits = sum(1 for event in timeline if event["retrieval_source"] == "prefetch-wait")
        within_budget = sum(1 for event in timeline if event["retrieval_within_budget"])

        avg_nprobe = (
            sum(event["nprobe"] for event in timeline) / retrieval_steps if retrieval_steps else float("nan")
        )

        overlap_ratio = (prefetch_hits / retrieval_steps) if retrieval_steps else 0.0

        return {
            "answer": partial_answer,
            "latency_ms": total_ms,
            "timeline": timeline,
            "segments": retrieval_steps,
            "overlap_ratio": overlap_ratio,
            "prefetch_wait_ratio": (prefetch_waits / retrieval_steps) if retrieval_steps else 0.0,
            "retrieval_within_budget_ratio": (within_budget / retrieval_steps) if retrieval_steps else 0.0,
            "average_nprobe": avg_nprobe,
            "config": {
                "enable_s1_pipeline": cfg.enable_s1_pipeline,
                "enable_s2_flexible_interval": cfg.enable_s2_flexible_interval,
                "enable_s3_adaptive_nprobe": cfg.enable_s3_adaptive_nprobe,
                "m_prime": effective_m,
                "stale_offset_tokens": stale_offset,
                "apply_stale_shift_to_chunks": cfg.apply_stale_shift_to_chunks,
            },
        }
