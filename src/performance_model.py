from __future__ import annotations

from dataclasses import dataclass
from statistics import mean


@dataclass
class RetrievalLatencyModel:
    """Linear latency model: T_ret(nprobe) = a * nprobe + b."""

    slope: float
    intercept: float
    min_nprobe: int = 1
    max_nprobe: int = 64

    def predict_ms(self, nprobe: int) -> float:
        return self.slope * float(nprobe) + self.intercept

    def pick_nprobe(self, budget_ms: float) -> int:
        if self.slope <= 0:
            return self.min_nprobe
        raw = int((budget_ms - self.intercept) / self.slope)
        return max(self.min_nprobe, min(self.max_nprobe, raw))


@dataclass
class GenerationLatencyEMA:
    """Tracks generation latency and provides a stable per-step budget."""

    initial_ms: float = 1200.0
    alpha: float = 0.25
    floor_ms: float = 150.0

    def __post_init__(self):
        self.current_ms = float(self.initial_ms)

    def observe(self, latency_ms: float):
        value = max(self.floor_ms, float(latency_ms))
        self.current_ms = self.alpha * value + (1.0 - self.alpha) * self.current_ms

    def budget(self, safety_factor: float = 0.9) -> float:
        return max(self.floor_ms, self.current_ms * float(safety_factor))


class PerformanceModelBuilder:
    """Profiles retriever latency and fits a simple linear nprobe model."""

    @staticmethod
    def profile_retrieval(
        retriever,
        queries: list[str],
        nprobe_values: list[int],
        top_k: int,
    ) -> dict[int, float]:
        if not queries:
            raise ValueError("queries must not be empty")

        measurements: dict[int, list[float]] = {n: [] for n in nprobe_values}

        for nprobe in nprobe_values:
            for query in queries:
                _, meta = retriever.retrieve(
                    queryText=query,
                    k=top_k,
                    nProbe=nprobe,
                    return_metadata=True,
                )
                measurements[nprobe].append(float(meta["latency_ms"]))

        return {n: mean(values) for n, values in measurements.items() if values}

    @staticmethod
    def fit_linear_model(
        latency_by_nprobe: dict[int, float],
        min_nprobe: int = 1,
        max_nprobe: int = 64,
    ) -> RetrievalLatencyModel:
        if len(latency_by_nprobe) < 2:
            # Conservative fallback if profiling points are not enough.
            return RetrievalLatencyModel(
                slope=0.0,
                intercept=latency_by_nprobe[next(iter(latency_by_nprobe))] if latency_by_nprobe else 0.0,
                min_nprobe=min_nprobe,
                max_nprobe=max_nprobe,
            )

        points = sorted((float(k), float(v)) for k, v in latency_by_nprobe.items())
        x_mean = mean(p[0] for p in points)
        y_mean = mean(p[1] for p in points)

        denom = sum((x - x_mean) ** 2 for x, _ in points)
        if denom == 0:
            slope = 0.0
        else:
            slope = sum((x - x_mean) * (y - y_mean) for x, y in points) / denom

        intercept = y_mean - slope * x_mean
        return RetrievalLatencyModel(
            slope=slope,
            intercept=intercept,
            min_nprobe=min_nprobe,
            max_nprobe=max_nprobe,
        )
