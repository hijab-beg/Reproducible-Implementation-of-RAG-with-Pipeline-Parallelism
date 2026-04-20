from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor


class PipelineController:
    """Handles one-step-ahead retrieval prefetch for pipeline overlap (S1)."""

    def __init__(self, retriever, max_workers: int = 1):
        self.retriever = retriever
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.prefetch_futures: dict[int, Future] = {}

    def schedule_prefetch(self, step: int, query: str, top_k: int, nprobe: int):
        if step in self.prefetch_futures:
            return

        future = self.executor.submit(
            self.retriever.retrieve,
            query,
            top_k,
            nprobe,
            True,
        )
        self.prefetch_futures[step] = future

    def consume_prefetched_with_status(self, step: int):
        """Consume prefetched retrieval and indicate whether caller had to wait."""
        future = self.prefetch_futures.pop(step, None)
        if future is None:
            return None, "none"

        if future.done():
            return future.result(), "prefetch-hit"

        # Not done yet: we still use it, but overlap was incomplete.
        return future.result(), "prefetch-wait"

    def shutdown(self):
        self.executor.shutdown(wait=True)
