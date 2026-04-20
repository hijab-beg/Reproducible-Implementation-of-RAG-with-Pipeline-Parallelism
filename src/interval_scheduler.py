from dataclasses import dataclass


@dataclass
class IntervalScheduler:
    """Schedules retrieval triggers for flexible intervals m -> m'."""

    m_prime: int = 32
    stale_offset_tokens: int | None = None

    def __post_init__(self):
        if self.m_prime <= 0:
            raise ValueError("m_prime must be positive")
        if self.stale_offset_tokens is None:
            # PipeRAG default policy: stale offset follows retrieval interval.
            self.stale_offset_tokens = self.m_prime
        if self.stale_offset_tokens < 0:
            raise ValueError("stale_offset_tokens must be >= 0")

    def should_retrieve(self, step_number: int) -> bool:
        if step_number < 1:
            return False
        return (step_number - 1) % self.m_prime == 0

    def next_retrieval_step(self, current_step: int) -> int:
        if current_step < 1:
            return 1
        remainder = (current_step - 1) % self.m_prime
        if remainder == 0:
            return current_step
        return current_step + (self.m_prime - remainder)

    def schedule(self, max_steps: int) -> list[int]:
        if max_steps < 1:
            return []
        return [step for step in range(1, max_steps + 1) if self.should_retrieve(step)]
