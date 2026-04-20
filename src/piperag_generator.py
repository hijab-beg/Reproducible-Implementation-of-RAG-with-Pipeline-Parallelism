from prompt_builder import build_augmented_prompt

from performance_model import PerformanceModelBuilder, RetrievalLatencyModel


class PipeRAGGenerator:
    def __init__(self, retriever, llm_client, top_k: int = 3, retrieval_interval: int = 1):
        self.retriever = retriever
        self.llm_client = llm_client
        self.top_k = top_k
        self.retrieval_interval = retrieval_interval

    def get_stale_query_window(self, generated_text: str, window_words: int = 40, stale_offset_words: int = 10) -> str:
        words = generated_text.split()

        if len(words) <= stale_offset_words:
            return generated_text.strip()

        end_index = max(0, len(words) - stale_offset_words)
        start_index = max(0, end_index - window_words)

        return " ".join(words[start_index:end_index]).strip()

    def should_retrieve(self, step_number: int) -> bool:
        return step_number % self.retrieval_interval == 0

    def initial_generation(self, user_query: str):
        retrieved_chunks = self.retriever.search(user_query, top_k=self.top_k)
        prompt = build_augmented_prompt(user_query, [chunk["text"] for chunk in retrieved_chunks])
        answer = self.llm_client.generate(prompt, max_tokens=180)

        return {
            "step": 1,
            "retrieved_chunks": retrieved_chunks,
            "answer": answer
        }

    def continue_generation_with_stale_retrieval(self, user_query: str, partial_answer: str, step_number: int):
        if self.should_retrieve(step_number):
            stale_query = self.get_stale_query_window(partial_answer)
            if not stale_query:
                stale_query = user_query
            retrieved_chunks = self.retriever.search(stale_query, top_k=self.top_k)
        else:
            stale_query = None
            retrieved_chunks = []

        prompt = build_augmented_prompt(
            user_query=user_query,
            retrieved_chunks=[chunk["text"] for chunk in retrieved_chunks],
            partial_answer=partial_answer
        )
        continuation = self.llm_client.generate(prompt, max_tokens=180)

        return {
            "step": step_number,
            "stale_query": stale_query,
            "retrieved_chunks": retrieved_chunks,
            "continuation": continuation
        }

    def build_adaptive_model(
        self,
        sample_queries: list[str],
        nprobe_values: list[int],
        min_nprobe: int = 1,
        max_nprobe: int = 64,
    ) -> RetrievalLatencyModel:
        latencies = PerformanceModelBuilder.profile_retrieval(
            retriever=self.retriever,
            queries=sample_queries,
            nprobe_values=nprobe_values,
            top_k=self.top_k,
        )
        return PerformanceModelBuilder.fit_linear_model(
            latency_by_nprobe=latencies,
            min_nprobe=min_nprobe,
            max_nprobe=max_nprobe,
        )