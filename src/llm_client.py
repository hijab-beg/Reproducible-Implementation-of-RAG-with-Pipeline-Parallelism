import os
import json
import re
import urllib.request
import urllib.error
from dotenv import load_dotenv
from google import genai

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

load_dotenv()


class LLMClient:
    def __init__(self, model_name: str | None = None, backend: str | None = None):
        configured_backend = (backend or os.getenv("LLM_BACKEND", "")).strip().lower()
        supported_backends = {"groq", "nvidia", "gemini", "ollama"}
        if configured_backend and configured_backend not in supported_backends:
            raise ValueError("LLM_BACKEND must be one of: groq, nvidia, gemini, ollama")

        self._model_override = model_name
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
        self._groq_client = None
        self._nvidia_client = None
        self._gemini_client = None

        self._explicit_backend = bool(configured_backend)
        if self._explicit_backend:
            self.backend_priority = [configured_backend]
        else:
            # Default auto-order requested by user: Groq -> NVIDIA -> Gemini -> Ollama.
            self.backend_priority = ["groq", "nvidia", "gemini", "ollama"]

        self.backend = self.backend_priority[0]

    def _is_backend_available(self, backend: str) -> bool:
        if backend == "groq":
            return OpenAI is not None and bool(os.getenv("GROQ_API_KEY"))
        if backend == "nvidia":
            return OpenAI is not None and bool(os.getenv("NVIDIA_API_KEY"))
        if backend == "gemini":
            return bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
        if backend == "ollama":
            return True
        return False

    def _build_model_name(self, backend: str) -> str:
        if self._model_override:
            return self._model_override
        if backend == "groq":
            return os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        if backend == "nvidia":
            return os.getenv("NVIDIA_MODEL", "minimaxai/minimax-m2.7")
        if backend == "gemini":
            return os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        return os.getenv("OLLAMA_MODEL", "llama3.1:8b")

    def _ensure_backend_client(self, backend: str):
        if backend == "groq":
            if self._groq_client is not None:
                return
            if OpenAI is None:
                raise ImportError("OpenAI library required for Groq backend. Install with: pip install openai")
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise ValueError("GROQ_API_KEY not found in environment.")
            self._groq_client = OpenAI(
                base_url=os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
                api_key=api_key,
            )
            return

        if backend == "nvidia":
            if self._nvidia_client is not None:
                return
            if OpenAI is None:
                raise ImportError("OpenAI library required for NVIDIA backend. Install with: pip install openai")
            api_key = os.getenv("NVIDIA_API_KEY")
            if not api_key:
                raise ValueError("NVIDIA_API_KEY not found in environment.")
            self._nvidia_client = OpenAI(
                base_url="https://integrate.api.nvidia.com/v1",
                api_key=api_key,
            )
            return

        if backend == "gemini":
            if self._gemini_client is not None:
                return
            api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY not found in environment.")
            self._gemini_client = genai.Client(api_key=api_key)
            return

    def _generate_ollama(self, prompt: str, max_tokens: int | None = 200) -> str:
        model_name = self._build_model_name("ollama")
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
        }

        if max_tokens is not None:
            payload["options"] = {"num_predict": max_tokens}

        request = urllib.request.Request(
            url=f"{self.ollama_base_url}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"Ollama HTTP error: {exc.code} {details}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                "Could not reach Ollama at OLLAMA_BASE_URL. "
                "Make sure Ollama is running and model is pulled."
            ) from exc

        data = json.loads(body)
        text = (data.get("response") or "").strip()
        return text

    def _generate_openai_chat(self, client, model_name: str, prompt: str, max_tokens: int | None = 200) -> str:
        request_args = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "top_p": 0.95,
            "stream": False,
        }
        if max_tokens is not None:
            request_args["max_tokens"] = max_tokens

        completion = client.chat.completions.create(**request_args)
        text = (completion.choices[0].message.content or "").strip()
        return text

    def _generate_with_backend(self, backend: str, prompt: str, max_tokens: int | None = 200) -> str:
        self._ensure_backend_client(backend)

        if backend == "groq":
            return self._generate_openai_chat(
                self._groq_client,
                self._build_model_name("groq"),
                prompt,
                max_tokens=max_tokens,
            )

        if backend == "nvidia":
            return self._generate_openai_chat(
                self._nvidia_client,
                self._build_model_name("nvidia"),
                prompt,
                max_tokens=max_tokens,
            )

        if backend == "gemini":
            response = self._gemini_client.models.generate_content(
                model=self._build_model_name("gemini"),
                contents=prompt,
            )

            text = getattr(response, "text", None)
            if text:
                return text.strip()

            return str(response).strip()

        if backend == "ollama":
            return self._generate_ollama(prompt, max_tokens=max_tokens)

        raise ValueError(f"Unsupported backend: {backend}")

    def _remove_repeated_sentences(self, text: str) -> str:
        normalized = " ".join(text.split()).strip()
        if not normalized:
            return normalized

        sentence_parts = re.split(r"(?<=[.!?])\s+", normalized)
        cleaned = []
        seen = set()

        for part in sentence_parts:
            sentence = part.strip()
            if not sentence:
                continue
            if sentence in seen:
                continue
            seen.add(sentence)
            cleaned.append(sentence)

        result = " ".join(cleaned).strip()
        if result and result[-1] not in ".!?":
            result += "."
        return result

    def generate(self, prompt: str, max_tokens: int | None = None) -> str:
        errors = []

        for backend in self.backend_priority:
            if not self._is_backend_available(backend):
                if self._explicit_backend:
                    errors.append(f"{backend}: backend unavailable due to missing credentials/dependencies")
                continue

            self.backend = backend
            try:
                result = self._generate_with_backend(backend, prompt, max_tokens=max_tokens)
                return self._remove_repeated_sentences(result)
            except Exception as exc:
                errors.append(f"{backend}: {exc}")
                if self._explicit_backend:
                    break

        details = " | ".join(errors) if errors else "no backend candidates available"
        raise RuntimeError(f"All configured LLM backends failed. Details: {details}")