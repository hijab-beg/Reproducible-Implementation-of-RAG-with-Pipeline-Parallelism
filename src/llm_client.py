import os
import json
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

        if configured_backend:
            self.backend = configured_backend
        elif os.getenv("NVIDIA_API_KEY"):
            self.backend = "nvidia"
        elif os.getenv("GEMINI_API_KEY"):
            self.backend = "gemini"
        else:
            self.backend = "ollama"

        if self.backend not in {"gemini", "ollama", "nvidia"}:
            raise ValueError("LLM_BACKEND must be one of: gemini, ollama, nvidia")

        if self.backend == "nvidia":
            if OpenAI is None:
                raise ImportError("OpenAI library required for NVIDIA backend. Install with: pip install openai")
            api_key = os.getenv("NVIDIA_API_KEY")
            if not api_key:
                raise ValueError("NVIDIA_API_KEY not found in environment.")
            self.client = OpenAI(
                base_url="https://integrate.api.nvidia.com/v1",
                api_key=api_key
            )
            self.model_name = model_name or os.getenv("NVIDIA_MODEL", "minimaxai/minimax-m2.7")
        elif self.backend == "gemini":
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY not found in environment.")
            self.client = genai.Client(api_key=api_key)
            self.model_name = model_name or "gemini-3-flash-preview"
        else:
            self.client = None
            self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
            self.model_name = model_name or os.getenv("OLLAMA_MODEL", "llama3.1:8b")

    def _generate_ollama(self, prompt: str, max_tokens: int = 200) -> str:
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
            },
        }

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

    def _generate_nvidia(self, prompt: str, max_tokens: int = 200) -> str:
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=1.0,
            top_p=0.95,
            max_tokens=max_tokens,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        return text

    def generate(self, prompt: str, max_tokens: int = 200) -> str:
        if self.backend == "ollama":
            return self._generate_ollama(prompt, max_tokens=max_tokens)
        elif self.backend == "nvidia":
            return self._generate_nvidia(prompt, max_tokens=max_tokens)

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
        )

        text = getattr(response, "text", None)
        if text:
            return text.strip()

        return str(response).strip()