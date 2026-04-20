import os
import json
import time
import urllib.request
import urllib.error
from collections import deque
from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors

load_dotenv()


class LLMClient:
    def __init__(self, model_name: str | None = None, backend: str | None = None):
        configured_backend = (backend or os.getenv("LLM_BACKEND", "")).strip().lower()
        gemini_api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        groq_api_key = os.getenv("GROQ_API_KEY")
        self.enable_gemini_fallback_to_ollama = os.getenv("GEMINI_FALLBACK_TO_OLLAMA", "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
        self.ollama_model = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
        self.groq_base_url = self._normalize_groq_base_url(
            os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
        )
        self.groq_api_key = groq_api_key
        self.groq_model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

        if configured_backend:
            self.backend = configured_backend
        elif groq_api_key:
            self.backend = "groq"
        elif gemini_api_key:
            self.backend = "gemini"
        else:
            self.backend = "ollama"

        if self.backend not in {"gemini", "ollama", "groq"}:
            raise ValueError("LLM_BACKEND must be one of: gemini, ollama, groq")

        if self.backend == "gemini":
            api_key = gemini_api_key
            if not api_key:
                raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY not found in environment.")
            self.client = genai.Client(api_key=api_key)
            self.model_name = model_name or "gemini-2.5-flash"
        elif self.backend == "groq":
            self.client = None
            if not self.groq_api_key:
                raise ValueError("GROQ_API_KEY not found in environment.")
            self.model_name = model_name or self.groq_model
        else:
            self.client = None
            self.model_name = model_name or os.getenv("OLLAMA_MODEL", "llama3.1:8b")
            self.ollama_model = self.model_name

        backend_rpm_env = "GEMINI_MAX_RPM" if self.backend == "gemini" else "GROQ_MAX_RPM"
        rpm_raw = os.getenv("LLM_MAX_RPM") or os.getenv(backend_rpm_env)
        if rpm_raw:
            try:
                self.max_rpm = max(1, int(rpm_raw))
            except ValueError as exc:
                raise ValueError("LLM_MAX_RPM and backend-specific RPM values must be integers.") from exc
        elif self.backend in {"gemini", "groq"}:
            # Keep a safety buffer below common 10 RPM limits.
            self.max_rpm = 8
        else:
            self.max_rpm = 0

        self._request_times = deque()

    @staticmethod
    def _normalize_groq_base_url(raw_url: str) -> str:
        url = (raw_url or "https://api.groq.com/openai/v1").strip().rstrip("/")

        if url.endswith("/chat/completions"):
            return url[: -len("/chat/completions")]

        if url.endswith("/openai/v1"):
            return url

        if url == "https://api.groq.com" or url.endswith("api.groq.com"):
            return f"{url}/openai/v1"

        return url

    def _apply_rate_limit(self) -> None:
        if self.backend not in {"gemini", "groq"} or self.max_rpm <= 0:
            return

        now = time.monotonic()
        window_seconds = 60.0

        while self._request_times and now - self._request_times[0] >= window_seconds:
            self._request_times.popleft()

        if len(self._request_times) >= self.max_rpm:
            sleep_for = window_seconds - (now - self._request_times[0]) + 0.05
            if sleep_for > 0:
                time.sleep(sleep_for)

            now = time.monotonic()
            while self._request_times and now - self._request_times[0] >= window_seconds:
                self._request_times.popleft()

        self._request_times.append(time.monotonic())

    def _generate_ollama(self, prompt: str, max_tokens: int = 200, model_name: str | None = None) -> str:
        payload = {
            "model": model_name or self.ollama_model,
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

    def _generate_groq(self, prompt: str, max_tokens: int = 200) -> str:
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.2,
        }

        request = urllib.request.Request(
            url=f"{self.groq_base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "NLP_PROJECT/1.0",
                "Authorization": f"Bearer {self.groq_api_key}",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="ignore")
            if exc.code == 429:
                raise RuntimeError(
                    "Groq quota/rate limit reached (HTTP 429). "
                    "Lower request rate or check Groq limits for your plan/model."
                ) from exc
            if exc.code in {401, 403}:
                raise RuntimeError(
                    "Groq request was rejected (HTTP 401/403). "
                    "Check GROQ_API_KEY, GROQ_MODEL access, and GROQ_BASE_URL. "
                    "If you pasted a temporary key into .env, replace it with a valid Groq API key."
                ) from exc
            raise RuntimeError(f"Groq HTTP error: {exc.code} {details}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError("Could not reach Groq API endpoint.") from exc

        data = json.loads(body)
        choices = data.get("choices") or []
        if not choices:
            return ""

        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, str):
            return content.strip()

        if isinstance(content, list):
            text_parts = [part.get("text", "") for part in content if isinstance(part, dict)]
            return "".join(text_parts).strip()

        return str(content or "").strip()

    def generate(self, prompt: str, max_tokens: int = 200) -> str:
        if self.backend == "ollama":
            return self._generate_ollama(prompt, max_tokens=max_tokens)

        if self.backend == "groq":
            self._apply_rate_limit()
            return self._generate_groq(prompt, max_tokens=max_tokens)

        self._apply_rate_limit()

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
            )
        except genai_errors.ClientError as exc:
            if exc.code == 429 and self.enable_gemini_fallback_to_ollama:
                return self._generate_ollama(
                    prompt,
                    max_tokens=max_tokens,
                    model_name=self.ollama_model,
                )
            if exc.code == 429:
                raise RuntimeError(
                    "Gemini quota exhausted (HTTP 429 RESOURCE_EXHAUSTED). "
                    "Enable billing/quota in Google AI Studio or rerun with Ollama. "
                    "Tip: set GEMINI_FALLBACK_TO_OLLAMA=1 to auto-fallback when quota is hit."
                ) from exc
            raise

        text = getattr(response, "text", None)
        if text:
            return text.strip()

        return str(response).strip()