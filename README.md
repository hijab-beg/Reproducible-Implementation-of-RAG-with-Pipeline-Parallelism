# PipeRAG Project (Python)

This project implements a retrieval-augmented generation pipeline with FAISS retrieval and a PipeRAG-style system layer including:

- pipelined retrieval/generation overlap
- flexible retrieval interval
- adaptive retrieval depth (`nprobe`)

## Project Structure

- `README.md`
- `data/`
  - `raw_docs.json`
  - `chunks.json`
  - `faiss_index.bin`
- `src/`
  - `run.py` (single launcher)
  - `main.py` (demo script)
  - `benchmark_piperag.py`
  - `test_retriever.py`
  - `llm_client.py`
  - `faiss_retriever.py`
  - other pipeline modules

## LLM Backends

The framework supports both:

- Groq (`LLM_BACKEND=groq`)
- Ollama (`LLM_BACKEND=ollama`)
- Google AI Studio Gemini (`LLM_BACKEND=gemini`)

For Groq API:

- `GROQ_API_KEY`
- Optional: `GROQ_MODEL` (default: `llama-3.3-70b-versatile`)
- Optional: `GROQ_BASE_URL` (default: `https://api.groq.com/openai/v1`)

Note: opening `https://api.groq.com/openai/v1` in a browser uses GET and will show an unknown URL error. This is expected. The app uses POST to `/chat/completions`.

For Gemini API keys, either variable works:

- `GEMINI_API_KEY`
- `GOOGLE_API_KEY`

If backend is not set, Groq is auto-selected when `GROQ_API_KEY` exists, otherwise Gemini is selected when Gemini keys exist; otherwise Ollama is used.

## Run With Groq

Set your API key as an environment variable in PowerShell and run the single launcher.

```powershell
$env:LLM_BACKEND='groq'
$env:GROQ_API_KEY='YOUR_GROQ_API_KEY'
python src/run.py --task demo
```

Or use CLI args for one run:

```powershell
python src/run.py --task demo --backend groq --groq-api-key "YOUR_GROQ_API_KEY"
```

Optional model override:

```powershell
python src/run.py --task demo --backend groq --model llama-3.3-70b-versatile
```

## Single Main File To Run

Use one entry point for everything:

```powershell
python src/run.py --task demo
```

Available tasks:

- `demo`
- `retriever-test`
- `benchmark`
- `all` (demo then benchmark)

Examples:

```powershell
# Demo with custom query
python src/run.py --task demo --query "When was the first barbie movie released?"

# Retriever test
python src/run.py --task retriever-test --query "What is retrieval-augmented generation?"

# Benchmark
python src/run.py --task benchmark
```

## Run With Google AI Studio (Gemini)

Set your API key as an environment variable in PowerShell and run the single launcher.

```powershell
$env:LLM_BACKEND='gemini'
$env:GEMINI_API_KEY='YOUR_GOOGLE_AI_STUDIO_KEY'
python src/run.py --task demo
```

Or use the CLI argument for one run only:

```powershell
python src/run.py --task demo --backend gemini --gemini-api-key "YOUR_GOOGLE_AI_STUDIO_KEY"
```

If you hit Gemini quota limits (`429 RESOURCE_EXHAUSTED`), run with automatic Ollama fallback:

```powershell
python src/run.py --task demo --backend gemini --fallback-ollama
```

You can also enable fallback through environment variable:

```powershell
$env:GEMINI_FALLBACK_TO_OLLAMA='1'
```

Optional model override:

```powershell
python src/run.py --task demo --backend gemini --model gemini-2.5-flash
```

Gemini request throttling is enabled by default at 8 RPM (to stay below common 10 RPM limits).

You can override it:

```powershell
python src/run.py --task demo --backend gemini --max-rpm 8
```

Or with environment variable:

```powershell
$env:LLM_MAX_RPM='8'
```

## Run With Ollama

```powershell
$env:LLM_BACKEND='ollama'
$env:OLLAMA_MODEL='llama3.1:8b'
python src/run.py --task demo
```

## Data Paths

By default, the launcher expects:

- `data/chunks.json`
- `data/faiss_index.bin`

You can override them:

```powershell
python src/run.py --task demo --chunks-path "C:\path\to\chunks.json" --index-path "C:\path\to\faiss_index.bin"
```

## Security Note

Do not commit API keys to the repository or `.env`.
