# AIBootCampW1 — Text API


A small [FastAPI](https://fastapi.tiangolo.com/) service that exposes text summarization and sentiment analysis using the OpenAI API.

## Requirements

- Python 3.10+
- An [OpenAI API key](https://platform.openai.com/api-keys)

## Setup

1. Create and activate a virtual environment (recommended):

   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Copy `.env.example` to `.env` and set your key (and optional overrides):

   ```env
   OPENAI_API_KEY=sk-your-key-here
   OPENAI_MODEL=gpt-4o
   PORT=8111
   ```

## Run

```bash
python -m app.main
```

Or with uvicorn directly:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8111 --reload
```

The API listens on `http://127.0.0.1:8111` by default (or the port in `.env`).

Interactive docs: [http://127.0.0.1:8111/docs](http://127.0.0.1:8111/docs) (Swagger UI).

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness check with UTC timestamp |
| `POST` | `/summarize` | Summarize text with an approximate max length (words) |
| `POST` | `/analyze-sentiment` | Sentiment (`positive` / `negative` / `neutral`), confidence, short explanation |

Request and response bodies are JSON; see `/docs` for schemas and try-it-out requests.

## Project layout

- `app/main.py` — FastAPI app and routes
- `app/config.py` — Settings loaded from `.env`
- `requirements.txt` — Python dependencies
