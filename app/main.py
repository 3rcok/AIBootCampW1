import json
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from openai import OpenAI
from pydantic import BaseModel, Field

from app.config import settings

app = FastAPI(title="Text API", version="1.0.0")
client = OpenAI(api_key=settings.openai_api_key)


class HealthResponse(BaseModel):
    status: str
    timestamp: str


class SummarizeRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text to summarize")
    max_length: int = Field(..., ge=10, le=2000, description="Approximate max length of summary in words")


class SummarizeResponse(BaseModel):
    summary: str


class SentimentRequest(BaseModel):
    text: str = Field(..., min_length=1)


class SentimentResponse(BaseModel):
    sentiment: str = Field(..., pattern="^(positive|negative|neutral)$")
    confidence: float = Field(..., ge=0.0, le=1.0)
    explanation: str


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.post("/summarize", response_model=SummarizeResponse)
def summarize(body: SummarizeRequest) -> SummarizeResponse:
    try:
        completion = client.chat.completions.create(
            model=settings.openai_model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a concise summarizer. Produce a clear summary that respects the requested maximum length.",
                },
                {
                    "role": "user",
                    "content": f"Summarize the following text in at most approximately {body.max_length} words:\n\n{body.text}",
                },
            ],
            temperature=0.3,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"OpenAI request failed: {e!s}") from e

    summary = (completion.choices[0].message.content or "").strip()
    if not summary:
        raise HTTPException(status_code=502, detail="Empty summary from model")
    return SummarizeResponse(summary=summary)


@app.post("/analyze-sentiment", response_model=SentimentResponse)
def analyze_sentiment(body: SentimentRequest) -> SentimentResponse:
    try:
        completion = client.chat.completions.create(
            model=settings.openai_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You analyze sentiment. Respond with JSON only, no markdown, with keys: "
                        'sentiment (exactly one of: positive, negative, neutral), '
                        "confidence (number from 0 to 1), explanation (short string)."
                    ),
                },
                {
                    "role": "user",
                    "content": body.text,
                },
            ],
            temperature=0.2,
            response_format={"type": "json_object"},
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"OpenAI request failed: {e!s}") from e

    raw = (completion.choices[0].message.content or "").strip()
    if not raw:
        raise HTTPException(status_code=502, detail="Empty response from model")

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=502, detail=f"Invalid JSON from model: {e!s}") from e

    try:
        return SentimentResponse(
            sentiment=str(data["sentiment"]).lower(),
            confidence=float(data["confidence"]),
            explanation=str(data["explanation"]),
        )
    except (KeyError, TypeError, ValueError) as e:
        raise HTTPException(status_code=502, detail=f"Unexpected sentiment payload: {e!s}") from e


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=settings.port, reload=True)
