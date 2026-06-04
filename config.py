import os

CHAT_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"
TRANSCRIPTION_MODEL = "gpt-4o-mini-transcribe"
VECTORSTORE_PROVIDER = "openai"


def get_openai_api_key():
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if api_key:
        return api_key
    raise RuntimeError(
        "Missing OpenAI API key. Add OPENAI_API_KEY=your_key to .env."
    )


def format_openai_error(exc):
    error_text = str(exc)
    lowered = error_text.lower()

    if "insufficient_quota" in lowered or "quota" in lowered:
        return "OpenAI API quota is unavailable. Please check billing/credits for this API key."
    if "invalid_api_key" in lowered or "api key" in lowered or "authentication" in lowered:
        return "The OpenAI API key is missing or invalid. Please update OPENAI_API_KEY in .env."
    if "resolving" in lowered or "dns" in lowered or "connection" in lowered:
        return "I could not reach OpenAI right now. Please check your internet connection and try again."
    return f"Something went wrong: {exc}"
