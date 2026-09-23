"""
LLM client plumbing shared by the AI helpers.

Every AI entry point takes ``client=`` — any object with the OpenAI chat
completions interface (``client.chat.completions.create(...)``). That
covers OpenAI, Anthropic's OpenAI-compatible endpoint, and local servers
(vLLM, Ollama, LM Studio) via ``base_url``. Without a client, one is built
from the openai package.

Defaults come from the environment:
    AGENTSTAN_MODEL     model name (default "gpt-5.5")
    AGENTSTAN_BASE_URL  API base URL (falls back to OPENAI_BASE_URL)
    OPENAI_API_KEY      read by the openai package itself
"""

import os
from typing import Optional

FALLBACK_MODEL = "gpt-5.5"


def default_model() -> str:
    return os.environ.get("AGENTSTAN_MODEL") or FALLBACK_MODEL


def resolve_model(model: Optional[str]) -> str:
    return model or default_model()


def make_client(api_key: Optional[str] = None, base_url: Optional[str] = None):
    """Build an OpenAI-compatible client (requires ``pip install agentstan[ai]``)."""
    try:
        from openai import OpenAI
    except ImportError as e:
        raise ImportError(
            "the AI helpers need the openai package: pip install 'agentstan[ai]' "
            "(or pass client= with any OpenAI-compatible client)"
        ) from e
    kwargs = {}
    if api_key:
        kwargs["api_key"] = api_key
    base_url = base_url or os.environ.get("AGENTSTAN_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def chat_json(client, model: str, messages: list) -> str:
    """One JSON-mode chat completion; returns the message text."""
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content
