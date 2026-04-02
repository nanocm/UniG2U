from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Optional

from openai import OpenAI


DEFAULT_ENDPOINTS_FILE = Path(os.getenv("AZURE_OPENAI_ENDPOINT_FILE", "/home/aiscuser/endpoint.json"))
DEFAULT_RESOURCE = os.getenv("AZURE_OPENAI_RESOURCE", "https://cognitiveservices.azure.com/")
TOKEN_CACHE: dict[str, float | str | None] = {
    "access_token": None,
    "expires_at": 0,
}


def _normalize_base_url(base_url: str) -> str:
    return base_url.strip().rstrip("/")


def load_endpoint_lines(path: Path = DEFAULT_ENDPOINTS_FILE) -> list[str]:
    if not path.exists():
        return []
    return [
        _normalize_base_url(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def select_base_url() -> Optional[str]:
    explicit = os.getenv("OPENAI_BASE_URL") or os.getenv("AZURE_OPENAI_COMPAT_BASE_URL")
    if explicit:
        return _normalize_base_url(explicit)

    endpoints = load_endpoint_lines()
    if not endpoints:
        return None

    try:
        index = int(os.getenv("AZURE_OPENAI_ENDPOINT_INDEX", "0"))
    except ValueError:
        index = 0
    return endpoints[index % len(endpoints)]


def has_endpoint_support() -> bool:
    return bool(select_base_url())


def get_model_name() -> str:
    return (
        os.getenv("OPENAI_JUDGE_MODEL")
        or os.getenv("AZURE_OPENAI_COMPAT_MODEL")
        or "gpt-4.1"
    )


def get_bearer_token(resource: str = DEFAULT_RESOURCE) -> str:
    now = time.time()
    cached_token = TOKEN_CACHE.get("access_token")
    expires_at = float(TOKEN_CACHE.get("expires_at", 0) or 0)
    if cached_token and now < expires_at - 300:
        return str(cached_token)

    command = [
        "az",
        "account",
        "get-access-token",
        "--resource",
        resource,
        "-o",
        "json",
    ]
    payload = json.loads(subprocess.check_output(command, text=True))
    access_token = payload["accessToken"]
    expires_at = float(payload.get("expires_on") or payload.get("expiresOn") or 0)
    TOKEN_CACHE["access_token"] = access_token
    TOKEN_CACHE["expires_at"] = expires_at
    return access_token


def build_client(base_url: Optional[str] = None, model: Optional[str] = None) -> tuple[OpenAI, str]:
    resolved_base_url = base_url or select_base_url()
    if not resolved_base_url:
        raise ValueError("No Azure OpenAI compatible endpoint available.")
    resolved_model = model or get_model_name()
    client = OpenAI(
        api_key=get_bearer_token(),
        base_url=resolved_base_url,
    )
    return client, resolved_model
