"""
GridWise Energy Optimizer - LLM-based operator note interpreter.

This module calls an OpenAI-compatible chat completions API and asks the
model to turn ONE natural-language operator note into a strict JSON object
matching the official DirectiveInterpretation shape.

CRITICAL DESIGN RULES
---------------------
1. NO hard-coded phrase matching here. The LLM is the interpretation path.
2. The LLM NEVER calls the optimizer directly. Its raw text output is
   parsed as JSON and handed to `app.guardrails.py`, which deterministically
   decides whether the result is acceptable.
3. API keys are read from environment variables - never hard-coded.

ENVIRONMENT VARIABLES
---------------------
LLM_PROVIDER         e.g. "openai", "groq", "openrouter", "google" ...
LLM_API_KEY          secret token for the chosen provider (required)
LLM_MODEL            model id, e.g. "gpt-4o-mini"
LLM_TIMEOUT_SECONDS  optional, defaults to 30
LLM_BASE_URL         optional override; defaults per provider
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import httpx

from app.schemas import OptimizeRequest


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class LLMConfigError(RuntimeError):
    """Missing or invalid LLM configuration (e.g. API key not set)."""


class LLMTimeoutError(RuntimeError):
    """The LLM provider did not respond before the timeout."""


class LLMAPIError(RuntimeError):
    """The LLM provider returned an HTTP error or unexpected payload."""


class LLMJSONError(RuntimeError):
    """The LLM did not return parseable JSON."""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Known provider -> base URL mapping for OpenAI-compatible chat completions.
# Override with LLM_BASE_URL for any provider not listed here.
_PROVIDER_BASE_URLS: dict[str, str] = {
    "openai": "https://api.openai.com/v1",
    "groq": "https://api.groq.com/openai/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "together": "https://api.together.xyz/v1",
    "mistral": "https://api.mistral.ai/v1",
    "google": "https://generativelanguage.googleapis.com/v1beta/openai",
}


@dataclass(frozen=True)
class LLMConfig:
    """Resolved LLM configuration read from the environment."""

    provider: str
    api_key: str
    model: str
    timeout_seconds: float
    base_url: str

    @classmethod
    def from_env(cls) -> "LLMConfig":
        provider = (os.getenv("LLM_PROVIDER") or "").strip()
        api_key = (os.getenv("LLM_API_KEY") or "").strip()
        model = (os.getenv("LLM_MODEL") or "").strip()
        timeout_raw = (os.getenv("LLM_TIMEOUT_SECONDS") or "30").strip()

        if not provider:
            raise LLMConfigError("LLM_PROVIDER is not set in the environment.")
        if not api_key:
            raise LLMConfigError("LLM_API_KEY is not set in the environment.")
        if not model:
            raise LLMConfigError("LLM_MODEL is not set in the environment.")

        try:
            timeout = float(timeout_raw)
        except ValueError as exc:
            raise LLMConfigError(
                f"LLM_TIMEOUT_SECONDS must be a number, got {timeout_raw!r}"
            ) from exc
        if timeout <= 0:
            raise LLMConfigError("LLM_TIMEOUT_SECONDS must be > 0.")

        base_url = (os.getenv("LLM_BASE_URL") or "").strip() or _PROVIDER_BASE_URLS.get(
            provider.lower(), ""
        )
        if not base_url:
            raise LLMConfigError(
                f"Unknown LLM_PROVIDER {provider!r}. Set LLM_BASE_URL explicitly."
            )

        return cls(
            provider=provider,
            api_key=api_key,
            model=model,
            timeout_seconds=timeout,
            base_url=base_url.rstrip("/"),
        )


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are the OPERATOR NOTE INTERPRETER for a grid energy optimizer.

Your ONLY job is to read one short operator note (a natural-language
instruction) and return ONE JSON object that conforms EXACTLY to the
DirectiveInterpretation schema described below.

You MUST NOT:
- call any tools,
- return any prose,
- invent new directive types,
- guess missing numbers.

SCHEMA (return EXACTLY this JSON object, no extra keys, no comments):

{
  "applies": boolean,
  "directive_type": string,
  "structured_adjustment": object | null,
  "explanation": string   // ONE short sentence describing what you did
}

SUPPORTED directive_type VALUES (exactly these six strings):
  - "solar_reduction"
  - "minimum_battery_reserve"
  - "no_charge_window"
  - "no_discharge_window"
  - "max_grid_window"
  - "no_op"

TIME / HOUR RULES
-----------------
- Hours are integers in the CLOSED range 0..23, where 0 = 12:00 AM
  (midnight) and 23 = 11:00 PM.
- "Morning" typically maps to hours 5..11, "afternoon" to 12..16,
  "evening" to 17..20, "night" to 21..23.
- For "overnight" use the union [22, 23, 0, 1, 2, 3, 4, 5] - emit these
  hours explicitly in the "hours" list, sorted ascending, no duplicates.

DIRECTIVE SHAPES
----------------
1. If the note is irrelevant or you cannot map it to a supported directive,
   return:
     {
       "applies": false,
       "directive_type": "no_op",
       "structured_adjustment": null,
       "explanation": "<short reason>"
     }

2. If the note IS relevant, return one of these shapes:

   - solar_reduction:
       {
         "applies": true,
         "directive_type": "solar_reduction",
         "structured_adjustment": {
           "hours": [int, ...],   // unique ints in 0..23, sorted ascending
           "factor": 0.0..1.0     // usable fraction remaining
                                  // (e.g. "cut solar use by 30%" -> 0.7)
         },
         "explanation": "<short>"
       }

   - minimum_battery_reserve:
       {
         "applies": true,
         "directive_type": "minimum_battery_reserve",
         "structured_adjustment": {
           "hours": [int, ...],
           "minimum_energy_kwh": number >= 0
         },
         "explanation": "<short>"
       }

   - no_charge_window:
       {
         "applies": true,
         "directive_type": "no_charge_window",
         "structured_adjustment": {
           "hours": [int, ...]   // hours during which battery may NOT charge
         },
         "explanation": "<short>"
       }

   - no_discharge_window:
       {
         "applies": true,
         "directive_type": "no_discharge_window",
         "structured_adjustment": {
           "hours": [int, ...]   // hours during which battery may NOT discharge
         },
         "explanation": "<short>"
       }

   - max_grid_window:
       {
         "applies": true,
         "directive_type": "max_grid_window",
         "structured_adjustment": {
           "hours": [int, ...],
           "max_grid_kwh": number >= 0
         },
         "explanation": "<short>"
       }

Output a single JSON object. No markdown, no code fences, no prose outside
the JSON.
"""


def _build_user_prompt(note: str, request: OptimizeRequest) -> str:
    """Compose the user-role prompt for a single operator note."""
    n_notes = len(request.operator_notes)
    note_idx = next(
        (i for i, n in enumerate(request.operator_notes) if n == note),
        -1,
    )
    return (
        f"Operator note index: {note_idx} of {n_notes - 1}\n"
        f"Operator note text:\n\"{note.strip()}\"\n\n"
        "Respond with ONLY the JSON object."
    )


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class LLMClient:
    """
    Thin OpenAI-compatible chat completions client.

    The LLM is the interpretation path - this class only calls the API,
    parses the JSON text out of the response, and lets guardrails validate
    it. It performs NO phrase matching of its own.
    """

    def __init__(self, config: LLMConfig | None = None) -> None:
        self._config = config or LLMConfig.from_env()

    @property
    def config(self) -> LLMConfig:
        return self._config

    # ------------------------------------------------------------------ public

    def interpret_notes(
        self, request: OptimizeRequest
    ) -> list[dict[str, Any]]:
        """
        Interpret every operator note in `request` independently.

        Returns a list of raw dicts in the SAME ORDER as request.operator_notes.
        Each dict is exactly what the LLM produced (after JSON parsing).
        Downstream code MUST validate these via app.guardrails.py before use.
        """
        return [self._interpret_one(note, request) for note in request.operator_notes]

    # ----------------------------------------------------------------- private

    def _interpret_one(self, note: str, request: OptimizeRequest) -> dict[str, Any]:
        payload = {
            "model": self._config.model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": _build_user_prompt(note, request)},
            ],
            # Force structured JSON output where supported.
            "response_format": {"type": "json_object"},
            "temperature": 0,
        }
        headers = {
            "Authorization": f"Bearer {self._config.api_key}",
            "Content-Type": "application/json",
        }
        url = f"{self._config.base_url}/chat/completions"

        try:
            response = httpx.post(
                url,
                json=payload,
                headers=headers,
                timeout=self._config.timeout_seconds,
            )
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError(
                f"LLM request timed out after {self._config.timeout_seconds}s"
            ) from exc
        except httpx.HTTPError as exc:
            raise LLMAPIError(f"LLM HTTP transport error: {exc}") from exc

        if response.status_code >= 400:
            raise LLMAPIError(
                f"LLM API error {response.status_code}: {_safe_truncate(response.text)}"
            )

        try:
            data = response.json()
        except json.JSONDecodeError as exc:
            raise LLMJSONError(
                f"LLM returned non-JSON response: {_safe_truncate(response.text)}"
            ) from exc

        return _extract_message_json(data)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_truncate(text: str, limit: int = 500) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit] + "...<truncated>"


def _extract_message_json(data: dict[str, Any]) -> dict[str, Any]:
    """Pull the assistant message content out of an OpenAI-style response."""
    try:
        choices = data["choices"]
        message = choices[0]["message"]
        content = message["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMJSONError(f"Unexpected LLM response shape: {data!r}") from exc

    if not isinstance(content, str):
        raise LLMJSONError("LLM message content is not a string.")

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise LLMJSONError(
            f"LLM did not return valid JSON: {_safe_truncate(content)}"
        ) from exc

    if not isinstance(parsed, dict):
        raise LLMJSONError(
            f"LLM JSON must be an object, got {type(parsed).__name__}."
        )

    return parsed
