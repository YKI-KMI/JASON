"""Shared agent plumbing: one JSON-repair retry, offline-mode labeling."""

from __future__ import annotations

import json
import re
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from core import model as model_client

T = TypeVar("T", bound=BaseModel)

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def ask_structured(prompt: str, schema: type[T], system: str | None = None) -> T:
    """Ask Claude for JSON conforming to `schema`; parse + validate.

    One bounded repair pass: if the model's JSON is malformed or fails
    validation, the validation error is fed back once. Still failing ->
    raise ValueError (orchestrator surfaces it; nothing silent).
    In offline mode (no key), raise ModelUnavailableError — callers decide
    whether to use their deterministic fallback.
    """
    raw = model_client.complete_json(prompt, system=system)
    return parse_structured(raw, schema)


def parse_structured(raw: str, schema: type[T]) -> T:
    data, err = _try_parse(raw, schema)
    if data is not None:
        return data
    # One repair pass with the validation error fed back.
    repair_prompt = (
        f"The following was supposed to be JSON matching a schema but failed:\n\n{raw[:4000]}\n\n"
        f"Error:\n{err}\n\nReturn the corrected JSON object only."
    )
    raw2 = model_client.complete_json(repair_prompt)
    data2, err2 = _try_parse(raw2, schema)
    if data2 is not None:
        return data2
    raise ValueError(f"model output did not match {schema.__name__}: {err2}")


def _try_parse(raw: str, schema: type[T]) -> tuple[T | None, str | None]:
    try:
        return schema.model_validate_json(raw), None
    except ValidationError as exc:
        candidate = raw
    except Exception as exc:  # noqa: BLE001 - any parse issue -> try extraction
        candidate = raw
        err = str(exc)
    m = _JSON_BLOCK_RE.search(raw)
    if m:
        candidate = m.group(0)
    try:
        data = json.loads(candidate)
        return schema.model_validate(data), None
    except (json.JSONDecodeError, ValidationError) as exc:
        return None, str(exc)
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)


def mode_label() -> str:
    return "claude" if model_client.available() else "offline_heuristic"
