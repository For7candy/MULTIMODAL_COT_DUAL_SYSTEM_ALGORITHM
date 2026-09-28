"""Structured-output wrapper for Open-Nav's language-model client."""

import json
from typing import Any, Callable, Dict, Optional


def _extract_json_object(text: str) -> Dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    payload = json.loads(stripped)
    if not isinstance(payload, dict):
        raise ValueError("structured backbone output must be a JSON object")
    return payload


class OpenNavBackboneAdapter:
    def __init__(
        self,
        model_type: str,
        api_key: Optional[str] = None,
        client: Any = None,
        client_factory: Optional[Callable[[], Any]] = None,
    ) -> None:
        if not model_type or model_type == "UNCONFIGURED":
            raise ValueError("a configured model_type is required")
        self.model_type = model_type
        self.api_key = api_key
        self._client = client
        self._client_factory = client_factory

    @property
    def is_loaded(self) -> bool:
        return self._client is not None

    def _load(self) -> Any:
        if self._client is not None:
            return self._client
        if self._client_factory is not None:
            self._client = self._client_factory()
            return self._client

        from vlnce_baselines.common.navigator.api import llmClient

        self._client = llmClient(self.model_type, self.api_key)
        return self._client

    def generate_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        schema_name: str,
    ) -> Dict[str, Any]:
        if not schema_name:
            raise ValueError("schema_name must not be empty")
        schema_instruction = (
            "\nReturn exactly one JSON object conforming to schema '{}'. "
            "Do not wrap it in prose.".format(schema_name)
        )
        response = self._load().gpt_infer(
            system_prompt + schema_instruction,
            user_prompt,
        )
        return _extract_json_object(response)
