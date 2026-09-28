"""Lazy local Qwen2.5-VL backend with direct spatial-token injection."""

import json
from typing import Any, Optional


def _unwrap_json_fence(text: str) -> str:
    """Remove one outer Markdown JSON fence without repairing its contents."""

    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if len(lines) < 3 or lines[-1].strip() != "```":
        return stripped
    if lines[0].strip().lower() not in ("```", "```json"):
        return stripped
    return "\n".join(lines[1:-1]).strip()


class Qwen25VLBackboneAdapter:
    """Generate structured JSON from text plus fused spatial embeddings."""

    def __init__(
        self,
        model_path: str,
        device: str = "cuda:0",
        dtype: str = "bfloat16",
        max_new_tokens: int = 768,
        model: Any = None,
        processor: Any = None,
    ) -> None:
        if not model_path:
            raise ValueError("model_path must not be empty")
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if (model is None) != (processor is None):
            raise ValueError("model and processor must be injected together")
        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        self.max_new_tokens = max_new_tokens
        self._model = model
        self._processor = processor
        self.last_raw_output: Optional[str] = None
        self.last_payload: Optional[dict] = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def _load(self) -> None:
        if self.is_loaded:
            return
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        dtype = getattr(torch, self.dtype, None)
        if dtype is None:
            raise ValueError("unsupported torch dtype: {}".format(self.dtype))
        self._processor = AutoProcessor.from_pretrained(
            self.model_path,
            local_files_only=True,
            use_fast=False,
        )
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_path,
            local_files_only=True,
            dtype=dtype,
            device_map={"": self.device},
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
        ).eval()

    def unload(self) -> None:
        """Release model references and return cached CUDA memory to PyTorch."""

        self._model = None
        self._processor = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def generate_structured(self, request: Any) -> dict:
        import torch

        self._load()
        if request.embeddings.shape[0] != 1:
            raise ValueError("Qwen runtime supports one navigation stream")
        schema_text = json.dumps(
            request.json_schema,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        messages = [
            {
                "role": "system",
                "content": (
                    request.system_prompt
                    + "\nThe spatial embeddings precede this text prompt."
                    + "\nRequired JSON Schema:\n"
                    + schema_text
                ),
            },
            {"role": "user", "content": request.user_prompt},
        ]
        prompt = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        text_inputs = self._processor(
            text=[prompt], padding=True, return_tensors="pt"
        )
        embedding_layer = self._model.get_input_embeddings()
        model_device = embedding_layer.weight.device
        model_dtype = embedding_layer.weight.dtype
        if request.embeddings.shape[-1] != embedding_layer.embedding_dim:
            raise ValueError(
                "spatial embedding size {} does not match Qwen embedding size {}".format(
                    request.embeddings.shape[-1], embedding_layer.embedding_dim
                )
            )
        input_ids = text_inputs["input_ids"].to(model_device)
        text_mask = text_inputs["attention_mask"].to(model_device)
        text_embeddings = embedding_layer(input_ids)
        spatial_embeddings = request.embeddings.detach().to(
            device=model_device, dtype=model_dtype
        )
        spatial_mask = request.attention_mask.to(
            device=model_device, dtype=text_mask.dtype
        )
        combined_embeddings = torch.cat(
            (spatial_embeddings, text_embeddings), dim=1
        )
        combined_mask = torch.cat((spatial_mask, text_mask), dim=1)
        with torch.inference_mode():
            generated_ids = self._model.generate(
                inputs_embeds=combined_embeddings,
                attention_mask=combined_mask,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                use_cache=True,
            )
        output = self._processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        self.last_raw_output = output
        normalized = _unwrap_json_fence(output)
        try:
            payload = json.loads(normalized)
        except json.JSONDecodeError as error:
            raise ValueError("Qwen output is not one valid JSON object") from error
        if not isinstance(payload, dict):
            raise ValueError("Qwen structured output must be a JSON object")
        self.last_payload = payload
        return payload
