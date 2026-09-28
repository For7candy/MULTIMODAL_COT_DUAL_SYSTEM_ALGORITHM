from types import SimpleNamespace
import unittest

try:
    import torch
    from torch import nn
except ImportError:
    torch = None
    nn = None

if torch is not None:
    from power_vln.adapters import Qwen25VLBackboneAdapter


class FakeProcessor:
    def __init__(self, response):
        self.response = response
        self.messages = None

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        self.messages = messages
        return "rendered prompt"

    def __call__(self, text, padding, return_tensors):
        return {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.ones(1, 3, dtype=torch.long),
        }

    def batch_decode(self, generated_ids, **kwargs):
        return [self.response]


class FakeModel(nn.Module if nn is not None else object):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(8, 4)
        self.generate_kwargs = None

    def get_input_embeddings(self):
        return self.embedding

    def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        return torch.tensor([[4, 5]])


@unittest.skipIf(torch is None, "PyTorch is not installed")
class QwenAdapterTests(unittest.TestCase):
    def _request(self):
        return SimpleNamespace(
            system_prompt="Return JSON only.",
            user_prompt="Describe the scene.",
            json_schema={
                "type": "object",
                "properties": {"ok": {"type": "boolean"}},
            },
            embeddings=torch.randn(1, 2, 4),
            attention_mask=torch.tensor([[True, False]]),
        )

    def test_injected_spatial_tokens_precede_text_embeddings(self):
        model = FakeModel()
        processor = FakeProcessor('```json\n{"ok":true}\n```')
        adapter = Qwen25VLBackboneAdapter(
            model_path="injected",
            model=model,
            processor=processor,
            max_new_tokens=20,
        )
        output = adapter.generate_structured(self._request())
        self.assertEqual(output, {"ok": True})
        self.assertEqual(model.generate_kwargs["inputs_embeds"].shape, (1, 5, 4))
        self.assertEqual(
            model.generate_kwargs["attention_mask"].tolist(), [[1, 0, 1, 1, 1]]
        )
        self.assertIn("Required JSON Schema", processor.messages[0]["content"])
        self.assertFalse(model.generate_kwargs["do_sample"])

    def test_non_json_generation_is_rejected(self):
        adapter = Qwen25VLBackboneAdapter(
            model_path="injected",
            model=FakeModel(),
            processor=FakeProcessor("not-json"),
        )
        with self.assertRaises(ValueError):
            adapter.generate_structured(self._request())

    def test_embedding_dimension_mismatch_is_rejected(self):
        request = self._request()
        request.embeddings = torch.randn(1, 2, 5)
        adapter = Qwen25VLBackboneAdapter(
            model_path="injected",
            model=FakeModel(),
            processor=FakeProcessor('{"ok":true}'),
        )
        with self.assertRaises(ValueError):
            adapter.generate_structured(request)

    def test_partial_markdown_fence_is_not_silently_repaired(self):
        adapter = Qwen25VLBackboneAdapter(
            model_path="injected",
            model=FakeModel(),
            processor=FakeProcessor('```json\n{"ok":true}'),
        )
        with self.assertRaises(ValueError):
            adapter.generate_structured(self._request())


if __name__ == "__main__":
    unittest.main()
