"""Pinned, locally executed Qwen inference for authoritative scoring."""

from __future__ import annotations

import threading

from .core import MAX_NEW_TOKENS, MODELS, MODEL_REVISIONS, TEMPERATURE, build_messages


class ModelEngine:
    def __init__(self):
        self._loaded = {}
        self._lock = threading.Lock()
        self.last_peak_vram_mib = None

    def _model(self, key: str):
        if key not in MODELS:
            raise ValueError("Unknown model")
        if key not in self._loaded:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            model_id = MODELS[key]
            # The source notebook used FP16 for Qwen2.5. On this RTX 5090 +
            # PyTorch 2.13 stack its logits contain NaNs. BF16 was checked
            # against the same prompt with finite logits and gradients.
            dtype = torch.bfloat16 if key == "qwen25" else torch.float32
            tokenizer = AutoTokenizer.from_pretrained(
                model_id, revision=MODEL_REVISIONS[key]
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=dtype, device_map="auto",
                attn_implementation="eager", revision=MODEL_REVISIONS[key]
            )
            model.eval()
            self._loaded[key] = (tokenizer, model)
        return self._loaded[key]

    def evaluate(self, prompt: str, model_key: str) -> list[str]:
        import torch

        # One student is permitted. The lock also avoids overlapping model
        # loads/generations when Jupyter retries while an evaluation is active.
        with self._lock:
            tokenizer, model = self._model(model_key)
            chat = tokenizer.apply_chat_template(
                build_messages(prompt),
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            encoded = tokenizer(chat, return_tensors="pt").to(model.device)
            input_len = encoded["input_ids"].shape[1]
            outputs = []
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            with torch.inference_mode():
                for _ in range(10):
                    generated = model.generate(
                        **encoded,
                        max_new_tokens=MAX_NEW_TOKENS,
                        do_sample=True,
                        temperature=TEMPERATURE,
                    )
                    outputs.append(
                        tokenizer.decode(generated[0][input_len:], skip_special_tokens=True)
                    )
            self.last_peak_vram_mib = (
                round(torch.cuda.max_memory_allocated() / (1024 ** 2), 1)
                if torch.cuda.is_available() else None
            )
            return outputs
