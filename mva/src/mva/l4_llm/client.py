"""L4 LLMClient — Qwen2.5-VL-7B wrapper with mock fallback + optional quantization.

Supports:
- Mock mode: no model_path → returns deterministic stub responses
- Real mode (FP16): model_path = "Qwen/Qwen2.5-VL-7B-Instruct" or a local
  LoRA checkpoint (lazy-loaded on first `complete` call)
- Real mode (INT4 / INT8): pass `quantization="int4"` (or "int8"). Uses
  `bitsandbytes` on-the-fly quantization. Required when the embedder
  (Qwen3-VL-Embedding-8B, ~18 GB) and gen LLM (~14 GB FP16 / ~5 GB INT4)
  must coexist on a 24 GB GPU.

The `load(model_path)` method is the §3.4 #5 interface for swapping in LoRA
SFT checkpoints (M5+).
"""
from __future__ import annotations

from typing import Literal, Optional

import numpy as np


_QuantizeMode = Optional[Literal["int4", "int8"]]


class LLMClient:
    def __init__(
        self,
        model_path: Optional[str] = None,
        quantization: _QuantizeMode = None,
    ) -> None:
        """Construct an LLM client.

        Parameters
        ----------
        model_path : str | None
            HuggingFace model id (e.g. "Qwen/Qwen2.5-VL-7B-Instruct") or
            local path. If None, the client returns mock responses so tests
            and scaffolding work without downloading 14 GB of weights.
        quantization : "int4" | "int8" | None
            Optional bitsandbytes on-the-fly quantization. None = FP16
            (default). "int4" cuts Qwen2.5-VL-7B to ~5 GB VRAM so it can
            coexist with Qwen3-VL-Embedding-8B on a 24 GB GPU. "int8" cuts
            it to ~8 GB. Quantization only applies when model_path is set.
        """
        if quantization not in (None, "int4", "int8"):
            raise ValueError(
                f"quantization must be None / 'int4' / 'int8', got {quantization!r}"
            )
        self.model_path = model_path
        self.quantization = quantization
        self._model = None
        self._processor = None

    def load(self, model_path: str) -> None:
        """Swap to a different checkpoint (e.g. a LoRA-tuned variant). 🔌 §3.4 #5."""
        self.model_path = model_path
        # Force re-init on next call
        self._model = None
        self._processor = None

    def unload(self) -> None:
        """Free the model's GPU memory. Safe to call multiple times.

        Useful when juggling multiple heavy models (embedder vs gen LLM)
        on a single GPU; less critical when INT4 quantization is on and
        both can coexist.
        """
        if self._model is None and self._processor is None:
            return
        try:
            del self._model
            del self._processor
        finally:
            self._model = None
            self._processor = None
        try:
            import torch  # type: ignore
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:  # pragma: no cover
            pass

    @property
    def is_mock(self) -> bool:
        return self.model_path is None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None and self._processor is not None

    # ----------------------------------------------------------------------
    # Public inference API
    # ----------------------------------------------------------------------

    def complete(
        self,
        prompt: str,
        images: Optional[list[np.ndarray]] = None,
        max_new_tokens: int = 256,
    ) -> str:
        """Generate a completion for `prompt` optionally conditioned on `images`.

        Images are BGR numpy arrays (OpenCV convention). They get converted
        to PIL.Image in RGB inside this method.
        """
        if self.is_mock:
            return self._mock_response(prompt, images)
        self._ensure_loaded()
        return self._real_complete(prompt, images or [], max_new_tokens)

    def complete_messages(
        self,
        messages: list[dict],
        max_new_tokens: int = 128,
    ) -> str:
        """Direct messages-format completion.

        For richer multi-modal prompts (mixed text + multi-video + multi-image
        interleaved) that don't fit the `complete(prompt, images)` shape.
        Used by MVU-Eval evaluation pipeline. `messages` follows the Qwen-VL
        chat-template format:

            [
              {"role": "system", "content": "..."},
              {"role": "user",   "content": [
                  {"type": "text",  "text": "..."},
                  {"type": "image", "image": "/path/or/PIL"},
                  {"type": "video", "video": "/path.mp4", "max_pixels": 720*720, "nframes": 32},
                  ...
              ]},
            ]
        """
        if self.is_mock:
            def _parts(messages):
                for m in messages:
                    content = m.get("content") or []
                    if isinstance(content, str):
                        content = [{"type": "text", "text": content}]
                    yield from content
            n_text = sum(1 for c in _parts(messages)
                         if isinstance(c, dict) and c.get("type") == "text")
            n_img = sum(1 for c in _parts(messages)
                        if isinstance(c, dict) and c.get("type") == "image")
            n_vid = sum(1 for c in _parts(messages)
                        if isinstance(c, dict) and c.get("type") == "video")
            return (
                f"[MOCK LLMClient] complete_messages: "
                f"{n_text} text + {n_img} image + {n_vid} video parts. "
                "Set model_path to use the real Qwen2.5-VL-7B."
            )
        self._ensure_loaded()
        return self._real_complete_messages(messages, max_new_tokens)

    # ----------------------------------------------------------------------
    # Internal
    # ----------------------------------------------------------------------

    def _mock_response(
        self, prompt: str, images: Optional[list[np.ndarray]]
    ) -> str:
        n_img = len(images) if images else 0
        return (
            "[MOCK LLMClient] Received "
            f"prompt({len(prompt)} chars) + {n_img} image(s). "
            "Set --llm <model_path> to use the real Qwen2.5-VL-7B."
        )

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._processor is not None:
            return
        try:
            import torch  # type: ignore
            from transformers import (  # type: ignore
                AutoProcessor,
                Qwen2_5_VLForConditionalGeneration,
            )
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "transformers / torch are required for the real LLMClient. "
                "Install with: pip install 'mva[llm]'"
            ) from exc

        from_kwargs: dict = {
            "torch_dtype": "auto",
            "device_map": "auto",
        }

        if self.quantization is not None:
            try:
                from transformers import BitsAndBytesConfig  # type: ignore
            except ImportError as exc:  # pragma: no cover
                raise ImportError(
                    "BitsAndBytesConfig requires the 'bitsandbytes' package. "
                    "Install with: pip install bitsandbytes"
                ) from exc
            if self.quantization == "int4":
                from_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
            elif self.quantization == "int8":
                from_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_8bit=True,
                )

        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_path, **from_kwargs
        )
        self._processor = AutoProcessor.from_pretrained(self.model_path)

    def _real_complete(
        self,
        prompt: str,
        images: list[np.ndarray],
        max_new_tokens: int,
    ) -> str:
        # Convert BGR → PIL RGB
        from PIL import Image  # type: ignore

        pil_images = [
            Image.fromarray(img[:, :, ::-1]) for img in images
        ] if images else []

        content: list[dict] = []
        for pil_img in pil_images:
            content.append({"type": "image", "image": pil_img})
        content.append({"type": "text", "text": prompt})

        messages = [{"role": "user", "content": content}]

        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._processor(
            text=[text],
            images=pil_images if pil_images else None,
            padding=True,
            return_tensors="pt",
        ).to(self._model.device)

        generated_ids = self._model.generate(
            **inputs, max_new_tokens=max_new_tokens
        )
        generated_only = generated_ids[:, inputs.input_ids.shape[1] :]
        out = self._processor.batch_decode(
            generated_only, skip_special_tokens=True
        )
        return out[0].strip()

    def _real_complete_messages(
        self, messages: list[dict], max_new_tokens: int,
    ) -> str:
        """Multi-modal multi-video chat-template path. Uses qwen-vl-utils to
        resolve video paths to frame tensors, then runs generate()."""
        try:
            from qwen_vl_utils import process_vision_info  # type: ignore
        except ImportError as exc:                          # pragma: no cover
            raise ImportError(
                "qwen-vl-utils is required for complete_messages() with videos. "
                "Install with: pip install 'mva[llm]'"
            ) from exc

        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self._processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self._model.device)

        generated_ids = self._model.generate(
            **inputs, max_new_tokens=max_new_tokens
        )
        generated_only = generated_ids[:, inputs.input_ids.shape[1] :]
        out = self._processor.batch_decode(
            generated_only, skip_special_tokens=True
        )
        return out[0].strip()
