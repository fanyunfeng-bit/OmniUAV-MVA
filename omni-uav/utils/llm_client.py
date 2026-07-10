import base64
import mimetypes
import os
from typing import Iterable, Optional, Dict, Any, List
from enum import Enum
from pathlib import Path

import requests


class ModelType(Enum):
    """Model type enumeration."""
    OPENAI_COMPATIBLE = "openai_compatible"  # OpenAI-compatible API (default)
    QWEN_VL = "qwen_vl"  # Qwen VL via DashScope API
    LOCAL = "local"  # Local model (e.g., VideoLlama3-7B)


# Model configurations
MODEL_CONFIGS = {
    "gpt-4o-mini": {"type": ModelType.OPENAI_COMPATIBLE, "api_key_env": "LLM_API_KEY"},
    "gpt-4o": {"type": ModelType.OPENAI_COMPATIBLE, "api_key_env": "LLM_API_KEY"},
    "qwen3-vl-plus": {"type": ModelType.QWEN_VL, "api_key_env": "DASHSCOPE_API_KEY", "model": "qwen3-vl-plus"},
    "qwen3-vl-max": {"type": ModelType.QWEN_VL, "api_key_env": "DASHSCOPE_API_KEY", "model": "qwen-vl-max"},
    "videollama3-7b": {"type": ModelType.LOCAL, "model_path": ""},  # Local model
}


class LlmClient:
    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: int = 60,
        model_path: Optional[str] = None,
    ):
        """
        Initialize LLM client.

        Args:
            model_name: Name of the model to use (e.g., "qwen3-vl-plus", "videollama3-7b")
            base_url: Base URL for OpenAI-compatible API (optional)
            api_key: API key (optional, will use env var if not provided)
            timeout: Request timeout in seconds
            model_path: Path to local model (for local models)
        """
        self.model_name = model_name
        self.timeout = timeout
        self.model_path = model_path

        # Get model configuration
        config = MODEL_CONFIGS.get(model_name.lower(), {})
        self.model_type = config.get("type", ModelType.OPENAI_COMPATIBLE)

        # Setup based on model type
        if self.model_type == ModelType.QWEN_VL:
            # Qwen VL uses DashScope API
            self.base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
            api_key_env = config.get("api_key_env", "DASHSCOPE_API_KEY")
            self.api_key = api_key or os.getenv(api_key_env)
            self.model = config.get("model", "qwen-vl-plus")
        elif self.model_type == ModelType.LOCAL:
            # Local model - VideoLlama3-7B
            self.base_url = None
            self.api_key = None
            self.model = model_name
            self._local_client = None  # Will be initialized in load()
        else:
            # OpenAI-compatible API
            default_url = os.getenv("LLM_BASE_URL") or "https://api.openai.com"
            self.base_url = self._normalize_base_url(base_url or default_url)
            api_key_env = config.get("api_key_env", "LLM_API_KEY")
            self.api_key = api_key or os.getenv(api_key_env)
            self.model = model_name

        self.loaded = False

    def load(self):
        """Load the model. For API models, this just validates credentials. For local models, loads the actual model."""
        if self.model_type == ModelType.LOCAL:
            # Load local model
            return self._load_local_model()
        else:
            # API models - just validate credentials
            if not self.api_key:
                raise RuntimeError(f"Missing API key. Please set {MODEL_CONFIGS.get(self.model_name.lower(), {}).get('api_key_env', 'LLM_API_KEY')} environment variable.")
            self.loaded = True
            return True

    def _load_local_model(self):
        """Load local VideoLLaMA3 model."""
        try:
            # Import local VideoLLaMA3 client
            from utils.videollama3_client import VideoLLaMA3Client

            # Determine model identifier (HuggingFace model ID or local path)
            # Priority: model_path parameter > VIDEOLLAMA_MODEL_PATH env var > default HuggingFace model
            model_identifier = self.model_path or os.getenv("VIDEOLLAMA_MODEL_PATH")

            if not model_identifier:
                # Use default HuggingFace model ID
                model_identifier = "DAMO-NLP-SG/VideoLLaMA3-7B"

            # Determine device
            import torch
            device = "cuda:0" if torch.cuda.is_available() else "cpu"

            # Create and initialize client
            self._local_client = VideoLLaMA3Client(model_name=model_identifier, device=device)
            self._local_client.initialize()

            self.loaded = True
            return True

        except ImportError as e:
            raise RuntimeError(
                f"Failed to import VideoLLaMA3 client: {e}\n"
                "Please install required dependencies: pip install transformers torch accelerate"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load local model: {e}")

    def chat(self, prompt: str, image_paths: Iterable[str]):
        """Chat with the LLM model."""
        if self.model_type == ModelType.LOCAL:
            return self._chat_local(prompt, image_paths)
        else:
            return self._chat_api(prompt, image_paths)

    def _chat_api(self, prompt: str, image_paths: Iterable[str]):
        """Chat using API-based models."""
        if not self.api_key:
            raise RuntimeError("Missing LLM_API_KEY environment variable")

        images = [self._image_part(path) for path in image_paths]
        content = [{"type": "text", "text": prompt}, *images]

        # Add system message to ensure language-aware responses
        messages = [
            {
                "role": "system",
                "content": "You are a helpful AI assistant. Please respond in the same language as the user's question. If the user asks in Chinese, respond in Chinese. If the user asks in English, respond in English. Apply this rule to all other languages as well."
            },
            {"role": "user", "content": content}
        ]

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": 4000,  # Increased for tracking responses with many IDs
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        url = f"{self.base_url}/chat/completions"
        response = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
        if response.status_code >= 400:
            raise RuntimeError(f"LLM request failed ({response.status_code}): {response.text}")

        data = response.json()
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError("LLM response missing choices")
        message = choices[0].get("message") or {}
        content_text = message.get("content")
        if not content_text:
            raise RuntimeError("LLM response missing content")
        return content_text

    def _chat_local(self, prompt: str, image_paths: Iterable[str]):
        """Chat using local VideoLLaMA3 model."""
        if not self._local_client:
            raise RuntimeError("Local model not loaded. Please call load() first.")

        # Convert to list
        image_paths_list = list(image_paths)

        if not image_paths_list:
            # Text-only generation
            return self._local_client.generate_text_only(prompt, max_new_tokens=600)
        elif len(image_paths_list) == 1:
            # Single image
            return self._local_client.analyze_single_image(image_paths_list[0], prompt, max_new_tokens=600)
        else:
            # Multiple images - process as a sequence
            # Build a prompt that mentions multiple images
            multi_image_prompt = f"以下是{len(image_paths_list)}张图片，请综合分析回答问题：\n\n问题：{prompt}"
            return self._local_client.generate_text_only(multi_image_prompt, max_new_tokens=600)

    def _normalize_base_url(self, base_url: str):
        base_url = base_url.rstrip("/")
        if base_url.endswith("/v1"):
            return base_url
        return f"{base_url}/v1"

    def _image_part(self, path: str):
        mime_type = mimetypes.guess_type(path)[0] or "image/jpeg"
        with open(path, "rb") as file:
            encoded = base64.b64encode(file.read()).decode("ascii")
        return {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{encoded}"}}
