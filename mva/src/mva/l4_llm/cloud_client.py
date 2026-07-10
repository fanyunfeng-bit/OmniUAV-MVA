"""DashScope(通义) 云端 LLM 适配器 —— QueryService 问答用云端 qwen3-vl-plus。

与本地 mva.l4_llm.client.LLMClient 的公开方法(complete / complete_messages)保持一致，
以便直接注入 QueryService。key 从环境变量 DASHSCOPE_API_KEY 读，禁止写死。
"""
from __future__ import annotations
import base64
import os
from typing import Any, Optional

import cv2
import numpy as np
import requests

_DEFAULT_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"


def _img_to_data_url(img: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", img)          # img: BGR/np.uint8 HxWx3
    if not ok:
        raise ValueError("cv2.imencode failed for image")
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


class DashScopeLLMClient:
    def __init__(
        self,
        model: str = "qwen3-vl-plus",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = 60,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError("DashScopeLLMClient 需要 API key：设置环境变量 DASHSCOPE_API_KEY")
        self.base_url = (base_url or os.environ.get("DASHSCOPE_BASE_URL")
                         or _DEFAULT_BASE).rstrip("/")
        self.timeout = timeout

    def _post(self, messages: list[dict], max_new_tokens: int) -> str:
        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}",
                     "Content-Type": "application/json"},
            json={"model": self.model, "messages": messages,
                  "max_tokens": max_new_tokens},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    def complete(self, prompt: str, images: Optional[list[np.ndarray]] = None,
                 max_new_tokens: int = 256) -> str:
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for img in images or []:
            content.append({"type": "image_url",
                            "image_url": {"url": _img_to_data_url(img)}})
        return self._post([{"role": "user", "content": content}], max_new_tokens)

    def complete_messages(self, messages: list[dict], max_new_tokens: int = 256) -> str:
        """把 MVA/Qwen 风格 messages 翻成 OpenAI content(文本+图像)。
        P0 限制：不支持视频段(video)。content 可为 str 或 list[part]；
        part: {"type":"text","text":...} / {"type":"image","image":np.ndarray or path}。"""
        out_msgs: list[dict[str, Any]] = []
        for m in messages:
            c = m.get("content")
            if isinstance(c, str):
                out_msgs.append({"role": m["role"], "content": c})
                continue
            parts: list[dict[str, Any]] = []
            for part in c or []:
                ptype = part.get("type")
                if ptype == "text":
                    parts.append({"type": "text", "text": part.get("text", "")})
                elif ptype == "image":
                    img = part.get("image")
                    arr = cv2.imread(img) if isinstance(img, str) else img
                    parts.append({"type": "image_url",
                                  "image_url": {"url": _img_to_data_url(arr)}})
                elif ptype == "video":
                    parts.append({"type": "text", "text": "[视频段在云端适配器 P0 中未支持]"})
            out_msgs.append({"role": m["role"], "content": parts})
        return self._post(out_msgs, max_new_tokens)
