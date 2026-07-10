"""OmniUAV → MVA sidecar 的 RPC 客户端(requests 封装)。契约见 spec §5。图像只传路径。"""
from __future__ import annotations
import os
from typing import Any, Optional

import requests

DEFAULT_BASE = os.environ.get("MVA_SIDECAR_URL", "http://127.0.0.1:8900")


class MvaClient:
    def __init__(self, base_url: str = DEFAULT_BASE, timeout: int = 60) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._s = requests.Session()

    def is_alive(self) -> bool:
        try:
            r = self._s.get(f"{self.base_url}/health", timeout=3)
            r.raise_for_status()
            return bool(r.json().get("engine_ready", False))
        except Exception:                                # noqa: BLE001
            return False

    def ingest_start(self, source: str, mode: str = "offline",
                     dataset: Optional[str] = None, config: Optional[dict] = None) -> str:
        r = self._s.post(f"{self.base_url}/ingest/start",
                         json={"source": source, "mode": mode,
                               "dataset": dataset, "config": config or {}},
                         timeout=self.timeout)
        r.raise_for_status()
        return r.json()["job_id"]

    def ingest_status(self, job_id: str) -> dict[str, Any]:
        r = self._s.get(f"{self.base_url}/ingest/status",
                        params={"job": job_id}, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def answer(self, query: str, attachments: Optional[list[str]] = None,
               session_id: Optional[str] = None) -> dict[str, Any]:
        r = self._s.post(f"{self.base_url}/answer",
                         json={"query": query, "attachments": attachments or [],
                               "session_id": session_id},
                         timeout=self.timeout)
        r.raise_for_status()
        return r.json()
