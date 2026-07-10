"""Unit tests for L4 LLMClient.

All tests run in mock mode (model_path=None) so the real ~14 GB Qwen2.5-VL-7B
weights are never loaded in pytest. Real-mode + quantization is exercised
manually via demo_matrix.py m2 --quantize int4.
"""
from __future__ import annotations

import pytest

from mva.l4_llm import LLMClient


def test_mock_mode_default():
    c = LLMClient()
    assert c.is_mock is True
    assert c.is_loaded is False
    assert c.quantization is None


def test_mock_complete_returns_stub():
    c = LLMClient()
    out = c.complete("hello")
    assert "MOCK" in out


def test_real_path_recorded():
    c = LLMClient(model_path="Qwen/Qwen2.5-VL-7B-Instruct")
    assert c.is_mock is False
    assert c.model_path == "Qwen/Qwen2.5-VL-7B-Instruct"
    # is_loaded stays False until first .complete() (lazy load)
    assert c.is_loaded is False


def test_quantization_param_accepts_known_modes():
    LLMClient(quantization=None)
    LLMClient(quantization="int4")
    LLMClient(quantization="int8")


def test_quantization_param_rejects_unknown():
    with pytest.raises(ValueError):
        LLMClient(quantization="fp4")
    with pytest.raises(ValueError):
        LLMClient(quantization="invalid")


def test_load_resets_loaded_flag():
    c = LLMClient(model_path="Qwen/Qwen2.5-VL-7B-Instruct")
    # Simulate having loaded (we can't actually load in tests)
    c._model = object()
    c._processor = object()
    assert c.is_loaded is True
    c.load("Qwen/something-else")
    assert c.is_loaded is False
    assert c.model_path == "Qwen/something-else"


def test_unload_is_safe_when_not_loaded():
    c = LLMClient()
    c.unload()
    c.unload()    # idempotent
    assert c.is_loaded is False


def test_unload_clears_state():
    c = LLMClient(model_path="x")
    c._model = object()
    c._processor = object()
    c.unload()
    assert c._model is None
    assert c._processor is None
    assert c.is_loaded is False
