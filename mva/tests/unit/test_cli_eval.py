"""Unit tests for `mva eval` plumbing (M3.9 + M5.2-pilot + M5.2-prep).

Covers:
- `_stratified_qa` yields up to N per task across all 8 MVU task types
- `_run_qa_with_timeout` returns '[TIMEOUT]' when the inner call exceeds
  the budget (P2-09)
- `_resolve_nframes` drops multi-video QAs to a reduced budget (P3-16)
- `_is_oom` detects torch CUDA OOM across modern + legacy forms (P3-16)
- `_run_qa_with_oom_retry` halves nframes and retries once on OOM (P3-16)

No model load / no GPU; uses a fake LLMClient + a tiny in-memory
MVUEvalDataset fixture.
"""
from __future__ import annotations

import json
import time

import pytest

from mva.cli.eval import (
    MVU_TASK_TYPES,
    _is_oom,
    _resolve_nframes,
    _run_qa_with_oom_retry,
    _run_qa_with_timeout,
    _stratified_qa,
)
from mva.datasets import MVUEvalDataset


class _FakeQA:
    """Minimal QA stand-in for nframes / retry tests (no real metadata
    pipeline needed)."""

    def __init__(self, video_paths=None, qa_id="test"):
        self.metadata = {"video_paths": list(video_paths or [])}
        self.qa_id = qa_id


@pytest.fixture
def mini_mvu_root(tmp_path):
    """Build a minimal MVU-Eval root with 3 questions per task type."""
    root = tmp_path / "mvu"
    root.mkdir()
    qa: dict[str, dict] = {}
    idx = 0
    for task in MVU_TASK_TYPES:
        for _ in range(3):
            qa[str(idx)] = {
                "video_paths": [],
                "question": f"placeholder for {task}",
                "options": ["A.", "B.", "C.", "D."],
                "ground_truth": "A",
                "task": task,
            }
            idx += 1
    (root / "MVU_Eval_QAs.json").write_text(json.dumps(qa))
    return root


def test_stratified_qa_yields_per_task_count(mini_mvu_root):
    ds = MVUEvalDataset(root=mini_mvu_root)
    qas = list(_stratified_qa(ds, per_task=2))
    # 8 task types × 2 each = 16
    assert len(qas) == 16
    # Each task type appears exactly twice
    from collections import Counter
    by_task = Counter(qa.task for qa in qas)
    assert by_task == {task: 2 for task in MVU_TASK_TYPES}


def test_stratified_qa_respects_available_max(mini_mvu_root):
    """If per_task > available, take all available without crashing.
    (Fixture has 3 per task; requesting 5 should yield 3 each.)"""
    ds = MVUEvalDataset(root=mini_mvu_root)
    qas = list(_stratified_qa(ds, per_task=5))
    # 8 task types × 3 each (max available) = 24
    assert len(qas) == 24


def test_run_qa_with_timeout_returns_timeout_sentinel(monkeypatch):
    """A slow `_run_qa` exceeding the budget returns '[TIMEOUT]' rather
    than blocking forever or raising. The eval loop continues."""
    from mva.cli import eval as eval_mod

    def slow_run_qa(*args, **kwargs):
        time.sleep(5.0)
        return "B"

    monkeypatch.setattr(eval_mod, "_run_qa", slow_run_qa)

    class _FakeQA:
        qa_id = "test-1"
    out = _run_qa_with_timeout(
        adapter=None, llm=None, qa=_FakeQA(),
        nframes=8, max_pixels=64, timeout_sec=1,
    )
    assert out == "[TIMEOUT]"


def test_run_qa_with_timeout_passes_through_fast_result(monkeypatch):
    """Fast `_run_qa` returns normally within budget."""
    from mva.cli import eval as eval_mod

    def fast_run_qa(*args, **kwargs):
        return "C"

    monkeypatch.setattr(eval_mod, "_run_qa", fast_run_qa)

    out = _run_qa_with_timeout(
        adapter=None, llm=None, qa=_FakeQA(qa_id="test-2"),
        nframes=8, max_pixels=64, timeout_sec=10,
    )
    assert out == "C"


# -------- P3-16: _resolve_nframes --------

def test_resolve_nframes_single_video_keeps_base():
    qa = _FakeQA(video_paths=["a.mp4"])
    assert _resolve_nframes(qa, base_nframes=32, threshold=4, reduced=16) == 32


def test_resolve_nframes_below_threshold_keeps_base():
    qa = _FakeQA(video_paths=["a.mp4", "b.mp4", "c.mp4"])
    assert _resolve_nframes(qa, base_nframes=32, threshold=4, reduced=16) == 32


def test_resolve_nframes_at_threshold_drops_to_reduced():
    qa = _FakeQA(video_paths=[f"v{i}.mp4" for i in range(4)])
    assert _resolve_nframes(qa, base_nframes=32, threshold=4, reduced=16) == 16


def test_resolve_nframes_above_threshold_drops_to_reduced():
    qa = _FakeQA(video_paths=[f"v{i}.mp4" for i in range(8)])
    assert _resolve_nframes(qa, base_nframes=32, threshold=4, reduced=16) == 16


def test_resolve_nframes_missing_video_paths_treated_as_zero():
    """Empty / missing video_paths shouldn't trip the multi-video branch."""
    qa = _FakeQA(video_paths=[])
    assert _resolve_nframes(qa, base_nframes=32, threshold=4, reduced=16) == 32


def test_resolve_nframes_reduced_capped_at_base():
    """If user explicitly passes a tiny --nframes, 'reduced' cannot
    increase it upward — capped at base."""
    qa = _FakeQA(video_paths=[f"v{i}.mp4" for i in range(6)])
    assert _resolve_nframes(qa, base_nframes=8, threshold=4, reduced=16) == 8


# -------- P3-16: _is_oom --------

def test_is_oom_detects_modern_oom_class_name():
    """torch.cuda.OutOfMemoryError is matched by class name."""
    class OutOfMemoryError(RuntimeError):  # noqa: N818 — mimic torch
        pass

    assert _is_oom(OutOfMemoryError("CUDA OOM")) is True


def test_is_oom_detects_legacy_runtime_error_message():
    assert _is_oom(RuntimeError("CUDA out of memory. Tried to allocate 1.5 GiB.")) is True


def test_is_oom_rejects_generic_runtime_error():
    assert _is_oom(RuntimeError("indices must be non-negative")) is False


def test_is_oom_rejects_unrelated_exception():
    assert _is_oom(ValueError("not OOM")) is False


# -------- P3-16: _run_qa_with_oom_retry --------

def test_oom_retry_fast_path_no_retry(monkeypatch):
    """No OOM → first call returns, no retry attempted."""
    from mva.cli import eval as eval_mod

    seen_nframes: list[int] = []

    def fast(adapter, llm, qa, nframes, max_pixels, *, timeout_sec):
        seen_nframes.append(nframes)
        return "A"
    monkeypatch.setattr(eval_mod, "_run_qa_with_timeout", fast)

    out = _run_qa_with_oom_retry(
        adapter=None, llm=None, qa=_FakeQA(qa_id="fast"),
        nframes=32, max_pixels=64, timeout_sec=10,
    )
    assert out == "A"
    assert seen_nframes == [32]


def test_oom_retry_recovers_with_halved_nframes(monkeypatch):
    """First call OOM, second call (halved nframes) succeeds."""
    from mva.cli import eval as eval_mod

    seen_nframes: list[int] = []

    def maybe_oom(adapter, llm, qa, nframes, max_pixels, *, timeout_sec):
        seen_nframes.append(nframes)
        if len(seen_nframes) == 1:
            raise RuntimeError("CUDA out of memory")
        return "B"
    monkeypatch.setattr(eval_mod, "_run_qa_with_timeout", maybe_oom)

    out = _run_qa_with_oom_retry(
        adapter=None, llm=None, qa=_FakeQA(qa_id="recover"),
        nframes=32, max_pixels=64, timeout_sec=10,
    )
    assert out == "B"
    assert seen_nframes == [32, 16]


def test_oom_retry_gives_up_after_second_oom(monkeypatch):
    """Two OOMs in a row → error sentinel returned (not raised)."""
    from mva.cli import eval as eval_mod

    def always_oom(*args, **kwargs):
        raise RuntimeError("CUDA out of memory")
    monkeypatch.setattr(eval_mod, "_run_qa_with_timeout", always_oom)

    out = _run_qa_with_oom_retry(
        adapter=None, llm=None, qa=_FakeQA(qa_id="fail"),
        nframes=32, max_pixels=64, timeout_sec=10,
    )
    assert "OOM retry failed" in out
    assert "32→16" in out  # message records the attempted reduction


def test_oom_retry_propagates_non_oom_exceptions(monkeypatch):
    """Non-OOM RuntimeErrors propagate to the cmd_eval-level handler."""
    from mva.cli import eval as eval_mod

    def broken(*args, **kwargs):
        raise ValueError("not OOM")
    monkeypatch.setattr(eval_mod, "_run_qa_with_timeout", broken)

    with pytest.raises(ValueError):
        _run_qa_with_oom_retry(
            adapter=None, llm=None, qa=_FakeQA(qa_id="propagate"),
            nframes=32, max_pixels=64, timeout_sec=10,
        )
