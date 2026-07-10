"""`mva query` — Mode A REPL backed by QueryService.

QueryService is the public facade — the CLI uses it, and a future web UI
should use the same class so the model loading / lifecycle / locking is
centralized. Do NOT bake LLM-specific logic into the CLI subcommand.
"""
from __future__ import annotations

import argparse
import threading
from pathlib import Path
from typing import Iterator, Optional

from mva.cli._common import (
    add_cross_view_arg,
    add_embedder_args,
    add_llm_args,
    add_store_args,
)
from mva.contracts import RichQuery
from mva.l4_llm import LLMClient
from mva.l5_state import (
    DEFAULT_DIM,
    DEFAULT_MODEL as DEFAULT_EMBEDDER_MODEL,
    MultimodalEmbedder,
    VectorStore,
    WorldStateStore,
)
from mva.l6_interaction import (
    Orchestrator,
    OrchestratorResult,
    QueryPlanner,
    build_default_registry,
)
from mva.l6_interaction.memory import ConversationMemory


# ----------------------------------------------------------------------
# QueryService facade
# ----------------------------------------------------------------------


class QueryService:
    """Stateful query entrypoint. One instance owns models + stores +
    orchestrator; `.answer(query)` is thread-safe via an internal lock.

    Lifecycle:
      service = QueryService(db_path=..., chroma_dir=..., llm_model=...)
      result = service.answer(RichQuery(text="...", attachments=[...]))
      service.close()

    A future UI imports this class, instantiates once at startup, and
    invokes `.answer()` per HTTP request."""

    def __init__(
        self,
        db_path: str,
        chroma_dir: Optional[str] = None,
        llm_model: Optional[str] = None,
        embedder_model: Optional[str] = DEFAULT_EMBEDDER_MODEL,
        embed_dim: int = DEFAULT_DIM,
        quantization: Optional[str] = None,
        device: Optional[str] = None,
        enable_cross_view: bool = True,
        llm=None,
        embedder=None,
    ) -> None:
        self.db_path = db_path
        self.chroma_dir = chroma_dir
        self.llm_model = llm_model
        self.embedder_model = embedder_model

        # Open the structured store
        self.store = WorldStateStore(db_path=db_path)

        # Open the embedder + vector store iff a chroma dir is configured.
        # 允许注入已加载的 embedder(engine 切库时复用，免重载 16G)；注入的不归本对象所有(close 不 unload)。
        self.embedder: Optional[MultimodalEmbedder] = embedder
        self._own_embedder = embedder is None
        self.vstore: Optional[VectorStore] = None
        if chroma_dir and (embedder is not None or embedder_model):
            if self.embedder is None:
                self.embedder = MultimodalEmbedder(
                    model_path=embedder_model, dim=embed_dim, device=device,
                )
                self.embedder._ensure_loaded()
            self.vstore = VectorStore(
                persist_dir=chroma_dir,
                embedding_function=self.embedder.as_chromadb_embedding_function(),
            )

        # Construct the gen LLM (lazy-loads on first .complete())
        # 允许注入自定义生成 LLM(如云端 DashScopeLLMClient)；否则用本地 LLMClient。
        if llm is not None:
            self.llm = llm
        else:
            self.llm = LLMClient(model_path=llm_model, quantization=quantization)

        # Wire up orchestrator
        self.registry = build_default_registry(
            self.store, vstore=self.vstore, llm=self.llm,
            enable_cross_view=enable_cross_view,
        )
        self.planner = QueryPlanner(
            self.llm, self.registry,
            db_schema=self.store.get_schema_summary(),
        )
        self.orchestrator = Orchestrator(
            self.llm, self.planner, self.registry,
            embedder=self.embedder, vstore=self.vstore,
            db_context=self.store.get_schema_summary(),
        )

        self._lock = threading.Lock()

    def answer(
        self,
        query: str | RichQuery,
        memory: Optional[ConversationMemory] = None,
    ) -> OrchestratorResult:
        """Run one query through the Mode A pipeline. Thread-safe.

        When `memory` is provided (session-scoped, owned by the caller — e.g.
        the UI's gr.State), the current question is embedded once (reused for
        both relevance selection and as this turn's stored vector), relevant
        history is rendered into a block and threaded into both LLM calls, and
        the completed turn is appended back into `memory`.
        """
        with self._lock:
            rich = query if isinstance(query, RichQuery) else RichQuery(text=query)
            history_block = ""
            q_emb: Optional[list[float]] = None
            if memory is not None:
                if self.embedder is not None and rich.text:
                    q_emb = self.embedder.encode_text(rich.text)   # list[float]
                relevant = memory.select_relevant(rich.text, q_emb)
                history_block = ConversationMemory.render_block(relevant)
            result = self.orchestrator.run(query, history_block=history_block)
            # Skip attachment-only turns (empty text): they carry no question
            # and no embedding, so they're inert for selection but would waste
            # a max_turns slot and render as a blank "用户:" line later.
            if memory is not None and rich.text:
                memory.add_turn(rich.text, result.answer, embedding=q_emb)
            return result

    def stream_answer(self, query: str | RichQuery) -> Iterator[str]:
        """🔌 Streaming version — currently yields the full answer as one
        chunk. v2+ swap for token-by-token streaming when the UI needs it."""
        result = self.answer(query)
        yield result.answer

    def list_tools(self) -> list[dict[str, str]]:
        """Inspectable tool catalog. UI can render this to show users
        what the system can do without revealing prompt internals."""
        return [
            {"name": name, "description": self.registry._tools[name].description}    # type: ignore[attr-defined]
            for name in self.registry.names()
        ]

    def close(self) -> None:
        """Release GPU + DB resources."""
        if self.embedder is not None and self._own_embedder:
            self.embedder.unload()
        self.llm.unload()
        self.store.close()

    def __enter__(self) -> "QueryService":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


# ----------------------------------------------------------------------
# `mva query` subcommand — REPL
# ----------------------------------------------------------------------


def add_subparser(sub) -> None:
    p = sub.add_parser(
        "query",
        help="Mode A REPL — type natural-language questions, get answers",
    )
    add_store_args(p, db_required=True)
    add_llm_args(p, llm_required=True)
    add_embedder_args(p)
    add_cross_view_arg(p)
    p.add_argument("--questions", type=Path, default=None,
                   help="One NL question per line; skips the REPL")
    p.set_defaults(func=cmd_query)


def cmd_query(args: argparse.Namespace) -> int:
    if not _check_db_populated(args.db_path):
        return 1
    _warn_hidden_cross_view(args.db_path, args.cross_view)

    effective_quantize = _resolve_quantize(args)

    with QueryService(
        db_path=args.db_path,
        chroma_dir=args.chroma_dir,
        llm_model=args.llm,
        embedder_model=args.embedder_model if args.chroma_dir else None,
        embed_dim=args.embed_dim,
        quantization=effective_quantize,
        enable_cross_view=(args.cross_view == "auto"),
    ) as service:
        print(f"[L6] tools: {', '.join(t['name'] for t in service.list_tools())} "
              f"(cross-view={args.cross_view})")

        if args.questions:
            for q in args.questions.read_text().splitlines():
                q = q.strip()
                if q and not q.startswith("#"):
                    _print_result(service.answer(q))
            return 0

        print("\n========================================")
        print("Interactive mode. Type a question (空行退出).")
        print("========================================")
        # Conversation memory persists across turns in the interactive REPL
        # (cue-gated recency + relevance). --questions batch mode above stays
        # non-conversational (each question independent).
        repl_memory = ConversationMemory()
        try:
            while True:
                text = input("\n> ").strip()
                if not text:
                    break
                _print_result(service.answer(text, memory=repl_memory))
        except (EOFError, KeyboardInterrupt):
            print()
    return 0


_VRAM_FORCE_INT4_THRESHOLD_BYTES = 30 * 1024 ** 3   # 30 GB — A100/H100 etc.


def _resolve_quantize(args: argparse.Namespace) -> Optional[str]:
    """M3.8 (PROBLEMS P2-02 + P2-03): figure out the effective LLM
    quantization for this CLI invocation **without mutating** args.

    - User explicitly set `--quantize` → respect it (any path)
    - User passes `--no-auto-quantize` → respect; never auto-force
    - With `--chroma-dir` (embedder loaded too):
        * GPU > 30 GB → skip auto-force; user keeps FP16
        * GPU ≤ 30 GB or no CUDA → auto-force INT4 (peak ~22 GB)
    - Without `--chroma-dir` → no embedder coexistence concern → leave as-is

    Returns the quantize value to pass to LLMClient (None = FP16).
    Prints a one-line decision banner so the user knows what they got.
    """
    if args.quantize is not None:
        return args.quantize
    if getattr(args, "no_auto_quantize", False):
        return None
    if not args.chroma_dir:
        return None

    # Embedder + gen LLM coexist case — decide based on VRAM
    total_vram = _probe_total_vram_bytes()
    if total_vram > _VRAM_FORCE_INT4_THRESHOLD_BYTES:
        print(
            f"[cli] --chroma-dir set + detected {total_vram / 1024**3:.1f} GB "
            f"VRAM → keeping FP16 (auto-quantize skipped; pass --quantize int4 "
            f"to override or --no-auto-quantize to silence)."
        )
        return None
    print(
        "[cli] --chroma-dir present + --quantize unset + <30 GB VRAM → "
        "forcing INT4 (embedder + gen FP16 would OOM). Pass --quantize "
        "fp16 / int8 or --no-auto-quantize to override."
    )
    return "int4"


def _probe_total_vram_bytes() -> int:
    """Best-effort GPU memory query. 0 = no CUDA / probe failed."""
    try:
        import torch  # type: ignore
        if not torch.cuda.is_available():
            return 0
        props = torch.cuda.get_device_properties(0)
        return int(props.total_memory)
    except Exception:
        return 0


def _warn_hidden_cross_view(db_path: str, cross_view_flag: str) -> None:
    """M3.6.B (PROBLEMS P1-07): when the user passes `--cross-view off`
    the cross-view tools (get/count_cross_view_links, find_across_views)
    are hidden from the LLM, but the DuckDB rows are NOT cleared. If the
    DB still has rows, emit a one-line warning that names the count so
    the user doesn't think "my links got deleted". No-op when
    `--cross-view auto` or when the DB is empty/missing."""
    if cross_view_flag != "off":
        return
    if db_path == ":memory:" or not Path(db_path).exists():
        return
    probe = WorldStateStore(db_path=db_path)
    try:
        n_links = len(probe.query_cross_view_links())
    finally:
        probe.close()
    if n_links > 0:
        print(
            f"[warn] --cross-view off but DB has {n_links} cross_view_links "
            f"rows. Tools (get/count_cross_view_links, find_across_views) "
            f"hidden from the LLM; data preserved. Pass --cross-view auto "
            f"to expose them."
        )


def _check_db_populated(db_path: str) -> bool:
    """Quick sanity check at REPL launch.

    Returns False when the situation is fatal (caller should exit
    non-zero); True otherwise.

    M3.8 (PROBLEMS P2-07): missing DB file is now FATAL rather than a
    silent "create empty + warn". Previously the user saw a warning,
    every query returned empty, and `mva ask` happily burned Qwen
    tokens on a doomed plan. Fail loudly so the user catches the
    typo / ingest gap immediately.

    Soft-warn (return True) cases retained:
    - DB exists but no segments (M2.8) AND no tracklets_<view>
      tables (M2.7 legacy) → most likely forgot `mva ingest`
    - DB is M2.7-shaped (segments empty but legacy tables present) →
      segment-level tools will be empty; informational
    """
    if db_path == ":memory:":
        return True
    if not Path(db_path).exists():
        print(
            f"[fatal] DB not found at {db_path}. Run "
            f"`mva ingest --db-path {db_path} --dataset <name> --scene <id> ...` "
            f"to populate it, or pass --db-path :memory: for an empty "
            f"in-memory store."
        )
        return False
    # Probe via a cheap read-only WorldStateStore
    probe = WorldStateStore(db_path=db_path)
    try:
        n_segments = len(probe.query_segments())
        # Legacy tracklets tables — pull from the in-memory _known_views set
        n_legacy_views = len(probe._known_views)
    finally:
        probe.close()
    if n_segments == 0 and n_legacy_views == 0:
        print(f"[warn] DB at {db_path} has no segments (M2.8) and no "
              f"tracklets_<view> tables (M2.7 legacy). Run `mva ingest` first.")
    elif n_segments == 0 and n_legacy_views > 0:
        print(f"[info] DB at {db_path} is M2.7-shaped ({n_legacy_views} legacy "
              f"view(s)). Segment-level tools will be empty; "
              f"`find_segment_by_description` falls back gracefully.")
    return True


def _print_result(result: OrchestratorResult) -> None:
    print("\n========================================")
    print(f"Q: {result.question}")
    print("----------------------------------------")
    print(f"[plan] intent={result.plan.intent}  "
          f"({len(result.plan.tool_calls)} tool call(s))")
    if result.plan.rationale:
        print(f"[rationale] {result.plan.rationale}")
    for inv in result.invocations:
        if inv.error:
            print(f"  ✗ {inv.tool}({inv.args}) → ERROR: {inv.error}")
        else:
            payload = repr(inv.result)
            if len(payload) > 300:
                payload = payload[:300] + "..."
            print(f"  ✓ {inv.tool}({inv.args}) → {payload}")
    if result.needs_disambiguation:
        print("[disambig] close candidates:")
        for c in result.disambiguation_candidates:
            print(f"  - {c}")
    print(f"\nA: {result.answer}")
