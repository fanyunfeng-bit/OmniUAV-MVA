import argparse
import uvicorn
from mva.service.app import create_app
from mva.service.engine import AnalysisEngine


def main() -> None:
    ap = argparse.ArgumentParser("mva.service")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8900)
    ap.add_argument("--db", required=True, help="DuckDB 世界状态库路径")
    ap.add_argument("--chroma-dir", default=None, help="ChromaDB 目录")
    ap.add_argument("--embedder-model", default="Qwen/Qwen3-VL-Embedding-8B")
    ap.add_argument("--device", default=None)
    ap.add_argument("--qa-model", default="qwen3-vl-plus", help="云端问答模型")
    args = ap.parse_args()

    from mva.l4_llm.cloud_client import DashScopeLLMClient
    llm = DashScopeLLMClient(model=args.qa_model)       # key 从 DASHSCOPE_API_KEY 读
    engine = AnalysisEngine(db_path=args.db, chroma_dir=args.chroma_dir,
                            embedder_model=args.embedder_model, device=args.device, llm=llm)
    uvicorn.run(create_app(engine), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
