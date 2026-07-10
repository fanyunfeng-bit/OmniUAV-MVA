from mva.service.models import (
    HealthResponse, IngestRequest, IngestStartResponse, IngestStatusResponse,
    AnswerRequest, AnswerResponse, Grounding,
)


class FakeEngine:
    """内存假引擎，满足 EngineProtocol，供 service 端点单测(无 GPU/网络)。"""
    def __init__(self):
        self.answers = {}
        self.jobs = {}
        self._n = 0

    def health(self) -> HealthResponse:
        return HealthResponse(engine_ready=True, db_path="/tmp/fake.duckdb")

    def ingest_start(self, req: IngestRequest) -> IngestStartResponse:
        self._n += 1
        jid = f"job{self._n}"
        self.jobs[jid] = IngestStatusResponse(job_id=jid, state="running")
        return IngestStartResponse(job_id=jid)

    def ingest_status(self, job_id: str) -> IngestStatusResponse:
        return self.jobs[job_id]

    def ingest_stop(self, job_id: str) -> None:
        self.jobs[job_id] = IngestStatusResponse(job_id=job_id, state="done")

    def answer(self, req: AnswerRequest) -> AnswerResponse:
        return self.answers.get(
            req.query,
            AnswerResponse(answer=f"echo:{req.query}",
                           groundings=[Grounding(view_id="view1", t=1.0)]),
        )

    def retrieve(self, req):
        from mva.service.models import RetrieveResponse, RetrieveHit
        return RetrieveResponse(
            hits=[RetrieveHit(view_id="view1", t=0.0, segment_idx=0, score=0.9,
                              kind="segment", thumbnail_path="/tmp/thumb.jpg")],
            n_vectors_searched=28,
        )
