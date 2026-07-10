from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, List, Protocol, Tuple, runtime_checkable

Bbox = Tuple[float, float, float, float]
Detection = Tuple[Bbox, str, float]        # (bbox, class_name, score)


@dataclass
class Track:
    track_id: str
    view_id: str
    t: float
    bbox: Bbox
    class_name: str
    score: float


@runtime_checkable
class Tracker(Protocol):
    def update(self, dets: Iterable[Detection], t: float) -> List[Track]: ...
    def reset(self) -> None: ...


class PassthroughTracker:
    """基线:每个检测独立 id，不做时序关联。占位，后续换 ByteTrack/BoT-SORT(§3.1)。"""
    def __init__(self):
        self._n = 0

    def reset(self) -> None:
        self._n = 0

    def update(self, dets: Iterable[Detection], t: float) -> List[Track]:
        out = []
        for bbox, cls, score in dets:
            self._n += 1
            out.append(Track(track_id=f"t{self._n}", view_id="", t=t,
                             bbox=bbox, class_name=cls, score=score))
        return out


class PerceptionPipeline(ABC):
    @abstractmethod
    def run(self, frame_source, view_id: str, detector, tracker: Tracker) -> List[Track]:
        ...


class DensePerceptionPipeline(PerceptionPipeline):
    """基线:遍历 FrameSource(密集帧)→detector→tracker→Track 列表。
    与嵌入段采样解耦(D10)。detector 需有 .detect(frame)->list[(bbox,cls,score)]。"""
    def run(self, frame_source, view_id: str, detector, tracker: Tracker) -> List[Track]:
        tracker.reset()
        tracks: List[Track] = []
        for t, frame in frame_source.iter_frames():
            dets = detector.detect(frame)
            for tr in tracker.update(dets, t):
                tr.view_id = view_id
                tracks.append(tr)
        return tracks
