from mva.detection import ObjectDetector, Segmenter
from mva.detection.fakes import NullDetector, NullSegmenter


def test_fakes_satisfy_protocols():
    assert isinstance(NullDetector(), ObjectDetector)
    assert isinstance(NullSegmenter(), Segmenter)


def test_null_detector_returns_empty():
    assert NullDetector().detect(object()) == []


def test_existing_detector_class_satisfies_interface():
    # 现有具体检测器结构上满足 ObjectDetector（类可 import，不实例化=不加载权重）
    from mva.l1_perception import Detector
    assert hasattr(Detector, "detect")
