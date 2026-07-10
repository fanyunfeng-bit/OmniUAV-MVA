"""Unit tests for the L1 Detector model-family detection (no model load)."""
from mva.l1_perception.detector import Detection, _model_family


def test_model_family_closed_yolo():
    assert _model_family("yolo11n.pt") == "yolo"
    assert _model_family("yolo11m.pt") == "yolo"
    assert _model_family("yolov8s.pt") == "yolo"


def test_model_family_yoloe():
    assert _model_family("yoloe-11l-seg.pt") == "yoloe"
    assert _model_family("yoloe-11s-seg.pt") == "yoloe"


def test_model_family_world():
    assert _model_family("yolov8x-worldv2.pt") == "world"
    assert _model_family("yolov8s-world.pt") == "world"


def test_detection_dataclass_shape():
    d = Detection(bbox=(1.0, 2.0, 3.0, 4.0), class_id=2, class_name="car",
                  confidence=0.9)
    assert d.bbox == (1.0, 2.0, 3.0, 4.0)
    assert d.class_name == "car"
    assert d.confidence == 0.9
