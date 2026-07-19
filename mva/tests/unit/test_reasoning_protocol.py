from mva.reasoning import EventDetector, TrajectoryPredictor, RelationModeler
from mva.reasoning.fakes import NullEventDetector, ConstantVelocityPredictor
from mva.contracts import GlobalTrajectory, GlobalPrediction


def test_reexports_relation_modeler():
    assert RelationModeler is not None            # 从既有 perception.relation 转出口


def test_fakes_satisfy_protocols():
    assert isinstance(NullEventDetector(), EventDetector)
    assert isinstance(ConstantVelocityPredictor(), TrajectoryPredictor)


def test_null_event_detector_empty():
    assert NullEventDetector().detect([], (0.0, 10.0)) == []


def test_cv_predictor_extrapolates():
    traj = [GlobalTrajectory(global_id="g1", t=0.0, x=0.0, y=0.0),
            GlobalTrajectory(global_id="g1", t=1.0, x=2.0, y=0.0)]
    preds = ConstantVelocityPredictor().predict(traj, horizon_s=1.0)
    assert len(preds) == 1
    assert isinstance(preds[0], GlobalPrediction)
    assert abs(preds[0].x - 4.0) < 1e-6 and abs(preds[0].y - 0.0) < 1e-6
