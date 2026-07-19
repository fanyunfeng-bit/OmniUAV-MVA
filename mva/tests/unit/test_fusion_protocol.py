from mva.fusion import CrossViewAssociator, Triangulator, GlobalTracker
from mva.fusion.fakes import SingletonAssociator, CentroidTriangulator, CountingGlobalTracker
from mva.contracts import Ray, WorldPoint, GlobalObject


def test_fakes_satisfy_protocols():
    assert isinstance(SingletonAssociator(), CrossViewAssociator)
    assert isinstance(CentroidTriangulator(), Triangulator)
    assert isinstance(CountingGlobalTracker(), GlobalTracker)


def test_singleton_associator_one_group_per_obs():
    groups = SingletonAssociator().associate(["a", "b", "c"], geometry=None)
    assert groups == [["a"], ["b"], ["c"]]


def test_centroid_triangulator_averages_ray_origins():
    r1 = Ray(origin=WorldPoint(x=0, y=0, z=10), direction=(0, 0, -1))
    r2 = Ray(origin=WorldPoint(x=4, y=2, z=10), direction=(0, 0, -1))
    wp = CentroidTriangulator().triangulate([r1, r2])
    assert (wp.x, wp.y, wp.z) == (2.0, 1.0, 10.0)


def test_counting_global_tracker_emits_one_object_per_group():
    objs = CountingGlobalTracker().step([["a", "b"], ["c"]], t=1.0)
    assert len(objs) == 2
    assert all(isinstance(o, GlobalObject) for o in objs)
