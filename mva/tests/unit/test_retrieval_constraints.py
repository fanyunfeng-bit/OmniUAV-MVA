from mva.service.retrieval import (
    resolve_view_ref, resolve_time, build_metadata_where,
)
from mva.service.query_understanding import QueryConstraints


def test_resolve_view_cam_and_view_forms():
    assert resolve_view_ref("1", ["cam01", "cam02", "cam03", "cam04"]) == "cam01"
    assert resolve_view_ref("3", ["cam01", "cam02", "cam03", "cam04"]) == "cam03"
    assert resolve_view_ref("2", ["view1", "view2", "view3"]) == "view2"


def test_resolve_view_disambiguates_view1_vs_view11():
    assert resolve_view_ref("1", ["view1", "view11"]) == "view1"


def test_resolve_view_nth_fallback_and_bounds():
    assert resolve_view_ref("1", ["left", "right"]) == "left"      # 无数字→排序第N个
    assert resolve_view_ref("9", ["view1", "view2"]) is None       # 越界
    assert resolve_view_ref("x", ["view1"]) is None                # 非数字
    assert resolve_view_ref("1", []) is None                       # 无 view


def test_resolve_time_absolute_passthrough():
    c = QueryConstraints(time_start=5.0, time_end=15.0, relative_to_end=False)
    assert resolve_time(c, duration=180.0) == (5.0, 15.0)


def test_resolve_time_relative_to_end():
    c = QueryConstraints(time_start=20.0, time_end=0.0, relative_to_end=True)
    assert resolve_time(c, duration=180.0) == (160.0, 180.0)


def test_resolve_time_relative_needs_duration():
    c = QueryConstraints(time_start=20.0, time_end=0.0, relative_to_end=True)
    assert resolve_time(c, duration=None) == (None, None)


def test_resolve_time_none():
    assert resolve_time(QueryConstraints(), duration=180.0) == (None, None)


def test_build_where_view_only():
    assert build_metadata_where("view1", None, None) == {"view_id_raw": "view1"}


def test_build_where_time_only():
    w = build_metadata_where(None, 0.0, 10.0)
    assert w == {"$and": [{"start_t": {"$lte": 10.0}}, {"end_t": {"$gte": 0.0}}]}


def test_build_where_view_and_time():
    w = build_metadata_where("view2", 0.0, 10.0)
    assert w == {"$and": [{"view_id_raw": "view2"},
                          {"start_t": {"$lte": 10.0}},
                          {"end_t": {"$gte": 0.0}}]}


def test_build_where_none():
    assert build_metadata_where(None, None, None) is None
