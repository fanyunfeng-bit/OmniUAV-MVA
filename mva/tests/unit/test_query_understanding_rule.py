from mva.service.query_understanding import RuleBasedConstraintParser, QueryConstraints


P = RuleBasedConstraintParser()


def test_view_and_residual_basic():
    c = P.parse("视角1里的黄车")
    assert c.view_ref == "1"
    assert c.semantic_text == "黄车"
    assert c.time_start is None and c.time_end is None
    assert c.source == "rule"
    assert c.has_constraint is True


def test_view_english_and_drone_forms():
    assert P.parse("view3 的船").view_ref == "3"
    assert P.parse("cam04 里的人").view_ref == "4"     # 前导0
    assert P.parse("第2个无人机").view_ref == "2"
    assert P.parse("3号镜头的卡车").view_ref == "3"


def test_time_first_n_and_range_and_point():
    assert (P.parse("前10秒的船").time_start,
            P.parse("前10秒的船").time_end) == (0.0, 10.0)
    r = P.parse("第5秒到第15秒的车")
    assert (r.time_start, r.time_end, r.relative_to_end) == (5.0, 15.0, False)
    p = P.parse("第30秒的人")
    assert (p.time_start, p.time_end) == (30.0, 30.0)


def test_time_relative_to_end():
    c = P.parse("最后20秒的红色卡车")
    assert c.relative_to_end is True
    assert (c.time_start, c.time_end) == (20.0, 0.0)
    assert c.semantic_text == "红色卡车"


def test_time_anchors():
    a = P.parse("开头那艘船")
    assert (a.time_start, a.time_end, a.relative_to_end) == (0.0, 10.0, False)
    b = P.parse("结尾的车")
    assert b.relative_to_end is True


def test_view_and_time_together():
    c = P.parse("开头那艘船在view2")
    assert c.view_ref == "2"
    assert (c.time_start, c.time_end) == (0.0, 10.0)
    assert "船" in c.semantic_text


def test_plain_query_no_constraint():
    c = P.parse("黄车")
    assert c.view_ref is None
    assert c.time_start is None
    assert c.semantic_text == "黄车"
    assert c.source == "none"
    assert c.has_constraint is False
