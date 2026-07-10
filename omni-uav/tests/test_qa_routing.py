from app import decide_qa_route


def test_route_sidecar_when_alive():
    assert decide_qa_route(True) == "sidecar"


def test_route_local_when_dead():
    assert decide_qa_route(False) == "local"
