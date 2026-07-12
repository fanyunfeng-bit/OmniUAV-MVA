import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PyQt5 import QtWidgets
from tabs.retrieval_tab import RetrievalTab

_app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _render(applied):
    tab = RetrievalTab(mva_client=None)
    tab.render({"hits": [], "n_vectors_searched": 5, "applied": applied})
    return tab.transparency_label.text()


def test_shows_view_time_semantic_and_source():
    txt = _render({"view_id": "view1", "time_start": 0.0, "time_end": 10.0,
                   "semantic_text": "黄车", "source": "rule", "fell_back": False})
    assert "视角 view1" in txt
    assert "黄车" in txt
    assert "规则" in txt


def test_shows_fallback_note():
    txt = _render({"view_id": "view1", "time_start": None, "time_end": None,
                   "semantic_text": "飞机", "source": "rule", "fell_back": True})
    assert "无命中" in txt


def test_plain_query_adds_no_applied_line():
    txt = _render({"view_id": None, "time_start": None, "time_end": None,
                   "semantic_text": "黄车", "source": "none", "fell_back": False})
    assert "已限定" not in txt
    assert "5" in txt          # 基础透明化行仍在


def test_missing_applied_is_safe():
    tab = RetrievalTab(mva_client=None)
    tab.render({"hits": [], "n_vectors_searched": 3})    # 无 applied 键
    assert "3" in tab.transparency_label.text()
