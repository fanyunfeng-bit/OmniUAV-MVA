import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PyQt5 import QtWidgets
from tabs.retrieval_tab import RetrievalTab

_app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


class _FakeClient:
    def retrieve(self, **k):
        return {"hits": [
            {"view_id": "view1", "t": 0.0, "score": 0.91, "class_name": None,
             "doc": "view1 [0.0-10.0s]", "thumbnail_path": None},
            {"view_id": "view2", "t": 10.0, "score": 0.80, "doc": "view2 [10-20s]"},
            {"view_id": "view1", "t": 20.0, "score": 0.72, "doc": "view1 [20-30s]"},
        ], "n_vectors_searched": 28}


def test_render_top3_and_transparency():
    tab = RetrievalTab(_FakeClient())
    tab.render(_FakeClient().retrieve())
    assert tab.results_list.count() == 3            # top-3 命中
    assert "28" in tab.transparency_label.text()    # 透明化:查了28个向量
