"""多视角检索面板：文字查询 → top-3 命中 + top-1 缩略图 + 透明化；点击命中跳帧。"""
from PyQt5 import QtCore, QtGui, QtWidgets


class RetrievalTab(QtWidgets.QWidget):
    jump_requested = QtCore.pyqtSignal(str, float)   # (view_id, t_sec)

    def __init__(self, mva_client, parent=None):
        super().__init__(parent)
        self.mva_client = mva_client
        self._build_ui()

    def _build_ui(self):
        root = QtWidgets.QVBoxLayout(self)
        bar = QtWidgets.QHBoxLayout()
        self.query_edit = QtWidgets.QLineEdit()
        self.query_edit.setPlaceholderText("检索:如 airplane / 一艘船 …")
        self.query_edit.returnPressed.connect(self._do_search)
        self.search_btn = QtWidgets.QPushButton("检索")
        self.search_btn.clicked.connect(self._do_search)
        bar.addWidget(self.query_edit)
        bar.addWidget(self.search_btn)
        root.addLayout(bar)

        self.transparency_label = QtWidgets.QLabel("")
        root.addWidget(self.transparency_label)

        body = QtWidgets.QHBoxLayout()
        self.results_list = QtWidgets.QListWidget()          # top-3 文字
        self.results_list.itemClicked.connect(self._on_item_clicked)
        body.addWidget(self.results_list, 3)
        self.thumb_label = QtWidgets.QLabel("(top-1 缩略图)")  # top-1 缩略图
        self.thumb_label.setAlignment(QtCore.Qt.AlignCenter)
        self.thumb_label.setMinimumSize(240, 180)
        self.thumb_label.setStyleSheet("border:1px solid #555;")
        body.addWidget(self.thumb_label, 2)
        root.addLayout(body)

    def _do_search(self):
        q = self.query_edit.text().strip()
        if not q:
            return
        try:
            res = self.mva_client.retrieve(text=q, top_k=3)
        except Exception as e:                               # noqa: BLE001
            self.transparency_label.setText(f"检索失败: {e}")
            return
        self.render(res)

    def render(self, res: dict):
        hits = res.get("hits") or []
        n = res.get("n_vectors_searched", 0)
        self.transparency_label.setText(
            f"检索透明化:查了 {n} 个向量 · 命中 {len(hits)} 条(显示 top-{min(3, len(hits))})"
        )
        self.results_list.clear()
        for i, h in enumerate(hits[:3]):
            cls = f" · {h.get('class_name')}" if h.get("class_name") else ""
            t = h.get("t")
            tstr = f"{t:.1f}s" if isinstance(t, (int, float)) else "?"
            item = QtWidgets.QListWidgetItem(
                f"#{i+1} {h.get('view_id')} @ {tstr}{cls} · 分数 {h.get('score', 0):.2f}\n"
                f"     {h.get('doc') or ''}"
            )
            item.setData(QtCore.Qt.UserRole, (h.get("view_id"), t))
            self.results_list.addItem(item)
        # top-1 缩略图
        thumb = hits[0].get("thumbnail_path") if hits else None
        if thumb:
            pix = QtGui.QPixmap(thumb)
            if not pix.isNull():
                self.thumb_label.setPixmap(
                    pix.scaled(self.thumb_label.size(), QtCore.Qt.KeepAspectRatio,
                               QtCore.Qt.SmoothTransformation))
                return
        self.thumb_label.setText("(top-1 无缩略图)")

    def _on_item_clicked(self, item):
        data = item.data(QtCore.Qt.UserRole)
        if data and data[1] is not None:
            self.jump_requested.emit(str(data[0]), float(data[1]))
