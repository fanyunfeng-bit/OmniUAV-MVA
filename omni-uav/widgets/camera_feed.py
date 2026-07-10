from typing import Callable, Optional, List, Tuple

from PyQt5 import QtCore, QtGui, QtWidgets


class CameraFeedWidget(QtWidgets.QFrame):
    # Signal emitted when a detection box is clicked: (uav_id, box_index)
    detection_clicked = QtCore.pyqtSignal(str, int)
    # Signal emitted when widget is clicked with no video loaded: (uav_id)
    load_video_requested = QtCore.pyqtSignal(str)
    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.title = title
        self.setFrameShape(QtWidgets.QFrame.Box)
        self.setLineWidth(1)
        # Reduce minimum size to allow more flexible resizing
        self.setMinimumSize(200, 150)  # Reduced from 260x210
        self.setContentsMargins(0, 0, 0, 0)
        self._tick = 0
        self._pixmap = None
        self._pause_callback: Optional[Callable[[str], None]] = None
        self._uav_id: Optional[str] = None
        self._detection_boxes: List[Tuple[int, int, int, int]] = []  # Store detection boxes [x1, y1, x2, y2]
        self._image_scale = 1.0  # Scale factor for displayed image

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        self.image_label = QtWidgets.QLabel()
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)
        self.image_label.setMinimumSize(180, 130)  # Reduced from 240x180
        self.image_label.setStyleSheet("background-color: #1e1e1e;")

        self.caption_label = QtWidgets.QLabel()
        self.caption_label.setAlignment(QtCore.Qt.AlignCenter)
        self.caption_label.setWordWrap(True)

        layout.addWidget(self.image_label, 1)
        bottom_row = QtWidgets.QHBoxLayout()
        bottom_row.addWidget(self.caption_label, 1)
        self.pause_btn = QtWidgets.QPushButton("暂停")
        self.pause_btn.setFixedWidth(72)
        self.pause_btn.clicked.connect(self._on_pause_clicked)
        bottom_row.addWidget(self.pause_btn, 0)
        layout.addLayout(bottom_row, 0)

    def update_frame(self, text: str):
        self._tick += 1
        hue = (self._tick * 23) % 360
        color = QtGui.QColor.fromHsl(hue, 160, 80)
        self._pixmap = None
        self.image_label.setPixmap(QtGui.QPixmap())
        self.image_label.setStyleSheet(
            f"background-color: {color.name()}; color: white;"
        )
        self.caption_label.setText(text)

    def set_image(self, pixmap: QtGui.QPixmap, caption: str):
        self._pixmap = pixmap
        self.image_label.setStyleSheet("background-color: transparent;")
        self.caption_label.setText(caption)
        self._refresh_pixmap()

    def set_frame(self, image: QtGui.QImage, caption: str):
        self.set_image(QtGui.QPixmap.fromImage(image), caption)

    def update_caption(self, caption: str):
        self.caption_label.setText(caption)

    def set_pause_callback(self, uav_id: str, callback: Callable[[str], None]):
        self._uav_id = uav_id
        self._pause_callback = callback

    def set_pause_state(self, paused: bool):
        self.pause_btn.setText("继续" if paused else "暂停")

    def set_detection_boxes(self, boxes: List[Tuple[int, int, int, int]]):
        """Store detection boxes for click handling."""
        self._detection_boxes = boxes

    def mousePressEvent(self, event):
        """Handle mouse clicks to select detection boxes or load video."""
        if event.button() == QtCore.Qt.LeftButton:
            # If no video loaded, emit signal to load video
            if not self._pixmap and self._uav_id:
                self.load_video_requested.emit(self._uav_id)
                return

            # Handle detection box clicks
            if self._detection_boxes and self._uav_id:
                # Get click position relative to image label
                click_pos = self.image_label.mapFrom(self, event.pos())

                # Get image label size and actual image size
                label_size = self.image_label.size()
                if not self._pixmap:
                    return

                pixmap_size = self._pixmap.size()

                # Calculate scale factor (image is scaled to fit label while keeping aspect ratio)
                scale_x = label_size.width() / pixmap_size.width()
                scale_y = label_size.height() / pixmap_size.height()
                scale = min(scale_x, scale_y)

                # Calculate actual displayed image size
                display_w = int(pixmap_size.width() * scale)
                display_h = int(pixmap_size.height() * scale)

                # Calculate offset (image is centered in label)
                offset_x = (label_size.width() - display_w) // 2
                offset_y = (label_size.height() - display_h) // 2

                # Convert click position to image coordinates
                img_x = (click_pos.x() - offset_x) / scale
                img_y = (click_pos.y() - offset_y) / scale

                # Check if click is inside any detection box
                for idx, (x1, y1, x2, y2) in enumerate(self._detection_boxes):
                    if x1 <= img_x <= x2 and y1 <= img_y <= y2:
                        # Emit signal with UAV ID and box index
                        self.detection_clicked.emit(self._uav_id, idx)
                        break

    def _refresh_pixmap(self):
        if not self._pixmap:
            return
        scaled = self._pixmap.scaled(
            self.image_label.size(),
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation,
        )
        self.image_label.setPixmap(scaled)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh_pixmap()

    def _on_pause_clicked(self):
        if self._pause_callback and self._uav_id:
            self._pause_callback(self._uav_id)
