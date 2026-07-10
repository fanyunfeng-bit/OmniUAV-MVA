import time
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from utils import VisDroneDetector, ObjectTrackerManager, get_settings
from utils.cross_camera_tracker import CrossCameraTracker
from widgets import CameraFeedWidget, StreamBase, VideoStream, ImageSequenceStream, RosbagStream, ROSLiveStream, ROS_AVAILABLE, TrackCropsSubscriber, ROSTrackingPublisher
# [MOD 2026-07-09 | 步骤1 真·实时接入] 主机无 rospy 时，用 rosbridge+roslibpy 实时订阅
from widgets import RosBridgeLiveStream, ROSBRIDGE_AVAILABLE


from enum import Enum


class SourceType(Enum):
    LOCAL = "本地文件"
    ROSBAG = "Rosbag"
    ROS_LIVE = "ROS实时"


@dataclass
class CameraSource:
    uav_id: str
    camera_id: str


class MultiUavCameraTab(QtWidgets.QWidget):
    # [MOD 2026-07-10 | ingest触发] 请求把当前文件夹当一个 scene 入库到 MVA。参数:(dataset_root, scene)
    ingest_requested = QtCore.pyqtSignal(str, str)

    def __init__(
        self,
        data_dir: Path,
        on_pause_frame: Optional[Callable[[str, QtGui.QImage], None]] = None,
        frame_buffer=None,
        system_output=None,
        parent=None,
    ):
        super().__init__(parent)
        self.data_dir = data_dir
        self.uav_ids = ["无人机-01", "无人机-02", "无人机-03", "无人机-04"]
        self.camera_ids = ["前视", "俯视", "左视", "右视"]
        self.uav_cam_map = {
            "无人机-01": "cam01",
            "无人机-02": "cam02",
            "无人机-03": "cam03",
            "无人机-04": "cam04",
        }
        self.system_output = system_output  # System output widget for logging
        self.video_streams: Dict[str, StreamBase] = self._init_video_streams()
        self.grid_feeds: Dict[str, CameraFeedWidget] = {}
        self.paused_uavs: Set[str] = set()
        self.paused_frames: Dict[str, QtGui.QImage] = {}
        self.on_pause_frame = on_pause_frame
        self.frame_buffer = frame_buffer  # Frame buffer for multi-frame analysis
        self.detection_enabled = False
        self.detector: Optional[VisDroneDetector] = None
        self.checkpoint_path: Optional[str] = None
        self.tracking_enabled = False
        self.tracker_manager = ObjectTrackerManager()
        self.last_detections: Dict[str, Tuple] = {}  # Store last detections for click selection

        # Cross-camera tracking
        self.cross_camera_tracker = None
        self.cross_camera_enabled = False
        settings = get_settings()
        self.cross_camera_settings = settings.get_category("cross_camera")

        # LLM-based tracking filter: filter trackers by natural language description
        self.tracking_filter_enabled = False
        self.tracking_filter_description = ""  # Description of objects to track (e.g., "red truck")
        self.tracking_filter_target_uavs = []  # Which UAVs to apply filter to (empty = all)
        self.filtered_tracker_ids: Dict[str, Set[int]] = {}  # UAV ID -> set of tracker IDs to show

        # Multi-frame tracking analysis buffer
        self.tracking_frame_buffer = []  # Buffer for 5 frames with tracking data
        self.tracking_buffer_max_frames = 5

        # ROS Live state
        self.source_type = SourceType.LOCAL
        self.ros_connected = False
        self.ros_streams: Dict[str, ROSLiveStream] = {}

        # ROS track crops subscribers and result publishers
        self.ros_track_crops_subscribers: Dict[str, TrackCropsSubscriber] = {}
        self.ros_tracking_publishers: Dict[str, ROSTrackingPublisher] = {}

        self._build_ui()
        self._setup_timer()
        self._refresh_grid()

    def _build_ui(self):
        root = QtWidgets.QHBoxLayout(self)

        control_panel = QtWidgets.QVBoxLayout()
        self.uav_combo = QtWidgets.QComboBox()
        self.uav_combo.addItems(self.uav_ids)
        # Hide camera selection - show all cameras in grid view only
        # self.camera_combo = QtWidgets.QComboBox()
        # self.camera_combo.addItems(self.camera_ids)

        self.view_mode = QtWidgets.QComboBox()
        self.view_mode.addItems(["单视图", "网格视图"])
        self.view_mode.currentIndexChanged.connect(self._toggle_view)

        control_panel.addWidget(QtWidgets.QLabel("无人机"))
        control_panel.addWidget(self.uav_combo)
        # Hide camera selection
        # control_panel.addWidget(QtWidgets.QLabel("镜头"))
        # control_panel.addWidget(self.camera_combo)
        control_panel.addSpacing(10)
        control_panel.addWidget(QtWidgets.QLabel("可视化模式"))
        control_panel.addWidget(self.view_mode)
        control_panel.addSpacing(10)
        control_panel.addWidget(QtWidgets.QLabel("视频/图像源"))

        # Source type selector
        self.source_type_combo = QtWidgets.QComboBox()
        self.source_type_combo.addItems([e.value for e in SourceType])
        # [MOD 2026-07-09 | 步骤1] rospy 不可用时，只要 rosbridge(roslibpy) 可用也允许 ROS 实时
        if not (ROS_AVAILABLE or ROSBRIDGE_AVAILABLE):
            # Disable ROS Live only if neither rospy nor rosbridge is available
            model = self.source_type_combo.model()
            model.item(self.source_type_combo.count() - 1).setEnabled(False)
        self.source_type_combo.currentIndexChanged.connect(self._on_source_type_changed)
        control_panel.addWidget(self.source_type_combo)

        # Local file controls (shown by default)
        self.local_file_widget = QtWidgets.QWidget()
        local_file_layout = QtWidgets.QVBoxLayout(self.local_file_widget)
        local_file_layout.setContentsMargins(0, 0, 0, 0)
        self.video_dir_label = QtWidgets.QLabel(str(self.data_dir))
        self.video_dir_label.setWordWrap(True)
        self.video_dir_btn = QtWidgets.QPushButton("选择视频文件夹")
        self.video_dir_btn.clicked.connect(self._select_video_dir)
        local_file_layout.addWidget(self.video_dir_label)
        local_file_layout.addWidget(self.video_dir_btn)
        # [MOD 2026-07-10 | ingest触发] 可选:把当前文件夹作为一个场景入库到 MVA 分析引擎
        self.ingest_btn = QtWidgets.QPushButton("入库到分析引擎")
        self.ingest_btn.setToolTip(
            "把当前文件夹作为一个场景送入 MVA(检测/跟踪/嵌入)，之后可对它做 grounded 问答。\n"
            "每个视频文件视为一个视角。需先启动 sidecar 引擎。"
        )
        self.ingest_btn.clicked.connect(self._request_ingest)
        local_file_layout.addWidget(self.ingest_btn)
        control_panel.addWidget(self.local_file_widget)

        # ROS Live controls (hidden by default)
        # Compact layout without scroll area for better usability
        self.ros_live_widget = QtWidgets.QWidget()
        ros_live_layout = QtWidgets.QVBoxLayout(self.ros_live_widget)
        ros_live_layout.setContentsMargins(0, 0, 0, 0)
        ros_live_layout.setSpacing(2)

        # ROS master URI configuration (compact)
        uri_row = QtWidgets.QHBoxLayout()
        uri_row.addWidget(QtWidgets.QLabel("ROS URI:"))
        self.ros_master_label = QtWidgets.QLabel(os.getenv("ROS_MASTER_URI", "http://localhost:11311"))
        self.ros_master_label.setStyleSheet("font-size: 8pt; color: gray;")
        uri_row.addWidget(self.ros_master_label)
        uri_row.addStretch(1)
        ros_live_layout.addLayout(uri_row)

        # ROS topic configuration - use a group box for better organization
        topic_group = QtWidgets.QGroupBox("ROS Topics")
        topic_layout = QtWidgets.QVBoxLayout()
        topic_layout.setSpacing(2)
        self.ros_topic_inputs = {}
        for i, uav_id in enumerate(["无人机-01", "无人机-02", "无人机-03", "无人机-04"]):
            drone_num = i + 1
            default_topic = f"/airsim_node/drone{drone_num}/front_center_custom/Scene"
            # Use shorter labels: "D1:", "D2:", etc.
            topic_label = QtWidgets.QLabel(f"{drone_num}:")
            topic_label.setStyleSheet("font-size: 9pt; min-width: 20px;")
            topic_input = QtWidgets.QLineEdit(default_topic)
            topic_input.setStyleSheet("font-size: 9pt;")
            # Connect editingFinished signal to automatically resubscribe when topic changes
            topic_input.editingFinished.connect(lambda uav=uav_id, input=topic_input: self._on_topic_changed(uav, input))
            self.ros_topic_inputs[uav_id] = topic_input
            row = QtWidgets.QHBoxLayout()
            row.addWidget(topic_label)
            row.addWidget(topic_input)
            topic_layout.addLayout(row)
        topic_group.setLayout(topic_layout)
        ros_live_layout.addWidget(topic_group)

        # Detection mode button (toggles between normal and tracked topics)
        self.detection_mode_btn = QtWidgets.QPushButton("检测")
        self.detection_mode_btn.setCheckable(True)
        self.detection_mode_btn.setStyleSheet("font-size: 9pt; padding: 2px 8px;")
        self.detection_mode_btn.clicked.connect(self._toggle_detection_mode)
        ros_live_layout.addWidget(self.detection_mode_btn)

        # ROS connection status and controls (in one row)
        conn_row = QtWidgets.QHBoxLayout()
        self.ros_status_label = QtWidgets.QLabel("ROS: 未连接")
        self.ros_status_label.setStyleSheet("color: red; font-weight: bold; font-size: 9pt;")
        conn_row.addWidget(self.ros_status_label)
        conn_row.addStretch(1)
        self.ros_connect_btn = QtWidgets.QPushButton("连接ROS")
        self.ros_connect_btn.setStyleSheet("font-size: 9pt; padding: 2px 8px;")
        self.ros_connect_btn.clicked.connect(self._toggle_ros_connection)
        conn_row.addWidget(self.ros_connect_btn)
        ros_live_layout.addLayout(conn_row)

        self.ros_live_widget.setVisible(False)
        control_panel.addWidget(self.ros_live_widget)
        control_panel.addSpacing(10)

        # Create hidden container for detection/tracking controls (not shown in UI)
        # These are created to prevent crashes when code references them
        self.hidden_controls = QtWidgets.QWidget()
        hidden_layout = QtWidgets.QVBoxLayout(self.hidden_controls)
        hidden_layout.setContentsMargins(0, 0, 0, 0)
        self.hidden_controls.setVisible(False)

        # Detection controls (hidden)
        self.detection_checkbox = QtWidgets.QCheckBox("启用VisDrone检测")
        self.detection_checkbox.stateChanged.connect(self._toggle_detection)
        hidden_layout.addWidget(self.detection_checkbox)

        # Model checkpoint selection (hidden)
        self.checkpoint_label = QtWidgets.QLabel("未选择模型")
        self.checkpoint_label.setWordWrap(True)
        self.checkpoint_label.setStyleSheet("font-size: 9pt; color: gray;")
        self.checkpoint_btn = QtWidgets.QPushButton("选择模型文件")
        self.checkpoint_btn.clicked.connect(self._select_checkpoint)
        hidden_layout.addWidget(self.checkpoint_label)
        hidden_layout.addWidget(self.checkpoint_btn)

        # Model architecture selection (hidden)
        self.model_combo = QtWidgets.QComboBox()
        self.model_combo.addItems([
            "fasterrcnn_resnet50",
            "fasterrcnn_mobilenet",
            "fcos_resnet50",
            "retinanet_resnet50"
        ])
        hidden_layout.addWidget(self.model_combo)

        # Score threshold (hidden)
        threshold_layout = QtWidgets.QHBoxLayout()
        threshold_layout.addWidget(QtWidgets.QLabel("置信度阈值"))
        self.threshold_spin = QtWidgets.QDoubleSpinBox()
        self.threshold_spin.setRange(0.1, 0.9)
        self.threshold_spin.setSingleStep(0.05)
        self.threshold_spin.setValue(0.5)
        self.threshold_spin.setDecimals(2)
        threshold_layout.addWidget(self.threshold_spin)
        hidden_layout.addLayout(threshold_layout)

        # Object tracking section (hidden)
        self.tracking_checkbox = QtWidgets.QCheckBox("启用目标跟踪")
        self.tracking_checkbox.stateChanged.connect(self._toggle_tracking)
        hidden_layout.addWidget(self.tracking_checkbox)

        self.track_all_btn = QtWidgets.QPushButton("跟踪所有检测")
        self.track_all_btn.clicked.connect(self._track_all_detections)
        hidden_layout.addWidget(self.track_all_btn)

        self.clear_trackers_btn = QtWidgets.QPushButton("清除所有跟踪")
        self.clear_trackers_btn.clicked.connect(self._clear_trackers)
        hidden_layout.addWidget(self.clear_trackers_btn)

        self.tracker_count_label = QtWidgets.QLabel("活跃跟踪: 0")
        hidden_layout.addWidget(self.tracker_count_label)

        # Cross-camera tracking section (hidden)
        self.cross_camera_checkbox = QtWidgets.QCheckBox("启用跨相机跟踪")
        self.cross_camera_checkbox.stateChanged.connect(self._toggle_cross_camera)
        hidden_layout.addWidget(self.cross_camera_checkbox)

        self.global_track_count_label = QtWidgets.QLabel("全局跟踪: 0")
        hidden_layout.addWidget(self.global_track_count_label)

        # LLM tracking filter status (hidden)
        self.tracking_filter_label = QtWidgets.QLabel("未启用")
        self.tracking_filter_label.setStyleSheet("font-size: 9pt; color: gray;")
        self.tracking_filter_label.setWordWrap(True)
        hidden_layout.addWidget(self.tracking_filter_label)

        self.clear_filter_btn = QtWidgets.QPushButton("清除过滤")
        self.clear_filter_btn.clicked.connect(self._clear_tracking_filter)
        self.clear_filter_btn.setEnabled(False)
        hidden_layout.addWidget(self.clear_filter_btn)

        control_panel.addStretch(1)

        self.stack = QtWidgets.QStackedWidget()
        self.single_feed = CameraFeedWidget("单视图")
        self.single_feed.detection_clicked.connect(self._on_detection_clicked)
        self.single_feed.load_video_requested.connect(self._on_load_video_requested)
        self.grid_container = QtWidgets.QWidget()
        self.grid_layout = QtWidgets.QGridLayout(self.grid_container)
        self.grid_layout.setContentsMargins(0, 0, 0, 0)
        self.grid_layout.setSpacing(6)
        self.stack.addWidget(self.single_feed)
        self.stack.addWidget(self.grid_container)

        # Use a horizontal splitter for control panel vs camera feeds
        # This allows both the control panel and camera feeds to be resizable
        internal_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        internal_splitter.setHandleWidth(3)
        internal_splitter.setChildrenCollapsible(True)  # Allow collapsing

        # Wrap control_panel in a widget for the splitter
        control_container = QtWidgets.QWidget()
        control_container.setLayout(control_panel)
        control_container.setMaximumWidth(300)  # Limit control panel max width

        internal_splitter.addWidget(control_container)
        internal_splitter.addWidget(self.stack)
        internal_splitter.setSizes([250, 700])  # Control panel 250px, feeds 700px

        root.addWidget(internal_splitter)

        # [MOD 2026-07-10 | 默认网格视图] 启动即为网格视图 (0=单视图, 1=网格视图；对应 config.json ui.default_view=grid)
        self.view_mode.setCurrentIndex(1)

        # [MOD 2026-07-09 | 步骤1] OMNIUAV_ROS_LIVE=1 时，启动后自动切到"ROS实时"并连接
        # (一键启动真·实时；不设置则无影响，保持默认本地模式)
        if os.getenv("OMNIUAV_ROS_LIVE", "").lower() in ("1", "true", "yes"):
            QtCore.QTimer.singleShot(1000, self._autoconnect_ros_live)

    def _autoconnect_ros_live(self):
        """[MOD 2026-07-09 | 步骤1] 自动切到 ROS 实时源并连接(供 OMNIUAV_ROS_LIVE=1 一键启动)。
        注意：setCurrentIndex 会同步触发 _reload_ros_streams→_disconnect_ros(清理)。
        因此先切源、再用 singleShot 延迟连接，避免"连上后又被切源清理关闭"的竞态。"""
        try:
            idx = self.source_type_combo.findText(SourceType.ROS_LIVE.value)
            if idx >= 0 and self.source_type_combo.currentIndex() != idx:
                self.source_type_combo.setCurrentIndex(idx)
            # 延迟连接，确保在切源清理完成之后再建立实时连接
            QtCore.QTimer.singleShot(400, self._connect_ros)
        except Exception as e:
            print(f"[ROS-live autoconnect] failed: {e}")

    def _setup_timer(self):
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(40)

    def _switch_camera(self):
        if self.view_mode.currentIndex() == 0:
            self._update_single_feed()
        else:
            self._refresh_grid()

    def _toggle_view(self):
        self.stack.setCurrentIndex(self.view_mode.currentIndex())
        self._switch_camera()

    def _update_single_feed(self):
        source = CameraSource(
            uav_id=self.uav_combo.currentText(),
            camera_id="前视",  # Default to front camera since camera_combo is hidden
        )
        stamp = time.strftime("%H:%M:%S")
        caption = f"{source.uav_id} | {source.camera_id}\n{stamp}"
        stream = self._get_stream_for_uav(source.uav_id)
        self.single_feed.set_pause_callback(source.uav_id, self._toggle_pause)
        self.single_feed.set_pause_state(source.uav_id in self.paused_uavs)
        if source.uav_id in self.paused_uavs:
            paused_image = self.paused_frames.get(source.uav_id)
            if paused_image:
                self.single_feed.set_frame(paused_image, caption)
            return
        if source.camera_id == "前视" and stream:
            image = stream.get_latest() or stream.read()
            if image:
                # Store frame in buffer for multi-frame analysis
                if self.frame_buffer is not None:
                    self.frame_buffer.add_frame(source.uav_id, image)

                # Apply detection and tracking if enabled
                if self.detection_enabled or self.tracking_enabled:
                    image = self._process_frame_with_detection(image, source.uav_id, self.single_feed)
                self.single_feed.set_frame(image, caption)
                return
        self.single_feed.update_frame(caption)

    def _refresh_grid(self):
        for i in reversed(range(self.grid_layout.count())):
            item = self.grid_layout.takeAt(i)
            widget = item.widget()
            if widget:
                widget.deleteLater()

        self.grid_feeds.clear()
        idx = 0
        for uav_id in self.uav_ids:
            cam_id = "前视"
            feed = CameraFeedWidget(f"{uav_id} | {cam_id}")
            caption = f"{uav_id} | {cam_id}"
            feed.update_frame(caption)
            row = idx // 2
            col = idx % 2
            self.grid_layout.addWidget(feed, row, col)
            feed.set_pause_callback(uav_id, self._toggle_pause)
            feed.set_pause_state(uav_id in self.paused_uavs)
            feed.detection_clicked.connect(self._on_detection_clicked)
            feed.load_video_requested.connect(self._on_load_video_requested)
            self.grid_feeds[uav_id] = feed
            idx += 1
        self._tick()

    def _tick(self):
        self._advance_streams()
        if self.view_mode.currentIndex() == 0:
            self._update_single_feed()
            return
        stamp = time.strftime("%H:%M:%S")
        for uav_id, widget in self.grid_feeds.items():
            stream = self._get_stream_for_uav(uav_id)
            caption = f"{uav_id} | 前视\n{stamp}"
            widget.set_pause_state(uav_id in self.paused_uavs)
            if uav_id in self.paused_uavs:
                paused_image = self.paused_frames.get(uav_id)
                if paused_image:
                    widget.set_frame(paused_image, caption)
                else:
                    widget.update_caption(caption)
                continue
            if stream:
                image = stream.get_latest()
                if image:
                    # [MOD 2026-07-10 | 帧率对齐] 仅当本 tick 该流真正前进了才入缓冲，避免节流时重复帧灌满 buffer
                    cam_id = self.uav_cam_map.get(uav_id)
                    if self.frame_buffer is not None and cam_id in getattr(self, "_advanced_cams", set()):
                        self.frame_buffer.add_frame(uav_id, image)

                    # Apply detection and tracking if enabled
                    if self.detection_enabled or self.tracking_enabled:
                        image = self._process_frame_with_detection(image, uav_id, widget)
                    widget.set_frame(image, caption)
                    continue
            widget.update_frame(caption)

    def _advance_streams(self):
        # [MOD 2026-07-10 | 帧率对齐] 记录本 tick 哪些流真正前进了(read 返回非 None)，供 _tick 决定是否入缓冲
        self._advanced_cams = set()
        for cam_id, stream in self.video_streams.items():
            try:
                if stream.read() is not None:
                    self._advanced_cams.add(cam_id)
            except Exception:  # noqa: BLE001
                pass

    def _get_stream_for_uav(self, uav_id: str) -> Optional[StreamBase]:
        cam_id = self.uav_cam_map.get(uav_id)
        if not cam_id:
            return None
        return self.video_streams.get(cam_id)

    def _init_video_streams(self) -> Dict[str, StreamBase]:
        base_dir = self.data_dir
        streams: Dict[str, StreamBase] = {}

        # Check for rosbag file
        bag_files = list(base_dir.glob("*.bag"))
        if bag_files:
            bag_path = bag_files[0]
            print(f"Found rosbag file: {bag_path}")

            # Define topic mapping for each drone
            topic_mapping = {
                "cam01": "/airsim_node/drone1/front_center_custom/Scene",
                "cam02": "/airsim_node/drone2/front_center_custom/Scene",
                "cam03": "/airsim_node/drone3/front_center_custom/Scene",
                "cam04": "/airsim_node/drone4/front_center_custom/Scene",
            }

            try:
                for cam_id, topic in topic_mapping.items():
                    streams[cam_id] = RosbagStream(bag_path, topic, cam_id)
                print(f"Successfully loaded {len(streams)} streams from rosbag")
                return streams
            except Exception as e:
                print(f"Error loading rosbag: {e}")
                print("Falling back to video/image files...")

        # Check for image sequence in rgb/ directory
        rgb_dir = base_dir / "rgb"
        has_rgb = rgb_dir.exists() and (
            any(rgb_dir.glob("*.jpg"))
            or any(rgb_dir.glob("*.jpeg"))
            or any(rgb_dir.glob("*.png"))
        )
        if has_rgb:
            for cam_id in ["cam01", "cam02", "cam03", "cam04"]:
                streams[cam_id] = ImageSequenceStream(rgb_dir, cam_id)
            return streams

        # Try to find video files with various naming patterns
        video_files = self._find_video_files(base_dir)
        if len(video_files) >= 1:
            # Map found videos to cam01-cam04 (up to 4 videos)
            for i, video_path in enumerate(video_files[:4]):
                cam_id = f"cam{i+1:02d}"
                streams[cam_id] = VideoStream(video_path, cam_id)
        return streams

    def _find_video_files(self, directory: Path) -> list:
        """Find video files in directory and sort them by naming pattern.

        Supports patterns like:
        - cam01.mp4, cam02.mp4, ...
        - drone01.mp4, drone02.mp4, ...
        - drone1.mp4, drone2.mp4, ...
        - 1.mp4, 2.mp4, ...
        - Any other video files (sorted alphabetically)
        """
        import re

        # Check if directory exists
        if not directory.exists() or not directory.is_dir():
            return []

        # Video extensions to look for
        video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".m4v"}

        # Get all video files
        video_files = []
        for file in directory.iterdir():
            if file.is_file() and file.suffix.lower() in video_extensions:
                video_files.append(file)

        if not video_files:
            return []

        # Try to extract numeric pattern from filename for sorting
        def extract_number(filename: str) -> tuple:
            """Extract number(s) from filename for sorting.
            Returns (prefix, number, suffix) tuple.
            """
            # Pattern 1: cam01, drone02, etc.
            match = re.search(r'(cam|drone|video)?(\d+)', filename.lower())
            if match:
                prefix = match.group(1) or ""
                number = int(match.group(2))
                return (0, prefix, number)  # 0 = has pattern

            # Pattern 2: just a number at start
            match = re.search(r'^(\d+)', filename.lower())
            if match:
                number = int(match.group(1))
                return (0, "", number)

            # No pattern found - sort alphabetically
            return (1, filename.lower(), 0)

        # Sort by extracted pattern
        video_files.sort(key=lambda f: extract_number(f.stem))

        return video_files

    def _toggle_pause(self, uav_id: str):
        if uav_id in self.paused_uavs:
            self.paused_uavs.remove(uav_id)
            self.paused_frames.pop(uav_id, None)
            return
        self.paused_uavs.add(uav_id)
        stream = self._get_stream_for_uav(uav_id)
        if not stream:
            return
        image = stream.get_latest() or stream.read()
        if image:
            self.paused_frames[uav_id] = image
            if self.on_pause_frame:
                self.on_pause_frame(uav_id, image)

    def _select_video_dir(self):
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self, "选择视频文件夹", str(self.data_dir)
        )
        if not directory:
            return
        new_dir = Path(directory)
        rgb_dir = new_dir / "rgb"
        has_rgb = rgb_dir.exists() and (
            any(rgb_dir.glob("*.jpg"))
            or any(rgb_dir.glob("*.jpeg"))
            or any(rgb_dir.glob("*.png"))
        )
        # Check for video files with various naming patterns
        video_files = self._find_video_files(new_dir)
        has_videos = len(video_files) >= 1

        if not (has_rgb or has_videos):
            QtWidgets.QMessageBox.warning(
                self,
                "缺少数据",
                "文件夹需要包含 rgb/ 目录或至少1个视频文件 (支持 .mp4, .avi, .mov, .mkv 等)。",
            )
            return

        # Show info about found videos
        if has_videos and not has_rgb:
            video_names = [f.name for f in video_files[:4]]
            if len(video_files) > 4:
                video_names.append(f"... (+{len(video_files)-4} more)")
            print(f"Found {len(video_files)} video(s): {', '.join(video_names)}")

        self._set_data_dir(new_dir)

    def _request_ingest(self):
        # [MOD 2026-07-10 | ingest触发] 当前文件夹当一个 pcl-sim scene: root=父目录, scene=目录名
        d = Path(self.data_dir)
        if not self._find_video_files(d):
            QtWidgets.QMessageBox.warning(
                self, "无可入库视频", "当前文件夹没有可入库的视频文件(每个视频=一个视角)。"
            )
            return
        self.ingest_requested.emit(str(d.parent), d.name)

    def _on_source_type_changed(self, index: int):
        """Handle source type combo box change."""
        # Get the source type from the index instead of currentText
        # This ensures the correct type is selected even when called programmatically
        source_types = list(SourceType)
        if 0 <= index < len(source_types):
            self.source_type = source_types[index]
        else:
            # Fallback to current text if index is out of range
            source_type_name = self.source_type_combo.currentText()
            self.source_type = SourceType(source_type_name)

        # Show/hide appropriate controls
        self.local_file_widget.setVisible(self.source_type == SourceType.LOCAL)
        self.ros_live_widget.setVisible(self.source_type == SourceType.ROS_LIVE)

        # Reinitialize streams based on new source type
        if self.source_type == SourceType.LOCAL:
            self._disconnect_ros()
            self._reload_local_streams()
        elif self.source_type == SourceType.ROS_LIVE:
            self._reload_ros_streams()

        # Force layout update to ensure proper resizing
        self.updateGeometry()

    def _toggle_ros_connection(self):
        """Toggle ROS connection on/off."""
        if self.ros_connected:
            self._disconnect_ros()
        else:
            self._connect_ros()

    def _connect_ros(self):
        """Connect to ROS and initialize ROS live streams."""
        # [MOD 2026-07-09 | 步骤1] 允许在仅有 rosbridge(roslibpy) 时连接
        if not (ROS_AVAILABLE or ROSBRIDGE_AVAILABLE):
            QtWidgets.QMessageBox.critical(
                self,
                "ROS不可用",
                "需要 rospy 或 roslibpy(rosbridge) 其一。\n"
                "主机可: pip install roslibpy，并在 ROS 侧运行 rosbridge_server"
            )
            return

        try:
            # Get ROS topics from UI inputs
            topics = {}
            for uav_id, topic_input in self.ros_topic_inputs.items():
                topics[uav_id] = topic_input.text()

            # Initialize ROS live streams
            self.ros_streams = {}
            for uav_id, topic in topics.items():
                cam_id = self.uav_cam_map.get(uav_id)
                if not cam_id:
                    continue
                try:
                    # [MOD 2026-07-09 | 步骤1] 优先 rospy(ROSLiveStream)；主机无 rospy 时
                    # 用 rosbridge+roslibpy(RosBridgeLiveStream) 实现真·实时订阅
                    if ROS_AVAILABLE:
                        stream = ROSLiveStream(topic, cam_id)
                    else:
                        stream = RosBridgeLiveStream(topic, cam_id)
                    self.ros_streams[cam_id] = stream
                except Exception as e:
                    print(f"Failed to connect to topic {topic}: {e}")

            if self.ros_streams:
                self.ros_connected = True
                self.ros_connect_btn.setText("断开ROS")
                self.ros_status_label.setText("ROS状态: 已连接")
                self.ros_status_label.setStyleSheet("color: green; font-weight: bold;")

                # Update video_streams to use ROS streams
                for cam_id, stream in self.ros_streams.items():
                    self.video_streams[cam_id] = stream

                # Refresh the grid to show ROS streams
                self._refresh_grid()

                print(f"Connected to {len(self.ros_streams)} ROS topics")

                # Start debug timer to show ROS stream status (disabled - too verbose)
                # self._start_ros_debug_timer()
            else:
                QtWidgets.QMessageBox.warning(
                    self,
                    "ROS连接失败",
                    "无法连接到任何ROS话题。请检查话题名称和ROS master是否运行。"
                )

        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self,
                "ROS连接错误",
                f"连接ROS时出错:\n{e}"
            )
            import traceback
            traceback.print_exc()

    def _disconnect_ros(self):
        """Disconnect from ROS and cleanup streams."""
        if self.ros_streams:
            for stream in self.ros_streams.values():
                try:
                    stream.close()
                except Exception as e:
                    print(f"Error closing ROS stream: {e}")
            self.ros_streams.clear()

        self.ros_connected = False
        self.ros_connect_btn.setText("连接ROS")
        self.ros_status_label.setText("ROS状态: 未连接")
        self.ros_status_label.setStyleSheet("color: red; font-weight: bold;")

        # Reset detection mode button
        self.detection_mode_btn.setChecked(False)

        # Reset topics to normal mode
        for _, topic_input in self.ros_topic_inputs.items():
            current_topic = topic_input.text()
            if current_topic.endswith("/track_image"):
                normal_topic = current_topic[:-len("/track_image")]
                topic_input.setText(normal_topic)

        # Stop debug timer if running
        if hasattr(self, '_ros_debug_timer') and self._ros_debug_timer:
            self._ros_debug_timer.stop()
            self._ros_debug_timer = None

        # Close track crops subscribers and tracking publishers
        self._close_ros_tracking_components()

    def _init_ros_tracking_components(self, target_drone_ids: List[str]):
        """Initialize track crops subscribers and result publishers for specified drones.

        Args:
            target_drone_ids: List of drone IDs (e.g., ["drone1", "drone2"])
        """
        if not ROS_AVAILABLE:
            return

        # Close existing components first
        self._close_ros_tracking_components()

        # Initialize subscribers and publishers for each target drone
        for drone_id in target_drone_ids:
            try:
                # Create track crops subscriber
                crops_topic = f"/airsim_node/{drone_id}/front_center_custom/Scene/track_crops"
                subscriber = TrackCropsSubscriber(crops_topic, drone_id)
                # Connect signal to notify when crops are received
                subscriber.crops_received.connect(self._on_track_crops_received)
                self.ros_track_crops_subscribers[drone_id] = subscriber

                # Create tracking result publisher (topic is auto-generated)
                publisher = ROSTrackingPublisher(drone_id)
                self.ros_tracking_publishers[drone_id] = publisher

                # print(f"[ROS Tracking] Initialized components for {drone_id}")
            except Exception as e:
                # print(f"[ROS Tracking] Failed to initialize for {drone_id}: {e}")
                pass

    def _on_track_crops_received(self, drone_id: str, crop_count: int):
        """Called when track crops are received from a drone.

        Args:
            drone_id: The drone ID (e.g., "drone1")
            crop_count: Number of crops received
        """
        # Map drone_id to Chinese display name
        drone_names = {
            "drone1": "1号无人机",
            "drone2": "2号无人机",
            "drone3": "3号无人机",
            "drone4": "4号无人机",
        }
        drone_name = drone_names.get(drone_id, drone_id)

        # Log to system output
        self._log_system(f"[ROS跟踪] 已接收到{drone_name}检测结果 ({crop_count} 个裁剪图像)")

    def _close_ros_tracking_components(self):
        """Close all track crops subscribers and tracking publishers."""
        for drone_id, subscriber in self.ros_track_crops_subscribers.items():
            try:
                subscriber.close()
            except Exception as e:
                # print(f"[ROS Tracking] Error closing subscriber for {drone_id}: {e}")
                pass
        self.ros_track_crops_subscribers.clear()

        for drone_id, publisher in self.ros_tracking_publishers.items():
            try:
                publisher.close()
            except Exception as e:
                # print(f"[ROS Tracking] Error closing publisher for {drone_id}: {e}")
                pass
        self.ros_tracking_publishers.clear()

    def get_ros_track_crops(self, drone_id: str) -> List[Dict]:
        """Get the latest track crops for a specific drone.

        Args:
            drone_id: The drone ID (e.g., "drone1")

        Returns:
            List of dicts with 'image' (numpy array) and 'frame_id'
        """
        if drone_id in self.ros_track_crops_subscribers:
            return self.ros_track_crops_subscribers[drone_id].get_latest_crops()
        return []

    def get_ros_track_crops_by_frame(self, drone_id: str) -> List[Dict]:
        """Get all crops from the latest frame (same frame_id) for a specific drone.

        Args:
            drone_id: The drone ID (e.g., "drone1")

        Returns:
            List of dicts with 'image' (numpy array) and 'frame_id'
        """
        if drone_id in self.ros_track_crops_subscribers:
            return self.ros_track_crops_subscribers[drone_id].get_latest_frame_crops()
        return []

    def publish_ros_tracking_result(self, drone_id: str, tracked: bool, ids: List[int]):
        """Publish LLM tracking result to ROS for a specific drone.

        Args:
            drone_id: The drone ID (e.g., "drone1")
            tracked: Whether any objects matched (deprecated, use is_matched)
            ids: List of tracker IDs that matched
        """
        print(f"[PUBLISH] camera_tab.publish_ros_tracking_result called: drone_id={drone_id}, tracked={tracked}, ids={ids}")
        print(f"[PUBLISH] Available publishers: {list(self.ros_tracking_publishers.keys())}")

        if drone_id in self.ros_tracking_publishers:
            # Use new parameter format: is_matched, matched_ids
            self.ros_tracking_publishers[drone_id].publish_result(tracked, ids)
        else:
            print(f"[PUBLISH] ERROR: No publisher found for {drone_id}")

    def parse_drone_ids_from_prompt(self, prompt: str) -> List[str]:
        """Parse drone IDs from user prompt.

        Args:
            prompt: User input string

        Returns:
            List of drone IDs (e.g., ["drone1", "drone2"])
        """
        drone_ids = []
        prompt_lower = prompt.lower()

        # Mapping from various formats to drone IDs
        drone_patterns = {
            "drone1": ["drone1", "uav1", "uav01", "无人机1", "无人机01", "1号无人机", "一号无人机", "1号", "一号"],
            "drone2": ["drone2", "uav2", "uav02", "无人机2", "无人机02", "2号无人机", "二号无人机", "2号", "二号"],
            "drone3": ["drone3", "uav3", "uav03", "无人机3", "无人机03", "3号无人机", "三号无人机", "3号", "三号"],
            "drone4": ["drone4", "uav4", "uav04", "无人机4", "无人机04", "4号无人机", "四号无人机", "4号", "四号"],
        }

        # Check each drone pattern
        for drone_id, patterns in drone_patterns.items():
            for pattern in patterns:
                if pattern in prompt_lower:
                    if drone_id not in drone_ids:
                        drone_ids.append(drone_id)
                    break

        return drone_ids

    def is_ros_mode(self) -> bool:
        """Check if currently in ROS live mode."""
        result = self.source_type == SourceType.ROS_LIVE and self.ros_connected
        print(f"[DEBUG] is_ros_mode: source_type={self.source_type}, ros_connected={self.ros_connected}, result={result}")
        return result

    def _toggle_detection_mode(self):
        """Toggle between normal and tracked image topics."""
        if not self.ros_connected:
            QtWidgets.QMessageBox.warning(
                self,
                "ROS未连接",
                "请先连接ROS才能切换检测模式。"
            )
            self.detection_mode_btn.setChecked(False)
            return

        is_detection_mode = self.detection_mode_btn.isChecked()

        for uav_id, topic_input in self.ros_topic_inputs.items():
            current_topic = topic_input.text()

            if is_detection_mode:
                # Switch to tracked topic
                if not current_topic.endswith("/track_image"):
                    new_topic = current_topic + "/track_image"
                else:
                    new_topic = current_topic
            else:
                # Switch back to normal topic
                if current_topic.endswith("/track_image"):
                    new_topic = current_topic[:-len("/track_image")]
                else:
                    new_topic = current_topic

            # Update the input field
            topic_input.setText(new_topic)

            # Resubscribe the stream
            cam_id = self.uav_cam_map.get(uav_id)
            if cam_id and cam_id in self.ros_streams:
                stream = self.ros_streams[cam_id]
                stream.resubscribe(new_topic)

        mode_str = "检测模式" if is_detection_mode else "正常模式"
        print(f"Switched to {mode_str}")

    def _on_topic_changed(self, uav_id: str, topic_input: QtWidgets.QLineEdit):
        """Handle when user manually changes a ROS topic in the input field.

        Automatically resubscribes to the new topic if ROS is connected.

        Args:
            uav_id: The UAV ID (e.g., "无人机-01")
            topic_input: The QLineEdit widget containing the new topic
        """
        # Only process if ROS is connected
        if not self.ros_connected:
            return

        # Get the new topic from input
        new_topic = topic_input.text().strip()
        if not new_topic:
            return

        # Get the camera ID for this UAV
        cam_id = self.uav_cam_map.get(uav_id)
        if not cam_id or cam_id not in self.ros_streams:
            return

        # Get the current stream
        stream = self.ros_streams[cam_id]
        current_topic = stream.topic

        # Only resubscribe if topic actually changed
        if current_topic == new_topic:
            return

        # Resubscribe to new topic
        success = stream.resubscribe(new_topic)
        if success:
            print(f"[ROS] {uav_id}: Resubscribed from '{current_topic}' to '{new_topic}'")
        else:
            print(f"[ROS] {uav_id}: Failed to resubscribe to '{new_topic}'")

    def _start_ros_debug_timer(self):
        """Start a timer to print ROS stream status for debugging."""
        if hasattr(self, '_ros_debug_timer') and self._ros_debug_timer:
            self._ros_debug_timer.stop()

        self._ros_debug_timer = QtCore.QTimer(self)
        self._ros_debug_timer.timeout.connect(self._print_ros_stream_status)
        self._ros_debug_timer.start(3000)  # Print every 3 seconds

    def _print_ros_stream_status(self):
        """Print current ROS stream status for debugging."""
        if not self.ros_streams:
            return

        for cam_id, stream in self.ros_streams.items():
            frame_count = stream.get_frame_count()
            has_frames = stream.has_frames()
            latest = stream.get_latest()
            latest_info = f"{latest.width()}x{latest.height()}" if latest else "No image"
            print(f"[ROS DEBUG] {cam_id}: frames={frame_count}, has_frames={has_frames}, latest={latest_info}")

    def _reload_local_streams(self):
        """Reload streams from local files."""
        self.video_streams = self._init_video_streams()
        self._refresh_grid()

    def _reload_ros_streams(self):
        """Reload ROS streams (doesn't auto-connect, user must click connect)."""
        # Clear existing ROS streams
        self._disconnect_ros()
        # Don't clear video_streams - keep existing feeds until ROS connects
        # The grid will be updated when ROS successfully connects
        # self._refresh_grid()  # Don't refresh grid to keep existing camera feeds visible

    def _init_ros_streams(self) -> Dict[str, StreamBase]:
        """Initialize ROS live streams from current topic configuration.

        Note: This requires ROS to be connected first via _connect_ros().
        Returns empty dict if not connected.
        """
        streams: Dict[str, StreamBase] = {}
        if not self.ros_connected or not self.ros_streams:
            return streams
        streams.update(self.ros_streams)
        return streams

    def _on_load_video_requested(self, uav_id: str):
        """Handle click on camera feed widget to load a single video."""
        cam_id = self.uav_cam_map.get(uav_id)
        if not cam_id:
            return

        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            f"为 {uav_id} 选择视频文件",
            str(self.data_dir),
            "视频文件 (*.mp4 *.avi *.mov *.mkv);;所有文件 (*.*)"
        )
        if not file_path:
            return

        # Close existing stream for this camera if any
        if cam_id in self.video_streams:
            self.video_streams[cam_id].close()

        # Load the new video
        try:
            self.video_streams[cam_id] = VideoStream(Path(file_path), cam_id)
            # Clear pause state for this UAV
            self.paused_uavs.discard(uav_id)
            self.paused_frames.pop(uav_id, None)
            print(f"Loaded video for {uav_id}: {file_path}")
        except Exception as e:
            QtWidgets.QMessageBox.warning(
                self,
                "加载视频失败",
                f"无法加载视频文件:\n{str(e)}"
            )

    def seek_to(self, cam_id: str, t_sec: float):
        # [MOD 2026-07-10 | P1 跳帧] 检索命中跳转:把该 cam 的视频流 seek 到 t_sec，并切单视图
        stream = self.video_streams.get(cam_id)
        if stream is None:
            print(f"[seek] 无该镜头流: {cam_id}")
            return
        cap = getattr(stream, "cap", None)   # VideoStream.cap = cv2.VideoCapture
        if cap is not None:
            try:
                import cv2
                cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, float(t_sec)) * 1000.0)
            except Exception as e:  # noqa: BLE001
                print(f"[seek] 失败: {e}")
        # 切到单视图看该镜头(view_mode: 0=单视图)
        try:
            self.view_mode.setCurrentIndex(0)
            uav = next(u for u, c in self.uav_cam_map.items() if c == cam_id)
            self.uav_combo.setCurrentIndex(self.uav_ids.index(uav))
        except Exception:  # noqa: BLE001
            pass

    def _set_data_dir(self, new_dir: Path):
        for stream in self.video_streams.values():
            stream.close()
        self.data_dir = new_dir
        self.video_dir_label.setText(str(new_dir))
        self.video_streams = self._init_video_streams()
        self.paused_uavs.clear()
        self.paused_frames.clear()
        self._refresh_grid()

    def _select_checkpoint(self):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "选择模型检查点",
            str(self.data_dir / "checkpoints"),
            "PyTorch Models (*.pth *.pt);;All Files (*.*)"
        )
        if file_path:
            self.checkpoint_path = file_path
            # Display shortened path
            path_obj = Path(file_path)
            display_name = f".../{path_obj.parent.name}/{path_obj.name}"
            self.checkpoint_label.setText(display_name)
            self.checkpoint_label.setStyleSheet("font-size: 9pt; color: green;")
            # Reset detector to force reload with new checkpoint
            if self.detector is not None:
                self.detector = None
                if self.detection_enabled:
                    self._init_detector()

    def _toggle_detection(self, state):
        self.detection_enabled = state == QtCore.Qt.Checked
        if self.detection_enabled and self.detector is None:
            self._init_detector()

    def _init_detector(self):
        try:
            # Get settings from UI
            model_name = self.model_combo.currentText()
            score_threshold = self.threshold_spin.value()

            if self.checkpoint_path:
                print(f"Initializing VisDrone detector with checkpoint: {self.checkpoint_path}")
                print(f"Model: {model_name}, Threshold: {score_threshold}")
            else:
                print(f"Initializing VisDrone detector with pretrained weights")
                print(f"Model: {model_name}, Threshold: {score_threshold}")

            self.detector = VisDroneDetector(
                model_name=model_name,
                checkpoint_path=self.checkpoint_path,
                score_threshold=score_threshold
            )
            print("✓ Detector initialized successfully")
        except Exception as e:
            print(f"Failed to initialize detector: {e}")
            self.detection_enabled = False
            self.detection_checkbox.setChecked(False)
            QtWidgets.QMessageBox.warning(
                self,
                "检测器初始化失败",
                f"无法初始化VisDrone检测器:\n{str(e)}"
            )

    def _toggle_tracking(self, state):
        self.tracking_enabled = state == QtCore.Qt.Checked
        if not self.tracking_enabled:
            self._clear_trackers()

    def _toggle_cross_camera(self, state):
        """Enable/disable cross-camera tracking."""
        self.cross_camera_enabled = state == QtCore.Qt.Checked

        if self.cross_camera_enabled:
            if not self.tracking_enabled:
                QtWidgets.QMessageBox.warning(
                    self,
                    "需要启用跟踪",
                    "跨相机跟踪需要先启用目标跟踪功能。"
                )
                self.cross_camera_checkbox.setChecked(False)
                self.cross_camera_enabled = False
                return

            # Initialize cross-camera tracker
            self.cross_camera_tracker = CrossCameraTracker(
                similarity_threshold=self.cross_camera_settings.get("similarity_threshold", 0.7),
                feature_type=self.cross_camera_settings.get("feature_type", "color_histogram"),
                max_frames_missing=30
            )
            print("Cross-camera tracking enabled")
        else:
            self.cross_camera_tracker = None
            print("Cross-camera tracking disabled")

        self._update_cross_camera_count()

    def _track_all_detections(self):
        """Initialize trackers for all currently detected objects."""
        if not self.tracking_enabled:
            QtWidgets.QMessageBox.warning(
                self,
                "跟踪未启用",
                "请先启用目标跟踪功能。"
            )
            return

        # Get current frame and detections
        if self.view_mode.currentIndex() == 0:
            # Single view mode
            uav_id = self.uav_combo.currentText()
            stream = self._get_stream_for_uav(uav_id)
            if not stream:
                return

            image = stream.get_latest()
            if not image:
                return

            # Convert QImage to numpy array
            width = image.width()
            height = image.height()
            ptr = image.bits()
            ptr.setsize(height * width * 3)
            arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 3))
            frame_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

            # Get detections for this UAV
            if uav_id in self.last_detections:
                boxes, labels, scores = self.last_detections[uav_id]
                self._add_trackers_from_detections(frame_bgr, boxes, labels)
        else:
            # Grid view mode - track detections from all UAVs
            for uav_id in self.uav_ids:
                stream = self._get_stream_for_uav(uav_id)
                if not stream:
                    continue

                image = stream.get_latest()
                if not image:
                    continue

                # Convert QImage to numpy array
                width = image.width()
                height = image.height()
                ptr = image.bits()
                ptr.setsize(height * width * 3)
                arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 3))
                frame_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

                # Get detections for this UAV
                if uav_id in self.last_detections:
                    boxes, labels, scores = self.last_detections[uav_id]
                    self._add_trackers_from_detections(frame_bgr, boxes, labels)

    def _add_trackers_from_detections(self, frame: np.ndarray, boxes: np.ndarray, labels: np.ndarray):
        """Add trackers for detected objects."""
        from utils.visdrone_detector import VISDRONE_CLASSES

        for box, label in zip(boxes, labels):
            x1, y1, x2, y2 = box.astype(int)
            w, h = x2 - x1, y2 - y1

            # Get class name
            class_name = (
                VISDRONE_CLASSES[label]
                if label < len(VISDRONE_CLASSES)
                else f"class_{label}"
            )

            # Initialize tracker
            self.tracker_manager.add_tracker(frame, (x1, y1, w, h), class_name)

        self._update_tracker_count()

    def _clear_trackers(self):
        self.tracker_manager.clear_all()
        self._update_tracker_count()

    def _update_tracker_count(self):
        count = self.tracker_manager.get_active_count()
        self.tracker_count_label.setText(f"活跃跟踪: {count}")

    def _update_cross_camera_count(self):
        """Update the global tracks count label."""
        if self.cross_camera_tracker:
            global_tracks = self.cross_camera_tracker.get_global_tracks()
            count = len(global_tracks)
            self.global_track_count_label.setText(f"全局跟踪: {count}")
        else:
            self.global_track_count_label.setText("全局跟踪: 0")

    def _draw_filtered_trackers(self, frame: np.ndarray, filtered_trackers: Dict[int, any]) -> np.ndarray:
        """Draw only the filtered trackers on frame.

        Args:
            frame: Frame to draw on (BGR format)
            filtered_trackers: Dict of tracker_id to TrackedObject to draw

        Returns:
            Frame with filtered trackers drawn
        """
        result = frame.copy()

        for tracker_id, tracked_obj in filtered_trackers.items():
            x, y, w, h = tracked_obj.bbox
            x1, y1, x2, y2 = x, y, x + w, y + h

            # Draw bounding box
            cv2.rectangle(result, (x1, y1), (x2, y2), tracked_obj.color, 3)

            # Draw label
            label_text = f"Track #{tracked_obj.tracker_id}"
            if tracked_obj.label:
                label_text += f": {tracked_obj.label}"

            # Draw label background
            (text_width, text_height), baseline = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(
                result,
                (x1, y1 - text_height - 8),
                (x1 + text_width, y1),
                tracked_obj.color,
                -1
            )

            # Draw label text
            cv2.putText(
                result,
                label_text,
                (x1, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

        return result

    def _on_detection_clicked(self, uav_id: str, box_index: int):
        """Handle detection box click to start tracking."""
        if not self.tracking_enabled:
            QtWidgets.QMessageBox.information(
                self,
                "跟踪未启用",
                "请先启用目标跟踪功能，然后点击检测框开始跟踪。"
            )
            return

        # Get detections for this UAV
        if uav_id not in self.last_detections:
            return

        boxes, labels, scores = self.last_detections[uav_id]
        if box_index >= len(boxes):
            return

        # Get the clicked detection
        box = boxes[box_index]
        label = labels[box_index]

        # Get current frame for this UAV
        stream = self._get_stream_for_uav(uav_id)
        if not stream:
            return

        image = stream.get_latest()
        if not image:
            return

        # Convert QImage to numpy array
        width = image.width()
        height = image.height()
        ptr = image.bits()
        ptr.setsize(height * width * 3)
        arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 3))
        frame_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

        # Convert box from [x1, y1, x2, y2] to [x, y, w, h]
        x1, y1, x2, y2 = box.astype(int)
        w, h = x2 - x1, y2 - y1

        # Get class name
        from utils.visdrone_detector import VISDRONE_CLASSES
        class_name = (
            VISDRONE_CLASSES[label]
            if label < len(VISDRONE_CLASSES)
            else f"class_{label}"
        )

        # Initialize tracker
        self.tracker_manager.add_tracker(frame_bgr, (x1, y1, w, h), class_name)
        self._update_tracker_count()

        print(f"Started tracking {class_name} from {uav_id}")

    def _process_frame_with_detection(self, qimage: QtGui.QImage, uav_id: str = "", widget: Optional[CameraFeedWidget] = None) -> QtGui.QImage:
        # Convert QImage to numpy array (BGR for OpenCV)
        width = qimage.width()
        height = qimage.height()
        ptr = qimage.bits()
        ptr.setsize(height * width * 3)
        arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 3))
        frame_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

        try:
            # Run detection if enabled
            if self.detection_enabled and self.detector is not None:
                boxes, labels, scores = self.detector.detect(frame_bgr)

                # Store detections for this UAV
                if uav_id:
                    self.last_detections[uav_id] = (boxes, labels, scores)

                # Store detection boxes in widget for click handling
                if widget:
                    detection_boxes = [(int(x1), int(y1), int(x2), int(y2))
                                      for x1, y1, x2, y2 in boxes]
                    widget.set_detection_boxes(detection_boxes)

                # Draw detections
                frame_bgr = self.detector.draw_detections(frame_bgr, boxes, labels, scores)

            # Apply tracking if enabled
            if self.tracking_enabled:
                self.tracker_manager.update(frame_bgr)

                # Apply LLM tracker filter if enabled
                if self.tracking_filter_enabled and uav_id:
                    if uav_id in self.filtered_tracker_ids:
                        # Filter to show only specified trackers
                        allowed_ids = self.filtered_tracker_ids[uav_id]
                        filtered_trackers = {
                            tid: self.tracker_manager.trackers[tid]
                            for tid in allowed_ids
                            if tid in self.tracker_manager.trackers and self.tracker_manager.trackers[tid].active
                        }
                        frame_bgr = self._draw_filtered_trackers(frame_bgr, filtered_trackers)
                    else:
                        # Not affected - show all trackers
                        frame_bgr = self.tracker_manager.draw_trackers(frame_bgr)
                else:
                    # No filter - show all trackers
                    frame_bgr = self.tracker_manager.draw_trackers(frame_bgr)

                self._update_tracker_count()

                # Apply cross-camera tracking if enabled
                if self.cross_camera_enabled and self.cross_camera_tracker and uav_id:
                    # Extract active tracks from tracker_manager
                    active_tracks = self.tracker_manager.get_active_tracks()

                    # Convert to format expected by cross-camera tracker
                    # (tracker_id, bbox, label)
                    local_tracks = []
                    for tracker_id, tracker_info in active_tracks.items():
                        bbox = tracker_info['bbox']  # (x, y, w, h)
                        label = tracker_info.get('label', '')
                        local_tracks.append((tracker_id, bbox, label))

                    # Update cross-camera tracker
                    self.cross_camera_tracker.update(uav_id, frame_bgr, local_tracks)

                    # Draw global IDs on the frame
                    for tracker_id, tracker_info in active_tracks.items():
                        global_id = self.cross_camera_tracker.get_global_id_for_camera_track(
                            uav_id, tracker_id
                        )

                        if global_id is not None:
                            # Get bbox and draw global ID
                            x, y, w, h = tracker_info['bbox']

                            # Get global track color
                            global_tracks = self.cross_camera_tracker.get_global_tracks()
                            if global_id in global_tracks:
                                color = global_tracks[global_id].color
                            else:
                                color = (255, 255, 0)  # Yellow default

                            # Draw global ID below the local tracker ID
                            text = f"G{global_id}"
                            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                            cv2.rectangle(
                                frame_bgr,
                                (x, y + h + 5),
                                (x + text_size[0] + 4, y + h + text_size[1] + 9),
                                color,
                                -1
                            )
                            cv2.putText(
                                frame_bgr,
                                text,
                                (x + 2, y + h + text_size[1] + 7),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (255, 255, 255),
                                2
                            )

                    # Update global track count
                    self._update_cross_camera_count()

            # Convert back to QImage
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            h, w, ch = frame_rgb.shape
            bytes_per_line = ch * w
            result = QtGui.QImage(
                frame_rgb.data,
                w,
                h,
                bytes_per_line,
                QtGui.QImage.Format_RGB888
            ).copy()

            return result
        except Exception as e:
            print(f"Detection/Tracking error: {e}")
            return qimage

    def set_tracking_filter(self, description: str, uav_ids: list = None, filtered_tracker_ids: dict = None) -> bool:
        """Set LLM-based tracking filter to show only matching trackers.

        Args:
            description: Natural language description of objects to track (e.g., "red truck", "person on left")
            uav_ids: List of UAV IDs to apply filter to (None = all UAVs)
            filtered_tracker_ids: Dict mapping UAV ID to set of tracker IDs to show

        Returns:
            True if filter was set successfully
        """
        # Auto-enable detection if not already enabled
        if not self.detection_enabled:
            self.detection_enabled = True
            self.detection_checkbox.setChecked(True)
            if self.detector is None:
                self._init_detector()

        # Auto-enable tracking if not already enabled
        if not self.tracking_enabled:
            self.tracking_enabled = True
            self.tracking_checkbox.setChecked(True)

        self.tracking_filter_enabled = True
        self.tracking_filter_description = description
        self.tracking_filter_target_uavs = uav_ids if uav_ids else []

        # Store filtered tracker IDs for each UAV
        if filtered_tracker_ids:
            self.filtered_tracker_ids = {k: set(v) for k, v in filtered_tracker_ids.items()}
        else:
            self.filtered_tracker_ids = {}

        # Update UI
        uav_str = ", ".join(uav_ids) if uav_ids else "所有无人机"
        match_count = sum(len(tracker_ids) for tracker_ids in self.filtered_tracker_ids.values())
        self.tracking_filter_label.setText(f"过滤: {description}\n应用于: {uav_str}\n匹配: {match_count} 个跟踪器")
        self.tracking_filter_label.setStyleSheet("font-size: 9pt; color: green;")
        self.clear_filter_btn.setEnabled(True)

        self._log_system(f"已设置跟踪过滤: '{description}' (应用于 {uav_str}, 匹配 {match_count} 个跟踪器)")
        return True

    def _clear_tracking_filter(self):
        """Clear the LLM-based tracking filter."""
        self.tracking_filter_enabled = False
        self.tracking_filter_description = ""
        self.tracking_filter_target_uavs = []
        self.filtered_tracker_ids = {}

        # Update UI
        self.tracking_filter_label.setText("未启用")
        self.tracking_filter_label.setStyleSheet("font-size: 9pt; color: gray;")
        self.clear_filter_btn.setEnabled(False)

        self._log_system("已清除跟踪过滤")

    def _log_system(self, message: str):
        """Log message to system output."""
        print(f"[跟踪过滤] {message}")
        if self.system_output:
            self.system_output.appendPlainText(f"[跟踪过滤] {message}")

    def get_latest_frames(self) -> Dict[str, QtGui.QImage]:
        frames = {}
        for uav_id in self.uav_ids:
            stream = self._get_stream_for_uav(uav_id)
            if not stream:
                continue
            image = stream.get_latest() or stream.read()
            # Check for None or null QImage
            if image is not None and not image.isNull():
                frames[uav_id] = image
        return frames

    def _get_tracker_info(self, uav_id: str) -> Dict[str, Dict]:
        """Get current tracking info for a UAV.

        Args:
            uav_id: The UAV ID to get tracker info for

        Returns:
            Dict mapping tracker_id to dict with bbox and label
        """
        tracker_info = {}
        for tracker_id, tracked_obj in self.tracker_manager.trackers.items():
            if tracked_obj.active:
                tracker_info[str(tracker_id)] = {
                    "bbox": list(tracked_obj.bbox),
                    "label": tracked_obj.label
                }
        return tracker_info

    def _collect_tracking_frames(self, target_uavs: List[str]) -> List[Dict]:
        """Collect up to 5 frames with tracking results by auto-advancing video.

        Auto-advances the stream by 5 frames to quickly collect tracking data.

        Args:
            target_uavs: List of UAV IDs to collect frames for

        Returns:
            List of frames with tracking data
        """
        frames_data = []

        for frame_num in range(self.tracking_buffer_max_frames):
            # For each target UAV, get current tracking state
            for uav_id in target_uavs:
                stream = self._get_stream_for_uav(uav_id)
                if not stream:
                    continue

                # Get current frame (check for None or null QImage)
                frame = stream.get_latest()
                if frame is None or frame.isNull():
                    continue

                # Get current tracking info for this UAV
                tracker_info = self._get_tracker_info(uav_id)

                if tracker_info:
                    frame_data = {
                        "frame_number": frame_num + 1,
                        "uav_id": uav_id,
                        "trackers": tracker_info
                    }
                    frames_data.append(frame_data)

            # Auto-advance stream by 1 frame
            self._advance_streams()

        return frames_data
