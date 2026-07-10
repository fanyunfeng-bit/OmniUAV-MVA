"""
Settings dialog for OmniUAV application.
Provides UI for configuring all application settings.
"""
from PyQt5 import QtWidgets, QtCore, QtGui
from utils.settings_manager import get_settings


class SettingsDialog(QtWidgets.QDialog):
    """Settings dialog with tabbed interface."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.settings = get_settings()
        self.setWindowTitle("设置")
        self.setMinimumSize(600, 500)
        self._build_ui()
        self._load_settings()

    def _build_ui(self):
        """Build the dialog UI."""
        layout = QtWidgets.QVBoxLayout(self)

        # Create tab widget
        self.tab_widget = QtWidgets.QTabWidget()
        layout.addWidget(self.tab_widget)

        # Add tabs
        self.tab_widget.addTab(self._create_llm_tab(), "大模型")
        self.tab_widget.addTab(self._create_detection_tab(), "目标检测")
        self.tab_widget.addTab(self._create_tracking_tab(), "目标跟踪")
        self.tab_widget.addTab(self._create_cross_camera_tab(), "跨相机跟踪")
        self.tab_widget.addTab(self._create_multi_frame_tab(), "多帧分析")
        self.tab_widget.addTab(self._create_ui_tab(), "界面")
        self.tab_widget.addTab(self._create_data_tab(), "数据")

        # Buttons
        button_layout = QtWidgets.QHBoxLayout()

        reset_btn = QtWidgets.QPushButton("恢复默认")
        reset_btn.clicked.connect(self._reset_to_defaults)
        button_layout.addWidget(reset_btn)

        button_layout.addStretch()

        cancel_btn = QtWidgets.QPushButton("取消")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)

        save_btn = QtWidgets.QPushButton("保存")
        save_btn.clicked.connect(self._save_and_close)
        save_btn.setDefault(True)
        button_layout.addWidget(save_btn)

        layout.addLayout(button_layout)

    def _create_llm_tab(self) -> QtWidgets.QWidget:
        """Create LLM settings tab."""
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QFormLayout(widget)
        layout.setFieldGrowthPolicy(QtWidgets.QFormLayout.ExpandingFieldsGrow)

        # API Key
        self.llm_api_key = QtWidgets.QLineEdit()
        self.llm_api_key.setEchoMode(QtWidgets.QLineEdit.Password)
        layout.addRow("API Key:", self.llm_api_key)

        # Base URL
        self.llm_base_url = QtWidgets.QLineEdit()
        layout.addRow("Base URL:", self.llm_base_url)

        # Model
        self.llm_model = QtWidgets.QLineEdit()
        layout.addRow("模型:", self.llm_model)

        # Timeout
        self.llm_timeout = QtWidgets.QSpinBox()
        self.llm_timeout.setRange(5, 300)
        self.llm_timeout.setSuffix(" 秒")
        layout.addRow("超时:", self.llm_timeout)

        # Auto analyze on pause
        self.llm_auto_analyze = QtWidgets.QCheckBox("暂停时自动分析")
        layout.addRow("", self.llm_auto_analyze)

        layout.addItem(QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding))

        return widget

    def _create_detection_tab(self) -> QtWidgets.QWidget:
        """Create detection settings tab."""
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QFormLayout(widget)

        # Model selection
        self.detection_model = QtWidgets.QComboBox()
        self.detection_model.addItems([
            "fasterrcnn_resnet50",
            "fasterrcnn_mobilenet",
            "fcos_resnet50",
            "retinanet_resnet50"
        ])
        layout.addRow("检测模型:", self.detection_model)

        # Confidence threshold
        self.detection_confidence = QtWidgets.QDoubleSpinBox()
        self.detection_confidence.setRange(0.1, 1.0)
        self.detection_confidence.setSingleStep(0.05)
        self.detection_confidence.setDecimals(2)
        layout.addRow("置信度阈值:", self.detection_confidence)

        # Device
        self.detection_device = QtWidgets.QComboBox()
        self.detection_device.addItems(["cuda", "cpu", "mps"])
        layout.addRow("计算设备:", self.detection_device)

        # Enabled by default
        self.detection_enabled = QtWidgets.QCheckBox("启动时自动启用")
        layout.addRow("", self.detection_enabled)

        layout.addItem(QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding))

        return widget

    def _create_tracking_tab(self) -> QtWidgets.QWidget:
        """Create tracking settings tab."""
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QFormLayout(widget)

        # Padding
        self.tracking_padding = QtWidgets.QDoubleSpinBox()
        self.tracking_padding.setRange(1.0, 5.0)
        self.tracking_padding.setSingleStep(0.5)
        self.tracking_padding.setDecimals(1)
        layout.addRow("边界填充:", self.tracking_padding)

        # Features
        self.tracking_features = QtWidgets.QComboBox()
        self.tracking_features.addItems(["gray", "color"])
        layout.addRow("特征类型:", self.tracking_features)

        # Kernel
        self.tracking_kernel = QtWidgets.QComboBox()
        self.tracking_kernel.addItems(["linear", "gaussian"])
        layout.addRow("核函数:", self.tracking_kernel)

        # Lambda
        self.tracking_lambda = QtWidgets.QDoubleSpinBox()
        self.tracking_lambda.setRange(1e-6, 1e-2)
        self.tracking_lambda.setSingleStep(1e-5)
        self.tracking_lambda.setDecimals(6)
        self.tracking_lambda.setValue(1e-4)
        layout.addRow("正则化参数:", self.tracking_lambda)

        # Enabled by default
        self.tracking_enabled = QtWidgets.QCheckBox("启动时自动启用")
        layout.addRow("", self.tracking_enabled)

        layout.addItem(QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding))

        return widget

    def _create_cross_camera_tab(self) -> QtWidgets.QWidget:
        """Create cross-camera tracking settings tab."""
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QFormLayout(widget)

        # Enabled
        self.cross_camera_enabled = QtWidgets.QCheckBox("启用跨相机跟踪")
        layout.addRow("", self.cross_camera_enabled)

        # Similarity threshold
        self.cross_camera_similarity = QtWidgets.QDoubleSpinBox()
        self.cross_camera_similarity.setRange(0.1, 1.0)
        self.cross_camera_similarity.setSingleStep(0.05)
        self.cross_camera_similarity.setDecimals(2)
        layout.addRow("相似度阈值:", self.cross_camera_similarity)

        # Max distance
        self.cross_camera_distance = QtWidgets.QSpinBox()
        self.cross_camera_distance.setRange(10, 500)
        self.cross_camera_distance.setSuffix(" 像素")
        layout.addRow("最大距离:", self.cross_camera_distance)

        # Feature type
        self.cross_camera_feature = QtWidgets.QComboBox()
        self.cross_camera_feature.addItems(["color_histogram", "deep_features"])
        layout.addRow("特征类型:", self.cross_camera_feature)

        # Info label
        info_label = QtWidgets.QLabel(
            "跨相机跟踪可以在多个无人机视角中识别同一目标。\n"
            "需要先启用目标跟踪功能。"
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: gray; font-size: 10pt;")
        layout.addRow("", info_label)

        layout.addItem(QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding))

        return widget

    def _create_multi_frame_tab(self) -> QtWidgets.QWidget:
        """Create multi-frame LLM analysis settings tab."""
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QFormLayout(widget)

        # Enabled
        self.multi_frame_enabled = QtWidgets.QCheckBox("启用多帧分析")
        layout.addRow("", self.multi_frame_enabled)

        # Frame count
        self.multi_frame_count = QtWidgets.QSpinBox()
        self.multi_frame_count.setRange(2, 10)
        self.multi_frame_count.setSuffix(" 帧")
        layout.addRow("帧数:", self.multi_frame_count)

        # Frame interval
        self.multi_frame_interval = QtWidgets.QSpinBox()
        self.multi_frame_interval.setRange(1, 50)
        self.multi_frame_interval.setSuffix(" 帧")
        layout.addRow("帧间隔:", self.multi_frame_interval)

        # Analysis type
        self.multi_frame_analysis = QtWidgets.QComboBox()
        self.multi_frame_analysis.addItems(["motion", "behavior", "scene_change"])
        layout.addRow("分析类型:", self.multi_frame_analysis)

        # Info label
        info_label = QtWidgets.QLabel(
            "多帧分析将多个连续帧发送给大模型进行时序推理。\n"
            "motion: 运动分析\n"
            "behavior: 行为分析\n"
            "scene_change: 场景变化检测"
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: gray; font-size: 10pt;")
        layout.addRow("", info_label)

        layout.addItem(QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding))

        return widget

    def _create_ui_tab(self) -> QtWidgets.QWidget:
        """Create UI settings tab."""
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QFormLayout(widget)

        # Theme
        self.ui_theme = QtWidgets.QComboBox()
        self.ui_theme.addItems(["dark", "light"])
        layout.addRow("主题:", self.ui_theme)

        # Default view
        self.ui_default_view = QtWidgets.QComboBox()
        self.ui_default_view.addItems(["single", "grid"])
        layout.addRow("默认视图:", self.ui_default_view)

        # Show FPS
        self.ui_show_fps = QtWidgets.QCheckBox("显示帧率")
        layout.addRow("", self.ui_show_fps)

        # Show confidence
        self.ui_show_confidence = QtWidgets.QCheckBox("显示置信度")
        layout.addRow("", self.ui_show_confidence)

        layout.addItem(QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding))

        return widget

    def _create_data_tab(self) -> QtWidgets.QWidget:
        """Create data settings tab."""
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QFormLayout(widget)

        # Default data directory
        data_dir_layout = QtWidgets.QHBoxLayout()
        self.data_default_dir = QtWidgets.QLineEdit()
        data_dir_layout.addWidget(self.data_default_dir)
        browse_btn = QtWidgets.QPushButton("浏览...")
        browse_btn.clicked.connect(lambda: self._browse_directory(self.data_default_dir))
        data_dir_layout.addWidget(browse_btn)
        layout.addRow("默认数据目录:", data_dir_layout)

        # Output directory
        output_dir_layout = QtWidgets.QHBoxLayout()
        self.data_output_dir = QtWidgets.QLineEdit()
        output_dir_layout.addWidget(self.data_output_dir)
        browse_btn2 = QtWidgets.QPushButton("浏览...")
        browse_btn2.clicked.connect(lambda: self._browse_directory(self.data_output_dir))
        output_dir_layout.addWidget(browse_btn2)
        layout.addRow("输出目录:", output_dir_layout)

        # Paused frames directory
        self.data_paused_frames_dir = QtWidgets.QLineEdit()
        layout.addRow("暂停帧目录:", self.data_paused_frames_dir)

        # Save detections
        self.data_save_detections = QtWidgets.QCheckBox("保存检测结果")
        layout.addRow("", self.data_save_detections)

        # Save tracking
        self.data_save_tracking = QtWidgets.QCheckBox("保存跟踪结果")
        layout.addRow("", self.data_save_tracking)

        layout.addItem(QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding))

        return widget

    def _browse_directory(self, line_edit: QtWidgets.QLineEdit):
        """Open directory browser and set selected path to line edit."""
        current_dir = line_edit.text() or ""
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "选择目录",
            current_dir,
            QtWidgets.QFileDialog.ShowDirsOnly | QtWidgets.QFileDialog.DontResolveSymlinks
        )
        if directory:
            line_edit.setText(directory)

    def _load_settings(self):
        """Load values from settings into all UI widgets."""
        # LLM settings
        self.llm_api_key.setText(self.settings.get("llm.api_key", ""))
        self.llm_base_url.setText(self.settings.get("llm.base_url", "https://api.openai.com/v1"))
        self.llm_model.setText(self.settings.get("llm.model", "gpt-4o-mini"))
        self.llm_timeout.setValue(self.settings.get("llm.timeout", 60))
        self.llm_auto_analyze.setChecked(self.settings.get("llm.auto_analyze_on_pause", True))

        # Detection settings
        detection_model = self.settings.get("detection.model", "fasterrcnn_resnet50")
        index = self.detection_model.findText(detection_model)
        if index >= 0:
            self.detection_model.setCurrentIndex(index)
        self.detection_confidence.setValue(self.settings.get("detection.confidence_threshold", 0.5))
        detection_device = self.settings.get("detection.device", "cuda")
        index = self.detection_device.findText(detection_device)
        if index >= 0:
            self.detection_device.setCurrentIndex(index)
        self.detection_enabled.setChecked(self.settings.get("detection.enabled_by_default", False))

        # Tracking settings
        self.tracking_padding.setValue(self.settings.get("tracking.padding", 2.0))
        tracking_features = self.settings.get("tracking.features", "gray")
        index = self.tracking_features.findText(tracking_features)
        if index >= 0:
            self.tracking_features.setCurrentIndex(index)
        tracking_kernel = self.settings.get("tracking.kernel", "linear")
        index = self.tracking_kernel.findText(tracking_kernel)
        if index >= 0:
            self.tracking_kernel.setCurrentIndex(index)
        self.tracking_lambda.setValue(self.settings.get("tracking.lambda_", 1e-4))
        self.tracking_enabled.setChecked(self.settings.get("tracking.enabled_by_default", False))

        # Cross-camera settings
        self.cross_camera_enabled.setChecked(self.settings.get("cross_camera.enabled", False))
        self.cross_camera_similarity.setValue(self.settings.get("cross_camera.similarity_threshold", 0.7))
        self.cross_camera_distance.setValue(self.settings.get("cross_camera.max_distance", 100))
        cross_camera_feature = self.settings.get("cross_camera.feature_type", "color_histogram")
        index = self.cross_camera_feature.findText(cross_camera_feature)
        if index >= 0:
            self.cross_camera_feature.setCurrentIndex(index)

        # Multi-frame settings
        self.multi_frame_enabled.setChecked(self.settings.get("multi_frame.enabled", False))
        self.multi_frame_count.setValue(self.settings.get("multi_frame.frame_count", 5))
        self.multi_frame_interval.setValue(self.settings.get("multi_frame.frame_interval", 5))
        multi_frame_analysis = self.settings.get("multi_frame.analysis_type", "motion")
        index = self.multi_frame_analysis.findText(multi_frame_analysis)
        if index >= 0:
            self.multi_frame_analysis.setCurrentIndex(index)

        # UI settings
        ui_theme = self.settings.get("ui.theme", "dark")
        index = self.ui_theme.findText(ui_theme)
        if index >= 0:
            self.ui_theme.setCurrentIndex(index)
        ui_default_view = self.settings.get("ui.default_view", "single")
        index = self.ui_default_view.findText(ui_default_view)
        if index >= 0:
            self.ui_default_view.setCurrentIndex(index)
        self.ui_show_fps.setChecked(self.settings.get("ui.show_fps", True))
        self.ui_show_confidence.setChecked(self.settings.get("ui.show_confidence", True))

        # Data settings
        self.data_default_dir.setText(self.settings.get("data.default_directory", "examples"))
        self.data_output_dir.setText(self.settings.get("data.output_directory", "outputs"))
        self.data_paused_frames_dir.setText(self.settings.get("data.paused_frames_directory", "paused_frames"))
        self.data_save_detections.setChecked(self.settings.get("data.save_detections", False))
        self.data_save_tracking.setChecked(self.settings.get("data.save_tracking", False))

    def _save_and_close(self):
        """Save all UI widget values to settings and close dialog."""
        # LLM settings
        self.settings.set("llm.api_key", self.llm_api_key.text())
        self.settings.set("llm.base_url", self.llm_base_url.text())
        self.settings.set("llm.model", self.llm_model.text())
        self.settings.set("llm.timeout", self.llm_timeout.value())
        self.settings.set("llm.auto_analyze_on_pause", self.llm_auto_analyze.isChecked())

        # Detection settings
        self.settings.set("detection.model", self.detection_model.currentText())
        self.settings.set("detection.confidence_threshold", self.detection_confidence.value())
        self.settings.set("detection.device", self.detection_device.currentText())
        self.settings.set("detection.enabled_by_default", self.detection_enabled.isChecked())

        # Tracking settings
        self.settings.set("tracking.padding", self.tracking_padding.value())
        self.settings.set("tracking.features", self.tracking_features.currentText())
        self.settings.set("tracking.kernel", self.tracking_kernel.currentText())
        self.settings.set("tracking.lambda_", self.tracking_lambda.value())
        self.settings.set("tracking.enabled_by_default", self.tracking_enabled.isChecked())

        # Cross-camera settings
        self.settings.set("cross_camera.enabled", self.cross_camera_enabled.isChecked())
        self.settings.set("cross_camera.similarity_threshold", self.cross_camera_similarity.value())
        self.settings.set("cross_camera.max_distance", self.cross_camera_distance.value())
        self.settings.set("cross_camera.feature_type", self.cross_camera_feature.currentText())

        # Multi-frame settings
        self.settings.set("multi_frame.enabled", self.multi_frame_enabled.isChecked())
        self.settings.set("multi_frame.frame_count", self.multi_frame_count.value())
        self.settings.set("multi_frame.frame_interval", self.multi_frame_interval.value())
        self.settings.set("multi_frame.analysis_type", self.multi_frame_analysis.currentText())

        # UI settings
        self.settings.set("ui.theme", self.ui_theme.currentText())
        self.settings.set("ui.default_view", self.ui_default_view.currentText())
        self.settings.set("ui.show_fps", self.ui_show_fps.isChecked())
        self.settings.set("ui.show_confidence", self.ui_show_confidence.isChecked())

        # Data settings
        self.settings.set("data.default_directory", self.data_default_dir.text())
        self.settings.set("data.output_directory", self.data_output_dir.text())
        self.settings.set("data.paused_frames_directory", self.data_paused_frames_dir.text())
        self.settings.set("data.save_detections", self.data_save_detections.isChecked())
        self.settings.set("data.save_tracking", self.data_save_tracking.isChecked())

        # Save to file
        self.settings.save()

        # Close dialog
        self.accept()

    def _reset_to_defaults(self):
        """Reset all settings to default values after confirmation."""
        reply = QtWidgets.QMessageBox.question(
            self,
            "确认重置",
            "确定要恢复所有设置到默认值吗？",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No
        )

        if reply == QtWidgets.QMessageBox.Yes:
            self.settings.reset_to_defaults()
            self._load_settings()
            self.settings.save()
            QtWidgets.QMessageBox.information(
                self,
                "重置完成",
                "所有设置已恢复到默认值。"
            )
