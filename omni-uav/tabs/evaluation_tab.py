"""
Evaluation tab for LLM benchmarking on MMUAVBench dataset.
"""
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from utils import get_settings


class EvaluationTab(QtWidgets.QWidget):
    """Tab for evaluating LLM models on MMUAVBench dataset."""

    # Signal emitted when data is loaded/cleared to notify main window
    data_loaded = QtCore.pyqtSignal(bool)  # True when data loaded, False when cleared

    def __init__(self, llm_client, system_output=None, parent=None):
        super().__init__(parent)
        self.llm_client = llm_client
        self.system_output = system_output
        # Default dataset root, will be overridden by settings
        self.dataset_root = Path("/home/wenjj/Documents/02-Research/08-PCL/03-Benchmarks/01-MMUAVBench/data/MM-UAVBench")
        self.current_data: List[Dict] = []
        self.results: List[Dict] = []
        self.evaluating = False
        self._has_data = False  # Track if data is loaded

        # Discovered tasks from dataset
        self.image_tasks: List[str] = []
        self.video_tasks: List[str] = []

        # Video preview support
        self.current_frames: List[QtGui.QImage] = []
        self.current_frame_index = 0
        self.video_timer = QtCore.QTimer(self)
        self.video_timer.timeout.connect(self._show_next_frame)

        # Evaluation state
        self.eval_state = "stopped"  # stopped, running, paused
        self.eval_current_index = 0
        self.eval_thread = None

        self._build_ui()
        self._load_dataset_root_from_settings()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)

        # Top control panel - Row 1: Dataset path and load button
        control_row1 = QtWidgets.QWidget()
        control_row1_layout = QtWidgets.QHBoxLayout(control_row1)
        control_row1_layout.setContentsMargins(0, 0, 0, 0)

        control_row1_layout.addWidget(QtWidgets.QLabel("路径:"))
        self.dataset_path_label = QtWidgets.QLabel("未设置")
        self.dataset_path_label.setStyleSheet("color: gray; border: 1px solid #444; padding: 2px 8px; border-radius: 3px;")
        self.dataset_path_label.setMaximumWidth(300)
        self.dataset_path_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        # Make label clickable for browsing
        self.dataset_path_label.mousePressEvent = self._on_path_label_clicked
        control_row1_layout.addWidget(self.dataset_path_label)

        self.load_btn = QtWidgets.QPushButton("加载数据集")
        self.load_btn.clicked.connect(self._load_dataset)
        control_row1_layout.addWidget(self.load_btn)

        self.toggle_eval_btn = QtWidgets.QPushButton("显示评估界面")
        self.toggle_eval_btn.setCheckable(True)
        self.toggle_eval_btn.clicked.connect(self._toggle_evaluation_sections)
        self.toggle_eval_btn.setEnabled(False)
        control_row1_layout.addWidget(self.toggle_eval_btn)

        control_row1_layout.addStretch()

        layout.addWidget(control_row1)

        # Top control panel - Row 2: Dataset and task selection
        control_row2 = QtWidgets.QWidget()
        control_row2_layout = QtWidgets.QHBoxLayout(control_row2)
        control_row2_layout.setContentsMargins(0, 0, 0, 0)

        control_row2_layout.addWidget(QtWidgets.QLabel("数据集类型:"))
        self.uav_type_combo = QtWidgets.QComboBox()
        self.uav_type_combo.addItems(["单无人机", "多无人机"])
        control_row2_layout.addWidget(self.uav_type_combo)

        control_row2_layout.addWidget(QtWidgets.QLabel("任务类型:"))
        self.task_type_combo = QtWidgets.QComboBox()
        self.task_type_combo.addItems(["图像任务", "视频任务", "所有任务"])
        self.task_type_combo.currentIndexChanged.connect(self._update_category_list)
        control_row2_layout.addWidget(self.task_type_combo)

        control_row2_layout.addWidget(QtWidgets.QLabel("任务类别:"))
        self.task_category_combo = QtWidgets.QComboBox()
        self.task_category_combo.addItems([])
        control_row2_layout.addWidget(self.task_category_combo)

        control_row2_layout.addStretch()

        layout.addWidget(control_row2)

        # Progress bar
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setFormat("%p% (%v/%m)")
        layout.addWidget(self.progress_bar)

        # Main content: horizontal split (image on left, evaluation panel on right)
        main_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)

        # Left: Image/Video display
        left_panel = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_panel)
        left_layout.addWidget(QtWidgets.QLabel("<b>图片/视频</b>"))

        self.image_label = QtWidgets.QLabel()
        self.image_label.setMinimumSize(400, 300)
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: #1e1e1e; border: 1px solid #444;")
        left_layout.addWidget(self.image_label, 1)

        # Image info
        self.image_info_label = QtWidgets.QLabel("未加载图片")
        self.image_info_label.setAlignment(QtCore.Qt.AlignCenter)
        left_layout.addWidget(self.image_info_label)

        main_splitter.addWidget(left_panel)

        # Right: Evaluation controls + Question/Answer/Result
        right_panel = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_panel)

        # Evaluation controls - Row 1: Random evaluation
        self.eval_control_row1 = QtWidgets.QWidget()
        eval_row1_layout = QtWidgets.QHBoxLayout(self.eval_control_row1)
        eval_row1_layout.setContentsMargins(0, 0, 0, 0)
        eval_row1_layout.addWidget(QtWidgets.QLabel("<b>随机评估:</b>"))
        self.random_btn = QtWidgets.QPushButton("随机")
        self.random_btn.clicked.connect(self._show_random_question)
        self.random_btn.setEnabled(False)
        eval_row1_layout.addWidget(self.random_btn)
        self.eval_current_btn = QtWidgets.QPushButton("评估")
        self.eval_current_btn.clicked.connect(self._evaluate_current_question)
        self.eval_current_btn.setEnabled(False)
        eval_row1_layout.addWidget(self.eval_current_btn)
        eval_row1_layout.addStretch()
        right_layout.addWidget(self.eval_control_row1)

        # Evaluation controls - Row 2: Batch evaluation
        self.eval_control_row2 = QtWidgets.QWidget()
        eval_row2_layout = QtWidgets.QHBoxLayout(self.eval_control_row2)
        eval_row2_layout.setContentsMargins(0, 0, 0, 0)
        eval_row2_layout.addWidget(QtWidgets.QLabel("<b>批量评估:</b>"))
        self.eval_start_btn = QtWidgets.QPushButton("开始")
        self.eval_start_btn.clicked.connect(self._start_batch_eval)
        self.eval_start_btn.setEnabled(False)
        eval_row2_layout.addWidget(self.eval_start_btn)
        self.eval_pause_btn = QtWidgets.QPushButton("暂停")
        self.eval_pause_btn.clicked.connect(self._pause_batch_eval)
        self.eval_pause_btn.setEnabled(False)
        eval_row2_layout.addWidget(self.eval_pause_btn)
        self.eval_stop_btn = QtWidgets.QPushButton("停止")
        self.eval_stop_btn.clicked.connect(self._stop_batch_eval)
        self.eval_stop_btn.setEnabled(False)
        eval_row2_layout.addWidget(self.eval_stop_btn)
        self.eval_resume_btn = QtWidgets.QPushButton("继续")
        self.eval_resume_btn.clicked.connect(self._resume_batch_eval)
        self.eval_resume_btn.setEnabled(False)
        eval_row2_layout.addWidget(self.eval_resume_btn)
        eval_row2_layout.addStretch()
        right_layout.addWidget(self.eval_control_row2)

        right_layout.addSpacing(10)

        # Question section
        self.question_section = QtWidgets.QWidget()
        question_section_layout = QtWidgets.QVBoxLayout(self.question_section)
        question_section_layout.setContentsMargins(0, 0, 0, 0)
        question_section_layout.addWidget(QtWidgets.QLabel("<b>问题</b>"))
        self.question_text = QtWidgets.QTextEdit()
        self.question_text.setReadOnly(True)
        self.question_text.setMaximumHeight(120)
        question_section_layout.addWidget(self.question_text)
        right_layout.addWidget(self.question_section)

        # Answer section
        self.answer_section = QtWidgets.QWidget()
        answer_section_layout = QtWidgets.QVBoxLayout(self.answer_section)
        answer_section_layout.setContentsMargins(0, 0, 0, 0)
        answer_section_layout.addWidget(QtWidgets.QLabel("<b>模型回答</b>"))
        self.answer_text = QtWidgets.QTextEdit()
        self.answer_text.setReadOnly(True)
        self.answer_text.setMaximumHeight(100)
        self.answer_text.setPlaceholderText("模型回答将显示在这里")
        answer_section_layout.addWidget(self.answer_text)
        right_layout.addWidget(self.answer_section)

        # Result section
        self.result_section = QtWidgets.QWidget()
        result_section_layout = QtWidgets.QVBoxLayout(self.result_section)
        result_section_layout.setContentsMargins(0, 0, 0, 0)
        result_section_layout.addWidget(QtWidgets.QLabel("<b>评估结果</b>"))
        self.result_label = QtWidgets.QLabel("未评估")
        self.result_label.setStyleSheet("font-size: 14pt; padding: 5px;")
        result_section_layout.addWidget(self.result_label)
        right_layout.addWidget(self.result_section)

        # Stats section
        self.stats_section = QtWidgets.QWidget()
        stats_section_layout = QtWidgets.QVBoxLayout(self.stats_section)
        stats_section_layout.setContentsMargins(0, 0, 0, 0)
        self.stats_label = QtWidgets.QLabel("总计: 0 | 正确: 0 | 准确率: 0%")
        self.stats_label.setStyleSheet("font-weight: bold; padding: 5px;")
        stats_section_layout.addWidget(self.stats_label)
        right_layout.addWidget(self.stats_section)

        # Results table (collapsed, expandable)
        self.results_table_section = QtWidgets.QWidget()
        results_table_section_layout = QtWidgets.QVBoxLayout(self.results_table_section)
        results_table_section_layout.setContentsMargins(0, 0, 0, 0)
        self.results_table = QtWidgets.QTableWidget()
        self.results_table.setColumnCount(5)
        self.results_table.setHorizontalHeaderLabels(["ID", "问题", "答案", "预测", "正确"])
        self.results_table.horizontalHeader().setStretchLastSection(True)
        self.results_table.setMaximumHeight(200)
        results_table_section_layout.addWidget(self.results_table, 1)
        right_layout.addWidget(self.results_table_section)

        # Hide evaluation controls and results by default
        self.eval_control_row1.hide()
        self.eval_control_row2.hide()
        self.question_section.hide()
        self.answer_section.hide()
        self.result_section.hide()
        self.stats_section.hide()
        self.results_table_section.hide()
        # Track visibility state
        self._eval_sections_visible = False

        main_splitter.addWidget(right_panel)
        main_splitter.setStretchFactor(0, 3)
        main_splitter.setStretchFactor(1, 2)

        layout.addWidget(main_splitter, 1)

    def _shorten_path(self, path: Path) -> str:
        """Shorten path for display (show last 2-3 components)."""
        path_str = str(path)
        parts = path_str.replace('\\', '/').split('/')
        if len(parts) > 3:
            return '.../' + '/'.join(parts[-2:])
        return path_str

    def _on_path_label_clicked(self, _event):
        """Handle path label click event to open browse dialog."""
        self._browse_dataset()

    def _load_dataset_root_from_settings(self):
        """Load dataset root from settings."""
        try:
            settings = get_settings()
            dataset_path = settings.get("evaluation", "dataset_root", "")
            if dataset_path:
                self.dataset_root = Path(dataset_path)
            self.dataset_path_label.setText(self._shorten_path(self.dataset_root))
            self.dataset_path_label.setToolTip(str(self.dataset_root))
            # Discover tasks after loading dataset root
            self._discover_tasks()
        except Exception:
            self.dataset_path_label.setText(self._shorten_path(self.dataset_root))

    def _browse_dataset(self):
        """Browse for dataset directory."""
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self, "选择数据集目录", str(self.dataset_root)
        )
        if directory:
            self.dataset_root = Path(directory)
            self.dataset_path_label.setText(self._shorten_path(self.dataset_root))
            self.dataset_path_label.setToolTip(str(self.dataset_root))
            # Save to settings
            try:
                settings = get_settings()
                settings.set("evaluation", "dataset_root", str(self.dataset_root))
                settings.save()
            except Exception:
                pass
            # Discover tasks after browsing
            self._discover_tasks()

    def _discover_tasks(self):
        """Scan dataset directory to discover available tasks."""
        tasks_dir = self.dataset_root / "tasks"
        self.image_tasks = []
        self.video_tasks = []

        if not tasks_dir.exists() or not tasks_dir.is_dir():
            if self.system_output:
                self.system_output.appendPlainText(f"[警告] 任务目录不存在: {tasks_dir}")
            self._update_category_list()
            return

        # Find all JSON files in tasks directory
        json_files = list(tasks_dir.glob("*.json"))
        if not json_files:
            if self.system_output:
                self.system_output.appendPlainText(f"[警告] 任务目录中没有找到任务文件: {tasks_dir}")
            self._update_category_list()
            return

        # Determine task type by reading each JSON file
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Check data_type from first item's metadata
                if data and isinstance(data, list) and len(data) > 0:
                    metadata = data[0].get('metadata', {})
                    data_type = metadata.get('data_type', '')

                    task_name = json_file.stem  # filename without .json
                    if data_type == 'single_image':
                        self.image_tasks.append(task_name)
                    elif data_type == 'video':
                        self.video_tasks.append(task_name)
            except Exception as e:
                if self.system_output:
                    self.system_output.appendPlainText(f"[警告] 无法读取任务文件 {json_file.name}: {str(e)}")

        # Sort task names
        self.image_tasks.sort()
        self.video_tasks.sort()

        # Update category combo
        self._update_category_list()

        if self.system_output:
            self.system_output.appendPlainText(f"[系统] 发现 {len(self.image_tasks)} 个图像任务, {len(self.video_tasks)} 个视频任务")

    def _update_category_list(self):
        """Update task category list based on task type selection."""
        task_type = self.task_type_combo.currentText()
        self.task_category_combo.clear()

        # Always add "所有类别" option at the top
        all_categories = ["所有类别"]

        if task_type == "图像任务":
            all_categories.extend(self.image_tasks)
        elif task_type == "视频任务":
            all_categories.extend(self.video_tasks)
        else:  # 所有任务
            all_categories.extend(self.image_tasks)
            all_categories.extend(self.video_tasks)

        self.task_category_combo.addItems(all_categories)

    def _load_dataset(self):
        """Load the selected dataset task(s)."""
        category = self.task_category_combo.currentText()

        # Handle "所有类别" (All Categories) option
        if category == "所有类别":
            self._load_all_categories()
            return

        # Load single category
        task_file = self.dataset_root / "tasks" / f"{category}.json"

        if not task_file.exists():
            QtWidgets.QMessageBox.warning(
                self, "错误", f"任务文件不存在: {task_file}"
            )
            return

        try:
            with open(task_file, 'r', encoding='utf-8') as f:
                self.current_data = json.load(f)

            # Enable evaluation buttons when data is loaded
            self.random_btn.setEnabled(True)
            self.eval_current_btn.setEnabled(True)
            self.eval_start_btn.setEnabled(True)
            self.toggle_eval_btn.setEnabled(True)
            self.eval_current_index = 0
            self.eval_state = "stopped"
            self.results.clear()
            self._update_results_table()
            self._has_data = True
            self.data_loaded.emit(True)  # Notify main window that data is loaded

            # Only hide evaluation sections if they aren't already visible
            if not self._eval_sections_visible:
                self._hide_evaluation_sections()
                self.toggle_eval_btn.setChecked(False)
                self.toggle_eval_btn.setText("显示评估界面")

            if self.system_output:
                self.system_output.appendPlainText(f"[系统] 已加载数据集: {category} ({len(self.current_data)} 个问题)")

            # Display first question preview
            if self.current_data:
                self._display_question(0)

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "错误", f"加载数据集失败: {str(e)}")
            if self.system_output:
                self.system_output.appendPlainText(f"[错误] 加载数据集失败: {str(e)}")

    def _load_all_categories(self):
        """Load all tasks based on task type selection."""
        task_type = self.task_type_combo.currentText()

        # Determine which task categories to load
        categories_to_load = []
        if task_type == "图像任务":
            categories_to_load = self.image_tasks
        elif task_type == "视频任务":
            categories_to_load = self.video_tasks
        else:  # 所有任务
            categories_to_load = self.image_tasks + self.video_tasks

        if not categories_to_load:
            QtWidgets.QMessageBox.warning(self, "警告", "没有可用的任务类别")
            return

        # Load all selected task files
        self.current_data = []
        for category in categories_to_load:
            task_file = self.dataset_root / "tasks" / f"{category}.json"
            if task_file.exists():
                try:
                    with open(task_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            self.current_data.extend(data)
                except Exception as e:
                    if self.system_output:
                        self.system_output.appendPlainText(f"[警告] 无法加载 {category}: {str(e)}")

        if not self.current_data:
            QtWidgets.QMessageBox.warning(self, "错误", "未能加载任何数据")
            return

        # Enable evaluation buttons when data is loaded
        self.random_btn.setEnabled(True)
        self.eval_current_btn.setEnabled(True)
        self.eval_start_btn.setEnabled(True)
        self.toggle_eval_btn.setEnabled(True)
        self.eval_current_index = 0
        self.eval_state = "stopped"
        self.results.clear()
        self._update_results_table()
        self._has_data = True
        self.data_loaded.emit(True)  # Notify main window that data is loaded

        # Only hide evaluation sections if they aren't already visible
        if not self._eval_sections_visible:
            self._hide_evaluation_sections()
            self.toggle_eval_btn.setChecked(False)
            self.toggle_eval_btn.setText("显示评估界面")

        if self.system_output:
            self.system_output.appendPlainText(f"[系统] 已加载数据集: 所有类别 ({len(self.current_data)} 个问题)")

        # Display first question preview
        if self.current_data:
            self._display_question(0)

    def _display_question(self, index: int, show_answer: bool = False):
        """Display a question by index."""
        if not self.current_data or index >= len(self.current_data):
            return

        qa = self.current_data[index]
        self.current_qa_index = index
        self.question_text.clear()

        # Stop any existing video timer
        self.video_timer.stop()

        # Display question
        question_text = f"<b>问题 ID:</b> {qa['question_id']}<br>"
        question_text += f"<b>问题:</b> {qa['question']}<br><br>"
        question_text += "<b>选项:</b><br>"
        for opt, text in qa['options'].items():
            question_text += f"{opt}. {text}<br>"
        if show_answer:
            question_text += f"<br><b>正确答案:</b> {qa['answer']}"

        self.question_text.setHtml(question_text)

        # Display image or video
        if qa['metadata']['data_type'] == 'single_image':
            self.video_timer.stop()
            image_path = self.dataset_root / qa['metadata']['data_resources'][0]['path']
            if image_path.exists():
                pixmap = QtGui.QPixmap(str(image_path))
                scaled_pixmap = pixmap.scaled(
                    self.image_label.size(),
                    QtCore.Qt.KeepAspectRatio,
                    QtCore.Qt.SmoothTransformation
                )
                self.image_label.setPixmap(scaled_pixmap)
                self.image_info_label.setText(f"问题 {qa['question_id']}")
            else:
                self.image_label.setText(f"图片未找到:\n{image_path}")
                self.image_info_label.setText("错误: 图片未找到")
        elif qa['metadata']['data_type'] == 'video':
            # Extract and display video frames
            video_path = self.dataset_root / qa['metadata']['data_resources'][0]['path']
            if video_path.exists():
                self.current_frames = self._extract_video_frames(video_path, num_frames=8)
                if self.current_frames:
                    self.current_frame_index = 0
                    self._show_frame(0)
                    # Start video preview timer (500ms per frame)
                    self.video_timer.start(500)
                    self.image_info_label.setText(f"问题 {qa['question_id']} (视频预览)")
                else:
                    self.image_label.setText("无法提取视频帧")
                    self.image_info_label.setText("错误: 视频读取失败")
            else:
                self.image_label.setText(f"视频未找到:\n{video_path}")
                self.image_info_label.setText("错误: 视频未找到")

        self.answer_text.clear()
        self.result_label.setText("未评估")
        self.result_label.setStyleSheet("font-size: 14pt; padding: 5px;")

    def _extract_video_frames(self, video_path: Path, num_frames: int = 8) -> List[QtGui.QImage]:
        """Extract frames from video for preview and evaluation."""
        frames = []
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                return frames

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames == 0:
                cap.release()
                return frames

            # Extract uniformly distributed frames
            frame_indices = [int(i * total_frames / num_frames) for i in range(num_frames)]

            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    h, w, ch = frame_rgb.shape
                    bytes_per_line = ch * w
                    q_image = QtGui.QImage(
                        frame_rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888
                    )
                    frames.append(q_image.copy())

            cap.release()

        except Exception as e:
            if self.system_output:
                self.system_output.appendPlainText(f"[错误] 提取视频帧失败: {str(e)}")

        return frames

    def _show_frame(self, index: int):
        """Display a specific frame."""
        if 0 <= index < len(self.current_frames):
            pixmap = QtGui.QPixmap.fromImage(self.current_frames[index])
            scaled_pixmap = pixmap.scaled(
                self.image_label.size(),
                QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation
            )
            self.image_label.setPixmap(scaled_pixmap)

    def _show_next_frame(self):
        """Show next frame in video preview loop."""
        if not self.current_frames:
            return

        self.current_frame_index = (self.current_frame_index + 1) % len(self.current_frames)
        self._show_frame(self.current_frame_index)

    def _save_video_frames_temp(self, video_path: Path, question_id: str, num_frames: int = 8) -> List[str]:
        """Extract video frames and save to temporary files for LLM evaluation."""
        temp_paths = []
        try:
            import tempfile

            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                return temp_paths

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames == 0:
                cap.release()
                return temp_paths

            # Extract uniformly distributed frames
            frame_indices = [int(i * total_frames / num_frames) for i in range(num_frames)]

            for i, frame_idx in enumerate(frame_indices):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret:
                    # Save to temp file
                    temp_file = tempfile.NamedTemporaryFile(
                        suffix=f"_frame_{i}.jpg",
                        prefix=f"video_{question_id}_",
                        delete=False
                    )
                    temp_path = temp_file.name
                    temp_file.close()

                    cv2.imwrite(temp_path, frame)
                    temp_paths.append(temp_path)

            cap.release()

        except Exception as e:
            if self.system_output:
                self.system_output.appendPlainText(f"[错误] 保存视频帧失败: {str(e)}")

        return temp_paths

    def _show_random_question(self):
        """Select and display a random question without evaluating."""
        if not self.current_data:
            return

        index = random.randint(0, len(self.current_data) - 1)
        self._display_question(index, show_answer=False)
        self.eval_current_btn.setEnabled(True)

    def _evaluate_current_question(self):
        """Evaluate the currently displayed question."""
        # Check if LLM client is loaded
        if not self.llm_client.loaded:
            QtWidgets.QMessageBox.warning(
                self, "未加载大模型",
                "请先在右侧'大模型交互'面板中加载大模型。\n\n"
                "1. 选择大模型\n"
                "2. 点击'加载模型'按钮"
            )
            return

        if not hasattr(self, 'current_qa_index'):
            return

        if self.evaluating:
            return

        self._evaluate_single(self.current_qa_index)

    def _evaluate_single(self, index: int):
        """Evaluate a single question."""
        if self.evaluating:
            return

        self.evaluating = True
        self.random_btn.setEnabled(False)
        self.eval_current_btn.setEnabled(False)
        self.eval_start_btn.setEnabled(False)

        qa = self.current_data[index]

        try:
            # Build prompt
            prompt = self._build_prompt(qa)

            # Get image/video paths
            image_paths = []
            if qa['metadata']['data_type'] == 'single_image':
                image_path = self.dataset_root / qa['metadata']['data_resources'][0]['path']
                if image_path.exists():
                    image_paths.append(str(image_path))
            elif qa['metadata']['data_type'] == 'video':
                # Extract video frames and save to temp files
                video_path = self.dataset_root / qa['metadata']['data_resources'][0]['path']
                if video_path.exists():
                    image_paths = self._save_video_frames_temp(video_path, qa['question_id'])

            # Get LLM response
            if self.system_output:
                self.system_output.appendPlainText(f"[评估] 正在评估问题 {qa['question_id']}...")

            self.answer_text.setPlainText("正在评估...")
            QtCore.QTimer.singleShot(100, lambda: self._do_evaluation(qa, prompt, image_paths, index))

        except Exception as e:
            error_msg = f"评估失败: {str(e)}"
            self.answer_text.setPlainText(error_msg)
            self.result_label.setText("错误")
            self.result_label.setStyleSheet("font-size: 14pt; padding: 5px; color: red;")
            if self.system_output:
                self.system_output.appendPlainText(f"[错误] {error_msg}")
            self.evaluating = False
            self.random_btn.setEnabled(True)
            self.eval_current_btn.setEnabled(True)
            self.eval_start_btn.setEnabled(True)

    def _start_batch_eval(self):
        """Start new batch evaluation from beginning."""
        # Check if LLM client is loaded
        if not self.llm_client.loaded:
            QtWidgets.QMessageBox.warning(
                self, "未加载大模型",
                "请先在右侧'大模型交互'面板中加载大模型。\n\n"
                "1. 选择大模型\n"
                "2. 点击'加载模型'按钮"
            )
            return

        # Load data for currently selected category first
        self._load_dataset()

        if not self.current_data:
            return

        # Start new evaluation (only when not paused)
        if self.eval_state == "paused":
            return

        # Clear previous results
        self.results.clear()
        self._update_results_table()

        self.eval_state = "running"
        self.eval_current_index = 0
        self.progress_bar.setVisible(True)
        self.progress_bar.setMaximum(len(self.current_data))
        self.progress_bar.setValue(0)

        # Start evaluation thread
        self.eval_thread = BatchEvaluationThread(
            self.current_data,
            self.llm_client,
            self.dataset_root,
            start_index=0
        )
        self.eval_thread.progress.connect(self._on_batch_eval_progress)
        self.eval_thread.finished.connect(self._on_batch_eval_finished)
        self.eval_thread.start()

        self._update_batch_eval_ui()

    def _pause_batch_eval(self):
        """Pause batch evaluation."""
        if self.eval_state == "running":
            self.eval_state = "paused"
            if self.eval_thread:
                self.eval_thread.pause()
            self._update_batch_eval_ui()

    def _resume_batch_eval(self):
        """Resume paused batch evaluation."""
        if self.eval_state == "paused":
            self.eval_state = "running"
            if self.eval_thread:
                self.eval_thread.resume()
            self._update_batch_eval_ui()

    def _stop_batch_eval(self):
        """Stop batch evaluation."""
        self.eval_state = "stopped"
        if self.eval_thread:
            self.eval_thread.stop()
            self.eval_thread.wait()
        self.progress_bar.setVisible(False)
        self._update_batch_eval_ui()

    def _update_batch_eval_ui(self):
        """Update UI based on evaluation state."""
        self.eval_start_btn.setEnabled(self.eval_state == "stopped")
        self.eval_pause_btn.setEnabled(self.eval_state == "running")
        self.eval_resume_btn.setEnabled(self.eval_state == "paused")
        self.eval_stop_btn.setEnabled(self.eval_state in ["running", "paused"])

    def _on_batch_eval_progress(self, data: Dict):
        """Handle batch evaluation progress update."""
        self.results.append(data)
        self._update_results_table()
        self.progress_bar.setValue(len(self.results))
        self._display_question(data['index'], show_answer=True)

        # Update result label
        if data.get('correct'):
            self.result_label.setText("✓ 正确")
            self.result_label.setStyleSheet("font-size: 14pt; padding: 5px; color: green; font-weight: bold;")
        else:
            self.result_label.setText("✗ 错误")
            self.result_label.setStyleSheet("font-size: 14pt; padding: 5px; color: red; font-weight: bold;")

        # Update answer text
        result_text = f"<b>模型回答:</b> {data.get('response', 'N/A')}<br><br>"
        result_text += f"<b>预测答案:</b> {data.get('prediction', 'N/A')}<br>"
        result_text += f"<b>正确答案:</b> {data.get('answer', 'N/A')}<br>"
        result_text += f"<b>结果:</b> {'✓ 正确' if data.get('correct') else '✗ 错误'}"
        self.answer_text.setHtml(result_text)

    def _on_batch_eval_finished(self):
        """Handle batch evaluation finished."""
        self.eval_state = "stopped"
        self._update_batch_eval_ui()
        self.progress_bar.setVisible(False)

        if self.system_output:
            correct = sum(1 for r in self.results if r['correct'])
            total = len(self.results)
            accuracy = correct / total if total > 0 else 0
            self.system_output.appendPlainText(f"[系统] 评估完成")
            self.system_output.appendPlainText(f"[系统] 评估结果: {correct}/{total} ({accuracy*100:.1f}%)")

    def _do_evaluation(self, qa: Dict, prompt: str, image_paths: List[str], index: int):
        """Actually perform the evaluation (called after UI update)."""
        try:
            response = self.llm_client.chat(prompt, image_paths)

            # Extract answer (should be a single letter)
            prediction = self._extract_answer(response)

            # Check correctness
            correct = prediction == qa['answer']

            # Update question text to show correct answer
            question_text = f"<b>问题 ID:</b> {qa['question_id']}<br>"
            question_text += f"<b>问题:</b> {qa['question']}<br><br>"
            question_text += "<b>选项:</b><br>"
            for opt, text in qa['options'].items():
                question_text += f"{opt}. {text}<br>"
            question_text += f"<br><b>正确答案:</b> {qa['answer']}"
            self.question_text.setHtml(question_text)

            # Display result in answer text
            result_text = f"<b>模型回答:</b> {response}<br><br>"
            result_text += f"<b>预测答案:</b> {prediction}<br>"
            result_text += f"<b>正确答案:</b> {qa['answer']}<br>"
            result_text += f"<b>结果:</b> {'✓ 正确' if correct else '✗ 错误'}"
            self.answer_text.setHtml(result_text)

            # Update result label
            self.result_label.setText("✓ 正确" if correct else "✗ 错误")
            if correct:
                self.result_label.setStyleSheet("font-size: 14pt; padding: 5px; color: green; font-weight: bold;")
            else:
                self.result_label.setStyleSheet("font-size: 14pt; padding: 5px; color: red; font-weight: bold;")

            # Save result
            self.results.append({
                'question_id': qa['question_id'],
                'question': qa['question'],
                'answer': qa['answer'],
                'prediction': prediction,
                'correct': correct,
                'response': response
            })

            self._update_results_table()

        except Exception as e:
            error_msg = f"评估失败: {str(e)}"
            self.answer_text.setPlainText(error_msg)
            self.result_label.setText("错误")
            self.result_label.setStyleSheet("font-size: 14pt; padding: 5px; color: red;")
            if self.system_output:
                self.system_output.appendPlainText(f"[错误] {error_msg}")

        self.evaluating = False
        self.random_btn.setEnabled(True)
        self.eval_current_btn.setEnabled(True)
        self.eval_start_btn.setEnabled(True)
        self.progress_bar.setVisible(False)

    def _build_prompt(self, qa: Dict) -> str:
        """Build prompt from question data."""
        prompt = "你是无人机领域的专家。请根据你的专业知识回答以下问题。\n\n"
        prompt += f"问题: {qa['question']}\n\n"
        prompt += "选项:\n"
        for opt, text in qa['options'].items():
            prompt += f"{opt}. {text}\n"
        prompt += "\n请仅选择正确选项的字母。"
        return prompt
    

    def _extract_answer(self, response: str) -> str:
        """Extract answer letter from response."""
        # Remove whitespace and get first character
        response = response.strip().upper()

        # Check if response starts with option letter
        for letter in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']:
            if response.startswith(letter):
                return letter

        # If no match, return first character
        if response:
            return response[0]

        return "A"  # Default

    def _show_evaluation_sections(self):
        """Show evaluation controls and results sections after data is loaded."""
        self.eval_control_row1.show()
        self.eval_control_row2.show()
        self.question_section.show()
        self.answer_section.show()
        self.result_section.show()
        self.stats_section.show()
        self.results_table_section.show()

    def _update_results_table(self):
        """Update results table."""
        self.results_table.setRowCount(len(self.results))

        for row, result in enumerate(self.results):
            self.results_table.setItem(row, 0, QtWidgets.QTableWidgetItem(result['question_id']))
            self.results_table.setItem(row, 1, QtWidgets.QTableWidgetItem(result['question'][:50] + "..."))
            self.results_table.setItem(row, 2, QtWidgets.QTableWidgetItem(result['answer']))
            self.results_table.setItem(row, 3, QtWidgets.QTableWidgetItem(result['prediction']))

            correct_item = QtWidgets.QTableWidgetItem("✓" if result['correct'] else "✗")
            correct_item.setForeground(QtGui.QColor("green" if result['correct'] else "red"))
            self.results_table.setItem(row, 4, correct_item)

        # Update stats
        total = len(self.results)
        correct = sum(1 for r in self.results if r['correct'])
        accuracy = correct / total if total > 0 else 0
        self.stats_label.setText(f"总计: {total} | 正确: {correct} | 准确率: {accuracy*100:.1f}%")

    def has_data(self) -> bool:
        """Check if evaluation data is loaded."""
        return self._has_data and len(self.current_data) > 0

    def get_current_question_image_paths(self) -> List[str]:
        """Get image paths for the currently displayed question.

        Returns:
            List of image paths for LLM interaction, or empty list if no data loaded.
        """
        if not self.has_data():
            return []

        # Get the currently displayed question (or first one if none displayed)
        index = getattr(self, 'current_qa_index', 0)
        if index >= len(self.current_data):
            index = 0

        qa = self.current_data[index]
        image_paths = []

        try:
            if qa['metadata']['data_type'] == 'single_image':
                image_path = self.dataset_root / qa['metadata']['data_resources'][0]['path']
                if image_path.exists():
                    image_paths.append(str(image_path))
            elif qa['metadata']['data_type'] == 'video':
                # Extract video frames and save to temp files
                video_path = self.dataset_root / qa['metadata']['data_resources'][0]['path']
                if video_path.exists():
                    image_paths = self._save_video_frames_temp(video_path, qa['question_id'])
        except Exception as e:
            if self.system_output:
                self.system_output.appendPlainText(f"[错误] 获取图片路径失败: {str(e)}")

        return image_paths

    def get_current_question_text(self) -> str:
        """Get the text of the currently displayed question.

        Returns:
            Question text for LLM context, or empty string if no data loaded.
        """
        if not self.has_data():
            return ""

        # Get the currently displayed question (or first one if none displayed)
        index = getattr(self, 'current_qa_index', 0)
        if index >= len(self.current_data):
            index = 0

        qa = self.current_data[index]
        return f"问题: {qa['question']}"

    def _toggle_evaluation_sections(self):
        """Toggle the visibility of evaluation controls and results sections."""
        if self.toggle_eval_btn.isChecked():
            self._show_evaluation_sections()
            self.toggle_eval_btn.setText("隐藏评估界面")
        else:
            self._hide_evaluation_sections()
            self.toggle_eval_btn.setText("显示评估界面")

    def _hide_evaluation_sections(self):
        """Hide evaluation controls and results sections."""
        self.eval_control_row1.hide()
        self.eval_control_row2.hide()
        self.question_section.hide()
        self.answer_section.hide()
        self.result_section.hide()
        self.stats_section.hide()
        self.results_table_section.hide()
        self._eval_sections_visible = False

    def _show_evaluation_sections(self):
        """Show evaluation controls and results sections."""
        self.eval_control_row1.show()
        self.eval_control_row2.show()
        self.question_section.show()
        self.answer_section.show()
        self.result_section.show()
        self.stats_section.show()
        self.results_table_section.show()
        self._eval_sections_visible = True


class BatchEvaluationThread(QtCore.QThread):
    """Background thread for batch evaluation with pause/resume/stop support."""

    progress = QtCore.pyqtSignal(dict)
    finished = QtCore.pyqtSignal()

    def __init__(self, data: List[Dict], llm_client, dataset_root: Path, start_index: int = 0):
        super().__init__()
        self.data = data
        self.llm_client = llm_client
        self.dataset_root = dataset_root
        self.start_index = start_index
        self._paused = False
        self._stopped = False
        self._mutex = QtCore.QMutex()

    def run(self):
        """Run evaluation."""
        for i in range(self.start_index, len(self.data)):
            # Check if stopped
            with QtCore.QMutexLocker(self._mutex):
                if self._stopped:
                    break

                # Check if paused
                while self._paused:
                    self.msleep(100)
                    if self._stopped:
                        break

            if self._stopped:
                break

            qa = self.data[i]
            try:
                # Build prompt
                prompt = "你是无人机领域的专家。请根据你的专业知识回答以下问题。\n\n"
                prompt += f"问题: {qa['question']}\n\n"
                prompt += "选项:\n"
                for opt, text in qa['options'].items():
                    prompt += f"{opt}. {text}\n"
                prompt += "\n请仅选择正确选项的字母。"

                # Get image paths
                image_paths = []
                if qa['metadata']['data_type'] == 'single_image':
                    image_path = self.dataset_root / qa['metadata']['data_resources'][0]['path']
                    if image_path.exists():
                        image_paths.append(str(image_path))
                elif qa['metadata']['data_type'] == 'video':
                    # Extract video frames and save to temp files
                    import tempfile
                    video_path = self.dataset_root / qa['metadata']['data_resources'][0]['path']
                    if video_path.exists():
                        cap = cv2.VideoCapture(str(video_path))
                        if cap.isOpened():
                            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                            if total_frames > 0:
                                num_frames = 8
                                frame_indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
                                for j, frame_idx in enumerate(frame_indices):
                                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                                    ret, frame = cap.read()
                                    if ret:
                                        temp_file = tempfile.NamedTemporaryFile(
                                            suffix=f"_frame_{j}.jpg",
                                            prefix=f"video_{qa['question_id']}_",
                                            delete=False
                                        )
                                        temp_path = temp_file.name
                                        temp_file.close()
                                        cv2.imwrite(temp_path, frame)
                                        image_paths.append(temp_path)
                            cap.release()

                # Get LLM response
                response = self.llm_client.chat(prompt, image_paths)

                # Extract answer
                prediction = self._extract_answer(response)
                correct = prediction == qa['answer']

                self.progress.emit({
                    'index': i,
                    'question_id': qa['question_id'],
                    'question': qa['question'],
                    'answer': qa['answer'],
                    'prediction': prediction,
                    'correct': correct,
                    'response': response
                })

                self.msleep(100)  # Small delay to prevent UI freezing

            except Exception as e:
                # Emit error result
                self.progress.emit({
                    'index': i,
                    'question_id': qa['question_id'],
                    'question': qa['question'],
                    'answer': qa['answer'],
                    'prediction': 'ERROR',
                    'correct': False,
                    'response': str(e)
                })

        self.finished.emit()

    def pause(self):
        """Pause the evaluation."""
        with QtCore.QMutexLocker(self._mutex):
            self._paused = True

    def resume(self):
        """Resume the evaluation."""
        with QtCore.QMutexLocker(self._mutex):
            self._paused = False

    def stop(self):
        """Stop the evaluation."""
        with QtCore.QMutexLocker(self._mutex):
            self._stopped = True
            self._paused = False  # Unpause to allow thread to exit

    def _extract_answer(self, response: str) -> str:
        """Extract answer letter from response."""
        response = response.strip().upper()
        for letter in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']:
            if response.startswith(letter):
                return letter
        if response:
            return response[0]
        return "A"
