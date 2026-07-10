import sys
import yaml
import os
import cv2
import json
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict

# Fix X11 display for GDM sessions
if not os.getenv('DISPLAY') or os.getenv('DISPLAY') == ':0':
    os.environ['DISPLAY'] = ':1'
if not os.getenv('XAUTHORITY'):
    xauth_path = '/run/user/1000/gdm/Xauthority'
    if os.path.exists(xauth_path):
        os.environ['XAUTHORITY'] = xauth_path

from PyQt5 import QtCore, QtGui, QtWidgets

from tabs import MultiUavCameraTab, PlyMeshTab, EvaluationTab
from utils import apply_dark_palette, LlmClient, get_settings, FrameBuffer, get_multi_frame_prompt, get_auto_pause_prompt
from workers import LlmWorker
from dialogs import SettingsDialog

# Load LLM configuration
config_llm_path = Path(__file__).resolve().parent / "configs" / "config_llm.yaml"
if config_llm_path.exists():
    with open(config_llm_path, 'r', encoding='utf-8') as f:
        config_llm = yaml.safe_load(f)
else:
    # Fallback to environment variables
    config_llm = {
        "LLM_API_KEY": os.getenv("LLM_API_KEY", ""),
        "LLM_BASE_URL": os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"),
        "LLM_MODEL": os.getenv("LLM_MODEL", "gpt-4o-mini"),
    }


# [MOD 2026-07-09 | 步骤3] 问答"时间感知"解析：判断问题是否指向过去某时刻/时段。
# 返回 None(未指定→按当前帧回答) 或 (start_ago_sec, end_ago_sec)(多少秒前的时间窗)。
def parse_temporal_scope(text):
    import re
    if not text:
        return None
    t = str(text)
    m = re.search(r'(\d+(?:\.\d+)?)\s*分钟?前', t)              # N 分钟前
    if m:
        n = float(m.group(1)) * 60.0
        return (max(0.0, n - 3.0), n + 3.0)
    m = re.search(r'(?:过去|最近|前)\s*(\d+(?:\.\d+)?)\s*秒', t)  # 过去/最近/前 N 秒 → 0..N
    if m:
        return (0.0, float(m.group(1)))
    m = re.search(r'(\d+(?:\.\d+)?)\s*秒(?:钟)?前', t)           # N 秒(钟)前 → 该时刻附近
    if m:
        n = float(m.group(1))
        return (max(0.0, n - 2.0), n + 2.0)
    if re.search(r'刚才|刚刚|方才|先前', t):                     # 刚才/刚刚 → 最近约 8 秒
        return (0.0, 8.0)
    return None


# [MOD 2026-07-10 | P0 grounded问答] 问答路由判定：引擎在→走 MVA sidecar，否则本地降级。
def decide_qa_route(engine_alive: bool) -> str:
    return "sidecar" if engine_alive else "local"


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, data_dir: Path):
        super().__init__()
        self.setWindowTitle("OmniUAV")
        self.resize(1200, 720)

        self._create_menu_bar()

        self.llm_client = LlmClient(
            model_name=config_llm.get("LLM_MODEL", "gpt-4o-mini"),
            base_url=config_llm.get("LLM_BASE_URL"),
            api_key=config_llm.get("LLM_API_KEY"),
        )
        # [MOD 2026-07-10 | P0] MVA sidecar 客户端 + 引擎状态灯(仅问答用；指令分支不受影响)
        from utils.mva_client import MvaClient
        self.mva_client = MvaClient()
        self._engine_alive = False
        self.engine_status_label = QtWidgets.QLabel("引擎:检测中…")
        self.statusBar().addPermanentWidget(self.engine_status_label)
        self._engine_timer = QtCore.QTimer(self)
        self._engine_timer.timeout.connect(self._refresh_engine_status)
        self._engine_timer.start(5000)
        self._refresh_engine_status()
        self.llm_worker = None
        self.llm_queue: List[Tuple[str, List[str]]] = []
        self.frame_store_dir = Path(__file__).resolve().parent / "paused_frames"
        self.frame_store_dir.mkdir(exist_ok=True)

        # Initialize frame buffer for multi-frame analysis
        self.frame_buffer = FrameBuffer(max_frames_per_camera=100)

        # Create system output first (needed by camera_tab)
        self.system_output = QtWidgets.QPlainTextEdit()
        self.system_output.setReadOnly(True)
        self.system_output.setPlaceholderText("中间过程日志会显示在这里。")
        # Allow system output to be resizable
        self.system_output.setMinimumHeight(100)  # Only constrain minimum height, allow flexible width

        # Redirect print statements to system output
        self._setup_print_redirect()

        tabs = QtWidgets.QTabWidget()
        tsdf_root = Path(__file__).resolve().parent
        tsdf_mesh = tsdf_root / "outputs" / "output_mesh.ply"
        self.camera_tab = MultiUavCameraTab(
            data_dir=data_dir, on_pause_frame=self._handle_pause_frame,
            frame_buffer=self.frame_buffer, system_output=self.system_output
        )
        # Pass LLM client to camera tab for command parsing fallback
        self.camera_tab.parent_llm_client = self.llm_client
        # [MOD 2026-07-10 | ingest触发] 相机 tab 的"入库"按钮 → MainWindow 调 sidecar 入库
        self.camera_tab.ingest_requested.connect(self._on_ingest_requested)
        tabs.addTab(self.camera_tab, "多无人机镜头")
        # [MOD 2026-07-10 | P1] 多视角检索面板
        from tabs.retrieval_tab import RetrievalTab
        self.retrieval_tab = RetrievalTab(self.mva_client)
        self.retrieval_tab.jump_requested.connect(self._on_jump_requested)
        tabs.addTab(self.retrieval_tab, "多视角检索")
        tabs.addTab(PlyMeshTab(data_dir=data_dir, output_mesh=tsdf_mesh), "场景重建")
        self.evaluation_tab = EvaluationTab(llm_client=self.llm_client, system_output=self.system_output)
        self.evaluation_tab.data_loaded.connect(self._on_evaluation_data_loaded)
        tabs.addTab(self.evaluation_tab, "大模型评估")

        llm_panel = self._build_llm_panel()
        system_panel = self._build_system_panel()

        # Create a vertical splitter for tabs and system output
        vertical_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        vertical_splitter.setChildrenCollapsible(False)
        vertical_splitter.setHandleWidth(5)  # Make handle more visible
        vertical_splitter.addWidget(tabs)
        vertical_splitter.addWidget(system_panel)
        # Set initial sizes (70% tabs, 30% system output)
        vertical_splitter.setSizes([500, 200])

        # Main horizontal splitter (left side vs LLM panel)
        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        splitter.setChildrenCollapsible(False)  # Prevent panels from being collapsed completely
        splitter.setHandleWidth(5)  # Make handle more visible

        splitter.addWidget(vertical_splitter)
        splitter.addWidget(llm_panel)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        # Set initial splitter sizes to prevent issues
        splitter.setSizes([700, 400])

        self.setCentralWidget(splitter)

    def _create_menu_bar(self):
        """Create the application menu bar."""
        menu_bar = self.menuBar()

        # File menu
        file_menu = menu_bar.addMenu("文件")

        settings_action = QtWidgets.QAction("设置", self)
        settings_action.triggered.connect(self._open_settings)
        file_menu.addAction(settings_action)

        file_menu.addSeparator()

        exit_action = QtWidgets.QAction("退出", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

    def _open_settings(self):
        """Open the settings dialog."""
        dialog = SettingsDialog(self)
        dialog.exec_()

    def _build_llm_panel(self):
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(6, 6, 6, 6)

        title = QtWidgets.QLabel("大模型交互")
        title.setStyleSheet("font-weight: bold;")

        # Model selection
        model_layout = QtWidgets.QHBoxLayout()
        model_layout.addWidget(QtWidgets.QLabel("大模型:"))
        self.model_combo = QtWidgets.QComboBox()
        self.model_combo.addItems([
            "gpt-4o-mini",
            "gpt-4o",
            "qwen3-vl-plus",
            "qwen3-vl-max",
            "videollama3-7b (本地)"
        ])
        # Set current model from settings or default to gpt-4o-mini
        settings = get_settings()
        current_model = settings.get("llm", "model", "gpt-4o-mini")
        index = self.model_combo.findText(current_model)
        if index >= 0:
            self.model_combo.setCurrentIndex(index)
        model_layout.addWidget(self.model_combo)
        model_layout.addStretch()

        self.load_model_btn = QtWidgets.QPushButton("加载模型")
        self.load_model_btn.clicked.connect(self._load_model)
        model_layout.addWidget(self.load_model_btn)

        demo_hint = QtWidgets.QLabel("可发送当前前视镜头帧给大模型。")
        demo_hint.setWordWrap(True)

        self.include_images_cb = QtWidgets.QCheckBox("附带最新前视帧")
        self.include_images_cb.setChecked(True)

        # Multi-frame analysis controls
        multi_frame_settings = settings.get_category("multi_frame_llm")

        self.multi_frame_cb = QtWidgets.QCheckBox("启用多帧分析")
        self.multi_frame_cb.setChecked(multi_frame_settings.get("enabled", False))
        self.multi_frame_cb.stateChanged.connect(self._toggle_multi_frame_controls)

        # Analysis type selection
        analysis_type_layout = QtWidgets.QHBoxLayout()
        analysis_type_layout.addWidget(QtWidgets.QLabel("分析类型:"))
        self.analysis_type_combo = QtWidgets.QComboBox()
        self.analysis_type_combo.addItems(["motion", "behavior", "scene_change"])
        current_type = multi_frame_settings.get("analysis_type", "motion")
        self.analysis_type_combo.setCurrentText(current_type)
        analysis_type_layout.addWidget(self.analysis_type_combo)
        analysis_type_layout.addStretch()

        # Frame count control
        frame_count_layout = QtWidgets.QHBoxLayout()
        frame_count_layout.addWidget(QtWidgets.QLabel("帧数:"))
        self.frame_count_spin = QtWidgets.QSpinBox()
        self.frame_count_spin.setRange(2, 10)
        self.frame_count_spin.setValue(multi_frame_settings.get("frame_count", 5))
        frame_count_layout.addWidget(self.frame_count_spin)
        frame_count_layout.addStretch()

        # Frame interval control
        frame_interval_layout = QtWidgets.QHBoxLayout()
        frame_interval_layout.addWidget(QtWidgets.QLabel("帧间隔:"))
        self.frame_interval_spin = QtWidgets.QSpinBox()
        self.frame_interval_spin.setRange(1, 50)
        self.frame_interval_spin.setValue(multi_frame_settings.get("frame_interval", 10))
        frame_interval_layout.addWidget(self.frame_interval_spin)
        frame_interval_layout.addStretch()

        self.llm_output = QtWidgets.QTextEdit()
        self.llm_output.setReadOnly(True)
        self.llm_output.setPlaceholderText("大模型返回内容会显示在这里。")
        self.llm_output.setMaximumHeight(200)

        input_label = QtWidgets.QLabel("输入")
        self.llm_input = QtWidgets.QPlainTextEdit()
        self.llm_input.setPlaceholderText("输入给大模型的提示词...")
        self.llm_input.setFixedHeight(80)

        self.send_btn = QtWidgets.QPushButton("发送")
        self.send_btn.clicked.connect(self._send_prompt)

        layout.addWidget(title)
        layout.addLayout(model_layout)
        layout.addWidget(demo_hint)
        layout.addWidget(self.include_images_cb)
        layout.addSpacing(10)
        layout.addWidget(self.multi_frame_cb)
        layout.addLayout(analysis_type_layout)
        layout.addLayout(frame_count_layout)
        layout.addLayout(frame_interval_layout)
        layout.addSpacing(10)
        layout.addWidget(input_label)
        layout.addWidget(self.llm_input)
        layout.addWidget(self.send_btn)

        # Quick questions section with category headers
        quick_questions_label = QtWidgets.QLabel("快捷问题")
        quick_questions_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(quick_questions_label)

        # Quick questions scroll area
        quick_questions_scroll = QtWidgets.QScrollArea()
        quick_questions_scroll.setWidgetResizable(True)
        quick_questions_scroll.setMaximumHeight(280)
        quick_questions_widget = QtWidgets.QWidget()
        quick_questions_layout = QtWidgets.QVBoxLayout(quick_questions_widget)
        quick_questions_layout.setSpacing(6)

        # Layer 1: 二维检测层
        layer1_label = QtWidgets.QLabel("二维检测层")
        layer1_label.setStyleSheet("font-weight: bold; color: #4a90d9;")
        quick_questions_layout.addWidget(layer1_label)
        layer1_grid = QtWidgets.QGridLayout()
        layer1_grid.setSpacing(4)
        self._add_quick_question_button(layer1_grid, 0, 0, "目标计数", "请统计当前画面中的车辆和行人数量。")
        self._add_quick_question_button(layer1_grid, 0, 1, "目标类型", "请列出画面中检测到的所有目标类型。")
        self._add_quick_question_button(layer1_grid, 0, 2, "特定目标", "请找出所有红色的车辆。")
        self._add_quick_question_button(layer1_grid, 0, 3, "文字识别", "请识别画面中可见的文字内容。")
        quick_questions_layout.addLayout(layer1_grid)

        # Layer 2: 时序跟踪层
        layer2_label = QtWidgets.QLabel("时序跟踪层")
        layer2_label.setStyleSheet("font-weight: bold; color: #4a90d9;")
        quick_questions_layout.addWidget(layer2_label)
        layer2_grid = QtWidgets.QGridLayout()
        layer2_grid.setSpacing(4)
        self._add_quick_question_button(layer2_grid, 0, 0, "运动方向", "请判断标记车辆的运动方向是直行、左转、右转还是静止？")
        self._add_quick_question_button(layer2_grid, 0, 1, "轨迹跟踪", "请分析目标车辆的行驶轨迹。")
        self._add_quick_question_button(layer2_grid, 0, 2, "速度估计", "请估计目标车辆的行驶速度。")
        self._add_quick_question_button(layer2_grid, 0, 3, "运动预测", "请预测目标车辆接下来的移动方向。")
        quick_questions_layout.addLayout(layer2_grid)

        # Layer 3: 三维空间层
        layer3_label = QtWidgets.QLabel("三维空间层")
        layer3_label.setStyleSheet("font-weight: bold; color: #50c878;")
        quick_questions_layout.addWidget(layer3_label)
        layer3_grid = QtWidgets.QGridLayout()
        layer3_grid.setSpacing(4)
        self._add_quick_question_button(layer3_grid, 0, 0, "位置关系", "请描述各目标之间的相对位置关系。")
        self._add_quick_question_button(layer3_grid, 0, 1, "距离估计", "请估算两辆车之间的距离。")
        self._add_quick_question_button(layer3_grid, 0, 2, "场景重建", "请描述场景的三维空间结构。")
        self._add_quick_question_button(layer3_grid, 0, 3, "高度判断", "请判断无人机飞行的高度大约是多少？")
        quick_questions_layout.addLayout(layer3_grid)

        # Layer 4: 语义理解层
        layer4_label = QtWidgets.QLabel("语义理解层")
        layer4_label.setStyleSheet("font-weight: bold; color: #50c878;")
        quick_questions_layout.addWidget(layer4_label)
        layer4_grid = QtWidgets.QGridLayout()
        layer4_grid.setSpacing(4)
        self._add_quick_question_button(layer4_grid, 0, 0, "场景理解", "请分析当前场景的类型和用途。")
        self._add_quick_question_button(layer4_grid, 0, 1, "事件识别", "请描述当前画面中正在发生什么事件？")
        self._add_quick_question_button(layer4_grid, 0, 2, "威胁评估", "请评估当前场景的安全威胁级别。")
        self._add_quick_question_button(layer4_grid, 0, 3, "异常检测", "请检测当前场景中的异常情况。")
        quick_questions_layout.addLayout(layer4_grid)

        quick_questions_layout.addStretch(1)
        quick_questions_scroll.setWidget(quick_questions_widget)
        layout.addWidget(quick_questions_scroll)

        layout.addSpacing(10)
        layout.addWidget(QtWidgets.QLabel("输出"))
        layout.addWidget(self.llm_output)
        layout.addStretch(1)

        # Initialize control states
        self._toggle_multi_frame_controls()

        return panel

    def _add_quick_question_button(self, layout, row, col, label_text, prompt_text):
        """Add a quick question button to the layout.

        Args:
            layout: The grid layout to add the button to
            row: Row position in the grid
            col: Column position in the grid
            label_text: Button label
            prompt_text: Prompt to send when button is clicked
        """
        btn = QtWidgets.QPushButton(label_text)
        btn.setMaximumHeight(28)
        btn.setStyleSheet("font-size: 11px; padding: 2px 6px;")
        # Use a closure to capture the prompt_text
        btn.clicked.connect(lambda checked, p=prompt_text: self._on_quick_question(p))
        layout.addWidget(btn, row, col)

    def _on_quick_question(self, prompt_text):
        """Handle quick question button click - fills input for user editing.

        Args:
            prompt_text: The prompt to fill in input
        """
        self.llm_input.setPlainText(prompt_text)
        # Focus the input so user can easily edit
        self.llm_input.setFocus()

    def _toggle_multi_frame_controls(self):
        """Enable/disable multi-frame controls based on checkbox state."""
        enabled = self.multi_frame_cb.isChecked()
        self.analysis_type_combo.setEnabled(enabled)
        self.frame_count_spin.setEnabled(enabled)
        self.frame_interval_spin.setEnabled(enabled)

        # Save setting
        settings = get_settings()
        settings.set("multi_frame_llm", "enabled", enabled)
        settings.save()

    def _load_model(self):
        """Load the selected LLM model."""
        model_name = self.model_combo.currentText()
        # Clean up the model name (remove suffix like " (本地)")
        if " (本地)" in model_name:
            model_name = model_name.replace(" (本地)", "")

        try:
            # Update the LLM client with the new model
            self.llm_client = LlmClient(model_name=model_name)
            self.llm_client.load()

            # Update evaluation tab's LLM client
            self.evaluation_tab.llm_client = self.llm_client

            # Save model setting
            settings = get_settings()
            settings.set("llm", "model", model_name)
            settings.save()

            # Show success message in system output
            self.system_output.appendPlainText(f"[系统] 模型加载成功: {model_name}")
            self.system_output.appendPlainText(f"[系统] 模型类型: {self.llm_client.model_type.value}")
        except NotImplementedError as e:
            self.system_output.appendPlainText(f"[错误] {str(e)}")
        except RuntimeError as e:
            self.system_output.appendPlainText(f"[错误] 模型加载失败: {str(e)}")
        except Exception as e:
            self.system_output.appendPlainText(f"[错误] 模型加载失败: {str(e)}")

    def _send_prompt(self):
        prompt = self.llm_input.toPlainText().strip()
        if not prompt:
            return

        self.llm_output.append(f"> {prompt}")
        self.llm_input.clear()

        # Step 1: Let LLM analyze user intent (no images needed)
        self._analyze_user_intent(prompt)

    def _analyze_user_intent(self, prompt: str):
        """Let LLM analyze user intent to determine if this is a tracking command."""
        self.llm_output.append("分析意图...")
        self.llm_output.append("")

        # Build intent analysis prompt
        intent_prompt = f"""You are analyzing a user's command for a UAV tracking system. Determine the user's intent.

User command: "{prompt}"

Analyze and return ONLY valid JSON (no other text):

{{
    "is_tracking": true/false,
    "target_drones": ["drone1", "drone2", ...],  // List of drones to track (empty = all)
    "tracking_description": "description of objects to track"  // What to look for
}}

Rules:
- is_tracking: true if user wants to find/filter/track specific objects
- target_drones: Extract which drones (drone1-drone4). Empty list means all drones.
- tracking_description: Brief description of what objects to find (color, type, etc.)
- If NOT a tracking command, set is_tracking=false and leave other fields empty

Examples:
- "2号无人机跟踪红色的车" → {{"is_tracking": true, "target_drones": ["drone2"], "tracking_description": "红色的车"}}
- "Find all red trucks" → {{"is_tracking": true, "target_drones": [], "tracking_description": "red trucks"}}
- "What's in the image?" → {{"is_tracking": false, "target_drones": [], "tracking_description": ""}}

Return JSON only:"""

        # Store original prompt for later use
        self._original_user_prompt = prompt

        # Send intent analysis request (no images, text-only)
        self._enqueue_intent_analysis_request(intent_prompt)

    def _enqueue_intent_analysis_request(self, prompt: str):
        """Enqueue an intent analysis request (text-only, no images)."""
        # Use a special marker for intent analysis
        self.llm_queue.append((prompt, [], "intent_analysis"))
        if not self.llm_worker or not self.llm_worker.isRunning():
            self._start_next_llm()

    def _on_intent_analysis_response(self, text: str):
        """Handle LLM response for intent analysis."""
        # DEBUG: Print raw LLM response for intent analysis
        print(f"\n{'='*60}")
        print(f"[INTENT ANALYSIS] LLM Response:")
        print(f"{'='*60}")
        print(text)
        print(f"{'='*60}\n")

        try:
            # Strip markdown code blocks if present
            json_text = text.strip()
            if json_text.startswith("```json"):
                json_text = json_text[7:]
            elif json_text.startswith("```"):
                json_text = json_text[3:]
            if json_text.endswith("```"):
                json_text = json_text[:-3]
            json_text = json_text.strip()

            # Find JSON object
            start_idx = json_text.find("{")
            if start_idx == -1:
                raise ValueError("No JSON found")

            # Find matching closing brace
            brace_count = 0
            end_idx = -1
            for i in range(start_idx, len(json_text)):
                if json_text[i] == '{':
                    brace_count += 1
                elif json_text[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end_idx = i
                        break

            if end_idx == -1:
                raise ValueError("Incomplete JSON")

            json_text = json_text[start_idx:end_idx + 1]
            result = json.loads(json_text)

            # print(f"[DEBUG] Intent analysis result: {result}")

            # Check if this is a tracking command
            is_tracking = result.get("is_tracking", False)
            print(f"[DEBUG] Intent analysis: is_tracking={is_tracking}, result={result}")

            if is_tracking:
                # Execute tracking command
                target_drones = result.get("target_drones", [])
                tracking_description = result.get("tracking_description", "")
                print(f"[DEBUG] Executing tracking command: description='{tracking_description}', drones={target_drones}")
                self._execute_tracking_command(tracking_description, target_drones)
            else:
                # Regular LLM interaction
                self._execute_regular_query()

        except Exception as e:
            print(f"[ERROR] Intent analysis failed: {e}")
            import traceback
            traceback.print_exc()
            # Fallback to regular query
            self._execute_regular_query()

    def _execute_tracking_command(self, tracking_description: str, target_drones: List[str]):
        """Execute tracking command with parsed parameters.

        Args:
            tracking_description: What objects to find (e.g., "红色的车")
            target_drones: List of drone IDs (e.g., ["drone1", "drone2"])
        """
        self.llm_output.append(f"[跟踪] 目标: {tracking_description}")
        if target_drones:
            drone_names = [d.replace("drone", "").replace("1", "1号").replace("2", "2号").replace("3", "3号").replace("4", "4号") for d in target_drones]
            self.llm_output.append(f"[跟踪] 无人机: {', '.join(drone_names)}")
        else:
            self.llm_output.append("[跟踪] 无人机: 全部")
        self.llm_output.append("")

        # Check if we're in ROS live mode
        is_ros = self.camera_tab.is_ros_mode()
        print(f"[DEBUG] _execute_tracking_command: is_ros_mode={is_ros}, target_drones={target_drones}")
        if is_ros:
            print(f"[DEBUG] Taking ROS tracking path")
            self._handle_ros_tracking_command(tracking_description, target_drones)
        else:
            print(f"[DEBUG] Taking local tracking path")
            # Convert drone IDs to UAV IDs for local mode
            target_uavs = self._drone_ids_to_uav_ids(target_drones)
            self._handle_local_tracking_command(tracking_description, target_uavs)
            # Convert drone IDs to UAV IDs for local mode
            target_uavs = self._drone_ids_to_uav_ids(target_drones)
            self._handle_local_tracking_command(tracking_description, target_uavs)

    def _drone_ids_to_uav_ids(self, drone_ids: List[str]) -> List[str]:
        """Convert drone IDs (drone1, drone2) to UAV IDs (无人机-01, 无人机-02).

        Args:
            drone_ids: List of drone IDs (e.g., ["drone1", "drone2"])

        Returns:
            List of UAV IDs (e.g., ["无人机-01", "无人机-02"])
        """
        if not drone_ids:
            return []  # Empty means all UAVs

        uav_ids = []
        for drone_id in drone_ids:
            # Extract number from drone_id (e.g., "drone1" -> "1")
            num = drone_id.replace("drone", "")
            try:
                num_int = int(num)
                uav_ids.append(f"无人机-{num_int:02d}")
            except ValueError:
                # If parsing fails, skip this drone_id
                pass
        return uav_ids

    def _refresh_engine_status(self):
        # [MOD 2026-07-10 | P0] 周期探活 MVA sidecar，更新状态灯 + _engine_alive(问答路由用)
        self._engine_alive = self.mva_client.is_alive()
        self.engine_status_label.setText("引擎●已连接" if self._engine_alive else "引擎○未连接")

    def _on_jump_requested(self, view_id: str, t_sec: float):
        # [MOD 2026-07-10 | P1] 检索命中 → 跳到对应视角那一帧(见 camera_tab.seek_to)
        cam = {"view1": "cam01", "view2": "cam02",
               "view3": "cam03", "view4": "cam04"}.get(view_id, view_id)
        try:
            self.camera_tab.seek_to(cam, t_sec)
            self.system_output.appendPlainText(f"[检索] 跳转 {view_id} → {t_sec:.1f}s")
        except Exception as e:  # noqa: BLE001
            self.system_output.appendPlainText(f"[检索] 跳转失败: {e}")

    def _on_ingest_requested(self, dataset_root: str, scene: str):
        # [MOD 2026-07-10 | ingest触发] 把当前文件夹作为 pcl-sim scene 送入 sidecar 入库
        if not getattr(self, "_engine_alive", False):
            self.system_output.appendPlainText("[入库] 引擎未连接，无法入库(请先启动 sidecar 引擎)。")
            return
        cfg = {"dataset_root": dataset_root, "segments_per_view": 4}
        yolo = "/home/fyf/fyf/PCL/Multi-Video-Analysis/yolo11n.pt"  # 已有权重，免下载
        if os.path.exists(yolo):
            cfg["detect_model"] = yolo
        try:
            job = self.mva_client.ingest_start(source=scene, dataset="pcl-sim", config=cfg)
        except Exception as e:  # noqa: BLE001
            self.system_output.appendPlainText(f"[入库] 触发失败: {e}")
            return
        self._ingest_job = job
        self.system_output.appendPlainText(f"[入库] 已开始: scene={scene} (job={job})，处理中…")
        self._ingest_timer = QtCore.QTimer(self)
        self._ingest_timer.timeout.connect(self._poll_ingest)
        self._ingest_timer.start(2000)

    def _poll_ingest(self):
        # [MOD 2026-07-10 | ingest触发] 轮询入库状态，完成/失败时收尾(running 静默，状态灯已反映引擎)
        try:
            st = self.mva_client.ingest_status(self._ingest_job)
        except Exception as e:  # noqa: BLE001
            self.system_output.appendPlainText(f"[入库] 查询状态失败: {e}")
            self._ingest_timer.stop()
            return
        state = st.get("state")
        if state == "done":
            self.system_output.appendPlainText(
                f"[入库] 完成 ✓ 段数={st.get('processed_segments')}。现在可对该场景做 grounded 问答。"
            )
            self._ingest_timer.stop()
        elif state == "error":
            self.system_output.appendPlainText(f"[入库] 失败: {st.get('error')}")
            self._ingest_timer.stop()

    def _execute_regular_query(self):
        """Execute regular LLM query (non-tracking)."""
        print(f"[DEBUG] _execute_regular_query called (NOT a tracking command)")
        # [MOD 2026-07-10 | P0] 问答分支路由：引擎在→MVA grounded 问答；失败/不在→落到原本地路径降级
        if decide_qa_route(getattr(self, "_engine_alive", False)) == "sidecar":
            try:
                result = self.mva_client.answer(self._original_user_prompt)
                ans = result.get("answer", "")
                g = result.get("groundings") or []
                src = ("  [溯源] " + ", ".join(
                    f"{x.get('view_id')}@{x.get('t')}" for x in g)) if g else ""
                self.llm_output.append(f"[grounded] {ans}{src}")
                return
            except Exception as e:                        # noqa: BLE001
                print(f"[P0] sidecar 问答失败，降级本地: {e}")
                # 不 return，继续走下面原有本地路径
        self.llm_output.append("处理中...")
        self.llm_output.append("")

        prompt = self._original_user_prompt
        image_paths = []
        if self.include_images_cb.isChecked():
            # Check if evaluation tab has data loaded - use it as image source
            if self.evaluation_tab.has_data():
                image_paths = self.evaluation_tab.get_current_question_image_paths()
            else:
                # [MOD 2026-07-09 | 步骤3] 时间感知选帧：
                #   未指定过去时刻 → 只用"当前帧"；指定了时刻/时段 → 取对应历史帧。
                #   (原逻辑按"启用多帧分析"复选框把最近多帧一起发；现改为按问题中的时间引用决定。)
                scope = parse_temporal_scope(prompt)
                if scope is not None:
                    image_paths = self._get_time_scoped_frame_paths(scope)
                    if image_paths:
                        self.llm_output.append(
                            f"[时间感知] 使用约 {scope[0]:.0f}-{scope[1]:.0f} 秒前的历史帧 ({len(image_paths)}张)"
                        )
                    else:
                        self.llm_output.append("[时间感知] 该时段缓冲区暂无历史帧，改用当前帧")
                        image_paths = self._get_latest_frame_paths()
                    self.llm_output.append("")
                else:
                    # 默认：仅当前帧
                    image_paths = self._get_latest_frame_paths()

        self._enqueue_llm_request(prompt, image_paths)

    def _handle_local_tracking_command(self, tracking_description: str, target_uavs: List[str]):
        """Handle tracking command for local file mode.

        Args:
            tracking_description: What objects to find (e.g., "红色的车")
            target_uavs: List of UAV IDs (e.g., ["无人机-01", "无人机-02"])
        """
        # If no UAVs specified, use all available UAVs
        if not target_uavs:
            target_uavs = self.camera_tab.uav_ids

        # Auto-enable detection if not already enabled
        if not self.camera_tab.detection_enabled:
            self.camera_tab.detection_enabled = True
            self.camera_tab.detection_checkbox.setChecked(True)
            if self.camera_tab.detector is None:
                self.camera_tab._init_detector()

        # Auto-enable tracking if not already enabled
        if not self.camera_tab.tracking_enabled:
            self.camera_tab.tracking_enabled = True
            self.camera_tab.tracking_checkbox.setChecked(True)

        # Initialize trackers for ALL current detections on target UAVs
        for uav_id in target_uavs:
            self._initialize_trackers_for_detections(uav_id)

        # Collect 5 frames with tracking data (auto-advances video)
        frames_data = self.camera_tab._collect_tracking_frames(target_uavs)

        if not frames_data:
            self.llm_output.append("[提示] 检测器无可用权重，未采集到跟踪数据；改用大模型直接分析画面。")
            self.llm_output.append("")
            self._execute_regular_query()
            return

        # Create annotated images (from latest frame after collection)
        annotated_paths = self._create_annotated_tracking_images(target_uavs)

        if not annotated_paths:
            self.llm_output.append("[提示] 未获取到跟踪框；改用大模型直接分析画面。")
            self.llm_output.append("")
            self._execute_regular_query()
            return

        # Build LLM prompt for tracker matching with multi-frame context
        llm_prompt = self._build_tracking_matching_prompt(tracking_description, frames_data, annotated_paths)

        # Enqueue tracking-specific LLM request
        self._enqueue_tracking_llm_request(llm_prompt, annotated_paths, target_uavs, user_prompt=tracking_description)

    def _handle_ros_tracking_command(self, tracking_description: str, target_drone_ids: List[str]):
        """Handle tracking command in ROS live mode.

        Args:
            tracking_description: What objects to find (e.g., "红色的车")
            target_drone_ids: List of drone IDs (e.g., ["drone1", "drone2"])

        Each crop image is a collage of tracked objects with IDs shown below.
        We analyze only the LATEST crop from each drone.
        """
        import tempfile
        import os

        self.llm_output.append("处理ROS跟踪命令...")
        self.llm_output.append("")

        # If no drones specified, use all drones
        if not target_drone_ids:
            target_drone_ids = ["drone1", "drone2", "drone3", "drone4"]

        # Initialize ROS tracking components (subscribers and publishers)
        self.camera_tab._init_ros_tracking_components(target_drone_ids)

        # Wait a bit for crops to arrive
        import time
        time.sleep(1.0)

        # Collect all crops from the LATEST frame for each target drone
        # Each drone may have multiple collage images from the same frame (split into smaller pieces)
        latest_frame_crops_by_drone = {}
        for drone_id in target_drone_ids:
            frame_crops = self.camera_tab.get_ros_track_crops_by_frame(drone_id)
            if frame_crops:
                latest_frame_crops_by_drone[drone_id] = frame_crops

        if not latest_frame_crops_by_drone:
            self.llm_output.append("[错误] 无法获取跟踪裁剪图像。")
            self.llm_output.append("请确保:")
            self.llm_output.append("  - ROS正在运行并发布track_crops话题")
            self.llm_output.append("  - 跟踪器已在目标无人机上运行")
            self.llm_output.append("")
            return

        # Show what crops we received
        total_crops = sum(len(crops) for crops in latest_frame_crops_by_drone.values())
        self.llm_output.append(f"[调试] 接收到 {len(latest_frame_crops_by_drone)} 个无人机的拼贴图像 (共 {total_crops} 张)\n")
        for drone_id, crops in latest_frame_crops_by_drone.items():
            frame_id = crops[0].get('frame_id', 'unknown') if crops else 'unknown'
            self.llm_output.append(f"  - {drone_id}: {len(crops)} 张拼贴图 (frame_id={frame_id})\n")
        self.llm_output.append("")

        # Save crops to temporary files for LLM analysis
        crop_paths = []
        for drone_id, frame_crops in latest_frame_crops_by_drone.items():
            for idx, crop_data in enumerate(frame_crops):
                crop_image = crop_data['image']

                # Convert RGB to BGR for cv2
                import cv2
                crop_bgr = cv2.cvtColor(crop_image, cv2.COLOR_RGB2BGR)

                # Create temp file with index to handle multiple crops per drone
                temp_fd, temp_path = tempfile.mkstemp(suffix=f'_{drone_id}_{idx}.jpg')
                os.close(temp_fd)
                cv2.imwrite(temp_path, crop_bgr)
                # Store path with drone_id and index
                crop_paths.append((temp_path, drone_id))

        # Build LLM prompt for crop analysis
        # Use the pre-extracted tracking description from LLM intent analysis
        description = tracking_description

        # Count how many crops per drone
        crops_per_drone = {drone_id: len(crops) for drone_id, crops in latest_frame_crops_by_drone.items()}
        drone_info = ", ".join([f"{drone_id}({count}张)" for drone_id, count in crops_per_drone.items()])

        llm_prompt = f"""You are analyzing images from different UAVs. Each image is a collage of tracked objects, where each object has its tracker ID shown below it.

Your task is to identify which tracker IDs match the following description: "{description}"

You will receive {len(crop_paths)} image(s) from {len(latest_frame_crops_by_drone)} drone(s): {drone_info}.
IMPORTANT: Multiple images may come from the same drone and the same frame - they are split into smaller collages for better visibility. You need to examine ALL images from the same drone to find all matching objects.

Think step-by-step:
1. For each drone, go through ALL its collage images
2. List ALL tracker IDs visible across all images from that drone
3. From the listed IDs, select those that match the description: "{description}"
4. Each drone's total number of matched objects must be less than 20

CRITICAL: You must respond with EXACTLY this JSON structure:
{{
    "is_matched": true,
    "matched": {{
        "drone1": [1, 5, 10],
        "drone2": [3, 7]
    }}
}}

Where:
- "is_matched": true if any object matches, false if none match
- "matched" contains drone IDs and their matching tracker IDs
- ONLY include drones that have matching trackers in "matched"
- If no matches: {{"is_matched": false, "matched": {{}}}}

Return ONLY this JSON, no other text.

Your response:"""

        # print(f"[DEBUG] ROS Tracking prompt:\n{llm_prompt}\n")

        # Store tracking context for response handling
        self._tracking_context = {
            "description": description,
            "target_drone_ids": target_drone_ids,
            "crops_by_drone": latest_frame_crops_by_drone,
            "crop_paths": crop_paths,
            "is_ros_mode": True
        }
        print(f"[DEBUG] _handle_ros_tracking_command: set _tracking_context with is_ros_mode=True")
        print(f"[DEBUG] _tracking_context keys: {list(self._tracking_context.keys())}")

        # Enqueue ROS tracking LLM request directly (don't use _enqueue_tracking_llm_request to avoid overwriting context)
        paths_only = [p for (p, _) in crop_paths]
        self.llm_queue.append((llm_prompt, paths_only, True))
        if not self.llm_worker or not self.llm_worker.isRunning():
            self._start_next_llm()

        # Clean up temp files after a delay
        def cleanup_temp_files():
            time.sleep(10)
            for (path, _) in crop_paths:
                try:
                    os.remove(path)
                except:
                    pass

        import threading
        threading.Thread(target=cleanup_temp_files, daemon=True).start()

    def _initialize_trackers_for_detections(self, uav_id: str):
        """Initialize trackers for all current detections on a specific UAV.

        Args:
            uav_id: The UAV ID to initialize trackers for
        """
        stream = self.camera_tab._get_stream_for_uav(uav_id)
        if not stream:
            return

        frame = stream.get_latest()
        # Check for None or null QImage (using isNull() for proper QImage validation)
        if frame is None or frame.isNull():
            # Try to read a new frame if get_latest() returned nothing
            frame = stream.read()
            if frame is None or frame.isNull():
                return

        # Get detections for this UAV
        if uav_id not in self.camera_tab.last_detections:
            return

        boxes, labels, scores = self.camera_tab.last_detections[uav_id]
        if len(boxes) == 0:
            return

        try:
            # Convert QImage to numpy array
            width = frame.width()
            height = frame.height()
            ptr = frame.bits()
            ptr.setsize(height * width * 3)
            arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 3))
            frame_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

            # Add trackers from detections
            self.camera_tab._add_trackers_from_detections(frame_bgr, boxes, labels)
            print(f"[DEBUG] Initialized trackers for {len(boxes)} detections on {uav_id}")
        except Exception as e:
            print(f"[ERROR] Failed to initialize trackers for {uav_id}: {e}")

    def _parse_uav_ids_from_prompt(self, prompt: str) -> Optional[List[str]]:
        """Parse UAV IDs from the prompt."""
        import re
        uav_ids = []

        # Pattern: drone01, cam02, uav03, 无人机-01, etc.
        patterns = [
            r"drone(\d+)",
            r"cam(\d+)",
            r"uav(\d+)",
            r"无人机[ -]?(\d+)",
            r"camera(\d+)",
        ]

        for pattern in patterns:
            matches = re.findall(pattern, prompt, re.IGNORECASE)
            for match in matches:
                uav_ids.append(f"无人机-{int(match):02d}")

        return uav_ids if uav_ids else None

    def _create_annotated_tracking_images(self, target_uavs: Optional[List[str]] = None) -> List[str]:
        """Create images with numbered tracker boxes for LLM analysis."""
        paths = []

        # Get frames from camera tab
        frames = self.camera_tab.get_latest_frames()
        print(f"[DEBUG] Available frames: {list(frames.keys())}")

        # Get active trackers for each UAV
        for uav_id, frame in frames.items():
            # Skip if not in target UAVs
            if target_uavs and uav_id not in target_uavs:
                print(f"[DEBUG] Skipping {uav_id} (not in target UAVs: {target_uavs})")
                continue

            # Get tracker info for this UAV
            tracker_info = self.camera_tab._get_tracker_info(uav_id)
            if not tracker_info:
                print(f"[DEBUG] Skipping {uav_id} (no active trackers)")
                continue

            print(f"[DEBUG] Processing {uav_id} with {len(tracker_info)} trackers")

            # Convert QImage to numpy array
            width = frame.width()
            height = frame.height()
            ptr = frame.bits()
            ptr.setsize(height * width * 3)
            arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 3))
            frame_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

            # Draw tracker boxes with tracker IDs
            for tracker_id_str, tracker_data in tracker_info.items():
                tracker_id = int(tracker_id_str)
                bbox = tracker_data["bbox"]
                label = tracker_data.get("label", "")

                x, y, w, h = bbox
                x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(width, x2), min(height, y2)

                # Get color for this tracker (generate consistent color based on ID)
                np.random.seed(tracker_id)
                color = tuple(np.random.randint(50, 255, 3).tolist())

                # Draw box
                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 3)

                # Draw tracker ID label
                label_text = f"Tracker[{tracker_id}]"
                if label:
                    label_text += f" {label}"

                (text_width, text_height), baseline = cv2.getTextSize(
                    label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
                )

                # Background for label
                label_y1 = max(y1 - text_height - 6, 0)
                label_y2 = label_y1 + text_height + 6
                cv2.rectangle(frame_bgr, (x1, label_y1), (x1 + text_width + 4, label_y2), color, -1)
                cv2.putText(
                    frame_bgr,
                    label_text,
                    (x1 + 2, label_y2 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            # Add UAV label at top
            cv2.putText(
                frame_bgr,
                uav_id,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2,
                cv2.LINE_AA,
            )

            # Save annotated image
            path = self._save_frame(f"{uav_id}_annotated", QtGui.QImage(
                frame_bgr.data, frame_bgr.shape[1], frame_bgr.shape[0],
                frame_bgr.shape[2] * frame_bgr.shape[1],
                QtGui.QImage.Format_RGB888
            ).copy().rgbSwapped())
            paths.append(path)

        return paths

    def _create_annotated_detection_images(self, target_uavs: Optional[List[str]] = None) -> List[str]:
        """Create images with numbered detection boxes for LLM analysis (deprecated - use _create_annotated_tracking_images)."""
        # Delegate to tracking version for backward compatibility
        return self._create_annotated_tracking_images(target_uavs)

    def _build_detection_matching_prompt(self, user_prompt: str, image_paths: List[str]) -> str:
        """Build prompt for LLM to identify matching detections (deprecated - use _build_tracking_matching_prompt)."""
        # Delegate to tracking version
        return self._build_tracking_matching_prompt(user_prompt, [], image_paths)

    def _build_tracking_matching_prompt(self, user_prompt: str, frames_data: List[Dict], image_paths: List[str]) -> str:
        """Build prompt for LLM to identify matching trackers across multiple frames."""
        # Build frames description
        frames_desc = ""
        if frames_data:
            frames_desc = f"\nTracking data for {len(frames_data)} frames:\n"
            # Group by UAV
            uav_frames = {}
            for frame in frames_data:
                uav_id = frame["uav_id"]
                if uav_id not in uav_frames:
                    uav_frames[uav_id] = []
                uav_frames[uav_id].append(frame)

            for uav_id, frames in uav_frames.items():
                frames_desc += f"\n{uav_id}:\n"
                for frame in frames:
                    frames_desc += f"  Frame {frame['frame_number']}: trackers {list(frame['trackers'].keys())}\n"

        return f"""You are analyzing camera feeds with tracked objects to find which trackers match a description.

User's request: "{user_prompt}"

Each image shows tracked objects with labels like "Tracker[0] car", "Tracker[1] person", etc.

IMPORTANT: Tracker IDs are CONSISTENT across all frames - the same tracker ID tracks the same object throughout the video.{frames_desc}

Your task:
1. Look at the images and identify which tracker IDs BEST match the user's description
2. Consider: object type, color, position, behavior across frames
3. Return ONLY a JSON object with this exact format:
{{
    "matches": {{
        "无人机-01": [0, 2]
    }},
    "reasoning": "Brief explanation of which trackers match and why"
}}

Rules:
- Return tracker IDs (not detection indices)
- If NO trackers match after all frames, return empty arrays
- Tracker IDs are consistent - use them for matching
- UAV IDs are shown in blue text at the top of each image
- Respond with valid JSON only, no other text

Respond with JSON only: """

    def _on_llm_response(self, text: str):
        self.llm_output.append(text)
        self.llm_output.append("")

    def _on_llm_error(self, message: str):
        self.llm_output.append(f"[大模型错误] {message}")
        self.llm_output.append("")

    def _on_evaluation_data_loaded(self, has_data: bool):
        """Handle evaluation tab data load/unload event.

        Args:
            has_data: True if data was loaded, False if cleared
        """
        if has_data:
            # Pause all camera tab UAVs when evaluation data is loaded
            for uav_id in self.camera_tab.uav_ids:
                self.camera_tab.paused_uavs.add(uav_id)
            if self.system_output:
                self.system_output.appendPlainText("[系统] 评估数据已加载，多视角镜头已暂停")
        else:
            if self.system_output:
                self.system_output.appendPlainText("[系统] 评估数据已清除")

    def _enqueue_llm_request(self, prompt: str, image_paths: List[str]):
        # Use a tuple to mark this as regular request: (prompt, image_paths, is_tracking)
        self.llm_queue.append((prompt, image_paths, False))
        if not self.llm_worker or not self.llm_worker.isRunning():
            self._start_next_llm()

    def _start_next_llm(self):
        if not self.llm_queue:
            self.send_btn.setEnabled(True)
            return
        self.send_btn.setEnabled(False)
        prompt, image_paths = self.llm_queue.pop(0)
        self.llm_worker = LlmWorker(self.llm_client, prompt, image_paths)
        self.llm_worker.completed.connect(self._on_llm_response)
        self.llm_worker.failed.connect(self._on_llm_error)
        self.llm_worker.finished.connect(self._start_next_llm)
        self.llm_worker.start()

    def _get_latest_frame_paths(self) -> List[str]:
        frames = self.camera_tab.get_latest_frames()
        ordered = [uav_id for uav_id in self.camera_tab.uav_ids if uav_id in frames]
        return [self._save_frame(uav_id, frames[uav_id]) for uav_id in ordered]

    def _get_multi_frame_paths(self) -> List[str]:
        """Get paths to multiple frames for temporal analysis.

        Returns:
            List of image paths for multi-frame analysis
        """
        frame_count = self.frame_count_spin.value()
        frame_interval = self.frame_interval_spin.value()

        image_paths = []

        # Get frames from each UAV
        for uav_id in self.camera_tab.uav_ids:
            # Check if buffer has sufficient frames
            if not self.frame_buffer.has_sufficient_frames(uav_id, frame_count, frame_interval):
                continue

            # Get recent frames with interval
            frame_entries = self.frame_buffer.get_recent_frames(uav_id, frame_count, frame_interval)

            # Save each frame and collect paths
            for idx, (frame, timestamp, frame_number) in enumerate(frame_entries):
                path = self._save_frame(f"{uav_id}_seq{idx}_f{frame_number}", frame)
                image_paths.append(path)

        return image_paths

    def _get_time_scoped_frame_paths(self, scope) -> List[str]:
        """[MOD 2026-07-09 | 步骤3] 按时间窗(scope=(start_ago,end_ago) 秒)从 FrameBuffer
        取各无人机的历史帧并落盘，供"指定过去时刻/时段"的问答使用。"""
        start_ago, end_ago = scope
        max_frames = self.frame_count_spin.value() if hasattr(self, "frame_count_spin") else 5
        image_paths = []
        for uav_id in self.camera_tab.uav_ids:
            entries = self.frame_buffer.get_frames_by_time(uav_id, start_ago, end_ago, max_frames)
            for idx, (frame, ts, frame_number) in enumerate(entries):
                image_paths.append(self._save_frame(f"{uav_id}_past{int(end_ago)}s_seq{idx}", frame))
        return image_paths

    def _save_frame(self, uav_id: str, image: QtGui.QImage) -> str:
        timestamp = QtCore.QDateTime.currentDateTime().toString("yyyyMMdd_HHmmss_zzz")
        path = self.frame_store_dir / f"{uav_id}_{timestamp}.jpg"
        image.save(str(path), "JPG")
        return str(path)

    def _handle_pause_frame(self, uav_id: str, image: QtGui.QImage):
        """Handle pause frame event with optional multi-frame analysis."""
        # Check if multi-frame analysis is enabled
        if self.multi_frame_cb.isChecked():
            frame_count = self.frame_count_spin.value()
            frame_interval = self.frame_interval_spin.value()
            analysis_type = self.analysis_type_combo.currentText()

            # Check if we have enough frames
            if self.frame_buffer.has_sufficient_frames(uav_id, frame_count, frame_interval):
                # Get multi-frame prompt
                base_prompt = get_auto_pause_prompt(uav_id, is_multi_frame=True, frame_count=frame_count)
                analysis_prompt = get_multi_frame_prompt(analysis_type, frame_count)
                prompt = f"{base_prompt}\n\n{analysis_prompt}"

                self.llm_output.append(f"> [自动-多帧] {uav_id} ({frame_count}帧, {analysis_type})")
                self.llm_output.append("处理中...")
                self.llm_output.append("")

                # Get frame sequence for this UAV
                frame_entries = self.frame_buffer.get_recent_frames(uav_id, frame_count, frame_interval)
                image_paths = []
                for idx, (frame, timestamp, frame_number) in enumerate(frame_entries):
                    path = self._save_frame(f"{uav_id}_paused_seq{idx}_f{frame_number}", frame)
                    image_paths.append(path)

                self._enqueue_llm_request(prompt, image_paths)
                return

        # Fall back to single frame analysis
        prompt = get_auto_pause_prompt(uav_id, is_multi_frame=False)
        self.llm_output.append(f"> [自动] {prompt}")
        self.llm_output.append("处理中...")
        self.llm_output.append("")
        image_path = self._save_frame(f"{uav_id}_paused", image)
        self._enqueue_llm_request(prompt, [image_path])

    def _enqueue_tracking_llm_request(self, prompt: str, image_paths: List[str], target_uavs: Optional[List[str]] = None, user_prompt: str = ""):
        """Enqueue a tracking-specific LLM request."""
        # Store tracking context for response handling
        self._tracking_context = {
            "target_uavs": target_uavs,
            "description": user_prompt  # Store original user prompt, not the LLM prompt
        }
        # Use a tuple to mark this as tracking request: (prompt, image_paths, is_tracking)
        self.llm_queue.append((prompt, image_paths, True))
        if not self.llm_worker or not self.llm_worker.isRunning():
            self._start_next_llm()

    def _on_tracking_llm_response(self, text: str):
        """Handle LLM response for tracking command."""
        print(f"[DEBUG] _on_tracking_llm_response called")
        print(f"[DEBUG] _tracking_context: {self._tracking_context}")
        try:
            # Check if this is ROS mode response
            is_ros_mode = self._tracking_context.get("is_ros_mode", False)
            print(f"[DEBUG] is_ros_mode from context: {is_ros_mode}")

            # Strip markdown code blocks if present
            json_text = text.strip()

            # Remove ```json and ``` markers
            if json_text.startswith("```json"):
                json_text = json_text[7:]
            elif json_text.startswith("```"):
                json_text = json_text[3:]

            if json_text.endswith("```"):
                json_text = json_text[:-3]

            json_text = json_text.strip()

            # Find JSON object in the text - look for outermost braces
            start_idx = json_text.find("{")
            if start_idx == -1:
                raise ValueError("No JSON object found in response")

            # Find matching closing brace by counting nesting
            brace_count = 0
            end_idx = -1
            for i in range(start_idx, len(json_text)):
                if json_text[i] == '{':
                    brace_count += 1
                elif json_text[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end_idx = i
                        break

            if end_idx == -1:
                raise ValueError("Incomplete JSON object (missing closing brace)")

            json_text = json_text[start_idx:end_idx + 1]
            # print(f"[DEBUG] Extracted JSON:\n{json_text}\n")

            # Parse JSON response
            result = json.loads(json_text)
            print(f"[DEBUG] Parsed JSON result: {result}")
            print(f"[DEBUG] Result keys: {list(result.keys())}")
            # Also show in UI for debugging (disabled for cleaner output)
            self.llm_output.append(f"[DEBUG] LLM返回: {json.dumps(result, ensure_ascii=False)}\n")
            self.llm_output.append(f"[DEBUG] 字段列表: {list(result.keys())}\n")

            # Check for new format with "matched" key
            if "matched" in result:
                print(f"[DEBUG] Found 'matched' key in result")
                matches = result["matched"]
                is_matched = result.get("is_matched", False)
                original_description = self._tracking_context.get("description", "")

                print(f"[DEBUG] is_matched={is_matched}, matches={matches}")

                if is_ros_mode:
                    # ROS mode: Publish results to ROS topics
                    target_drone_ids = self._tracking_context.get("target_drone_ids", [])
                    print(f"[DEBUG] ROS mode, target_drone_ids={target_drone_ids}")

                    # Map internal tracker IDs to actual tracker IDs from crops
                    # and publish results for each drone
                    for drone_id in target_drone_ids:
                        print(f"[DEBUG] Checking drone {drone_id} in matches: {drone_id in matches}")
                        if drone_id in matches:
                            tracker_ids = matches[drone_id]
                            # Use LLM's is_matched value
                            tracked = is_matched

                            print(f"[DEBUG] Found matches for {drone_id}: {tracker_ids}")
                            # Publish to ROS
                            self.camera_tab.publish_ros_tracking_result(
                                drone_id, tracked, tracker_ids
                            )

                    # Display summary
                    total_matches = sum(len(tracker_ids) for tracker_ids in matches.values())
                    if total_matches > 0:
                        result_text = f"✓ ROS跟踪结果: 匹配 {total_matches} 个目标\n"
                        for drone_id, tracker_ids in matches.items():
                            if tracker_ids:
                                result_text += f"  • {drone_id}: ID {tracker_ids} -> 已发布到 llm_track_result\n"
                        self.llm_output.append(result_text)
                    else:
                        self.llm_output.append("✓ 未找到匹配的目标\n")
                else:
                    # Local file mode: Apply filter to camera tab
                    target_uavs = self._tracking_context.get("target_uavs")

                    # Filter matches to only include targeted UAVs
                    # If specific UAVs were targeted, only keep matches for those UAVs
                    if target_uavs:
                        matches = {k: v for k, v in matches.items() if k in target_uavs}

                    # Clean up description
                    clean_desc = original_description
                    for prefix in ["track", "tracking", "跟踪", "追踪", "show only", "只显示", "filter", "过滤"]:
                        if clean_desc.lower().startswith(prefix):
                            clean_desc = clean_desc[len(prefix):].strip()
                    import re
                    clean_desc = re.sub(r'^[\w-]+\s+', '', clean_desc)  # Remove drone2/cam02 at start

                    # Apply filter to camera tab (now uses tracker IDs instead of detection indices)
                    self.camera_tab.set_tracking_filter(clean_desc, target_uavs, matches)

                    # Display clean summary
                    total_matches = sum(len(tracker_ids) for tracker_ids in matches.values())
                    if matches:
                        result_text = f"✓ 匹配 {total_matches} 个跟踪器: {clean_desc}\n"
                        for uav_id, tracker_ids in matches.items():
                            if tracker_ids:
                                result_text += f"  • {uav_id}: 跟踪器 {tracker_ids}\n"
                        self.llm_output.append(result_text)
                    else:
                        self.llm_output.append("✓ 未找到匹配的跟踪器\n")
            else:
                # "matched" key not found - try to handle old format or direct drone keys
                print(f"[DEBUG] 'matched' key not found, checking for direct drone keys...")
                print(f"[DEBUG] Available keys: {list(result.keys())}")
                self.llm_output.append(f"[DEBUG] 未找到'matched'字段，检查是否为直接drone格式...\n")

                # Check if result has direct drone keys (old format from buggy LLM)
                drone_keys = [k for k in result.keys() if k.startswith("drone")]
                print(f"[DEBUG] Found drone keys: {drone_keys}")
                if drone_keys:
                    print(f"[DEBUG] Found direct drone keys, converting to matched format")
                    # Convert old format to new format
                    matches = {k: result[k] for k in drone_keys}
                    is_matched = len(matches) > 0
                    # Continue to process as ROS mode...
                    if self.camera_tab.is_ros_mode():
                        target_drone_ids = self._tracking_context.get("target_drone_ids", [])
                        for drone_id in target_drone_ids:
                            if drone_id in matches:
                                tracker_ids = matches[drone_id]
                                print(f"[DEBUG] Publishing for {drone_id}: {tracker_ids}")
                                self.camera_tab.publish_ros_tracking_result(
                                    drone_id, is_matched, tracker_ids
                                )
                    # Display summary
                    total_matches = sum(len(tracker_ids) for tracker_ids in matches.values())
                    if total_matches > 0:
                        result_text = f"✓ ROS跟踪结果: 匹配 {total_matches} 个目标\n"
                        for drone_id,_tracker_ids in matches.items():
                            if tracker_ids:
                                result_text += f"  • {drone_id}: ID {tracker_ids} -> 已发布到 llm_track_result\n"
                        self.llm_output.append(result_text)
                    else:
                        self.llm_output.append("✓ 未找到匹配的目标\n")
                else:
                    print(f"[DEBUG] ERROR: No 'matched' key and no drone keys found in LLM response")
                    print(f"[DEBUG] Response was: {result}")
                    self.llm_output.append("✗ LLM返回的JSON格式不正确\n")
                    self.llm_output.append(f"收到的字段: {list(result.keys())}\n")
                    self.llm_output.append(f"完整内容: {json.dumps(result, ensure_ascii=False)}\n")
        except Exception as e:
            print(f"[ERROR] Failed to process tracking response: {e}")
            import traceback
            traceback.print_exc()
            self.llm_output.append(f"✗ 处理失败: {str(e)}\n")
            self.llm_output.append(f"原始响应: {text[:200]}...\n")

    def _start_next_llm(self):
        if not self.llm_queue:
            self.send_btn.setEnabled(True)
            return
        self.send_btn.setEnabled(False)
        prompt, image_paths, request_type = self.llm_queue.pop(0)

        self.llm_worker = LlmWorker(self.llm_client, prompt, image_paths)

        # Connect appropriate signal handlers based on request type
        if request_type == "intent_analysis":
            self.llm_worker.completed.connect(self._on_intent_analysis_response)
        elif request_type == "ros_crop":
            # ROS crop analysis - not implemented yet, use tracking response
            self.llm_worker.completed.connect(self._on_tracking_llm_response)
        elif request_type is True:  # Old style tracking request
            self.llm_worker.completed.connect(self._on_tracking_llm_response)
        else:  # False or None - regular query
            self.llm_worker.completed.connect(self._on_llm_response)

        self.llm_worker.failed.connect(self._on_llm_error)
        self.llm_worker.finished.connect(self._start_next_llm)
        self.llm_worker.start()

    def _build_system_panel(self):
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(6, 6, 6, 6)
        title = QtWidgets.QLabel("系统输出")
        title.setStyleSheet("font-weight: bold;")
        # system_output is already created in __init__
        layout.addWidget(title)
        layout.addWidget(self.system_output, 1)
        return panel

    def _setup_print_redirect(self):
        """Redirect print statements to system output widget."""
        import sys

        class EmittingStream(QtCore.QObject):
            text_written = QtCore.pyqtSignal(str)

            def write(self, text):
                if text.strip():  # Only emit non-empty text
                    self.text_written.emit(text)

            def flush(self):
                pass

            def isatty(self):
                return False

        # Create emitting stream
        self._emit_stream = EmittingStream()
        self._emit_stream.text_written.connect(
            lambda text: self.system_output.appendPlainText(text.rstrip())
        )

        # Redirect stdout and stderr
        sys.stdout = self._emit_stream
        sys.stderr = self._emit_stream


def main():
    app = QtWidgets.QApplication(sys.argv)
    apply_dark_palette(app)
    default_dir = Path(__file__).resolve().parent / "examples"
    window = MainWindow(data_dir=default_dir)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
