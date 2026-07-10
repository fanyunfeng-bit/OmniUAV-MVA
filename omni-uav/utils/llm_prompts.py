"""
LLM prompt templates for multi-frame temporal analysis.
Provides specialized prompts for different types of video analysis.
"""
from typing import Dict


def get_multi_frame_prompt(analysis_type: str, frame_count: int) -> str:
    """Get appropriate prompt for multi-frame analysis.

    Args:
        analysis_type: Type of analysis ("motion", "behavior", "scene_change")
        frame_count: Number of frames being analyzed

    Returns:
        Formatted prompt string for the LLM
    """
    prompts = {
        "motion": _get_motion_analysis_prompt(frame_count),
        "behavior": _get_behavior_analysis_prompt(frame_count),
        "scene_change": _get_scene_change_prompt(frame_count),
    }

    return prompts.get(analysis_type, _get_default_prompt(frame_count))


def _get_motion_analysis_prompt(frame_count: int) -> str:
    """Prompt for motion and movement analysis."""
    return f"""分析这{frame_count}帧连续图像序列中的运动模式。

请关注以下方面：
1. 物体移动方向和速度
2. 运动轨迹和路径
3. 加速或减速趋势
4. 多个物体之间的相对运动
5. 异常或突然的运动变化

请提供详细的运动分析报告。"""


def _get_behavior_analysis_prompt(frame_count: int) -> str:
    """Prompt for behavior and activity analysis."""
    return f"""分析这{frame_count}帧连续图像序列中的行为模式。

请关注以下方面：
1. 人员或车辆的活动类型
2. 行为的持续时间和变化
3. 交互行为（如果有多个对象）
4. 异常或可疑的行为模式
5. 行为的目的或意图推断

请提供详细的行为分析报告。"""


def _get_scene_change_prompt(frame_count: int) -> str:
    """Prompt for scene change and environment analysis."""
    return f"""分析这{frame_count}帧连续图像序列中的场景变化。

请关注以下方面：
1. 场景中新出现或消失的物体
2. 环境条件的变化（光照、天气等）
3. 场景布局或结构的改变
4. 重要的视觉变化或事件
5. 场景的整体演变趋势

请提供详细的场景变化分析报告。"""


def _get_default_prompt(frame_count: int) -> str:
    """Default prompt for general temporal analysis."""
    return f"""分析这{frame_count}帧连续图像序列。

请描述：
1. 图像序列中的主要内容和变化
2. 时间上的演变和趋势
3. 值得注意的事件或模式
4. 任何异常或重要的观察

请提供详细的分析报告。"""


def get_single_frame_prompt(camera_id: str) -> str:
    """Get prompt for single frame analysis (backward compatibility).

    Args:
        camera_id: Camera identifier

    Returns:
        Prompt for single frame analysis
    """
    return f"分析 {camera_id} 前视镜头的当前帧。请描述图像中的主要内容、物体和场景。"


def get_auto_pause_prompt(camera_id: str, is_multi_frame: bool = False, frame_count: int = 1) -> str:
    """Get prompt for automatic pause frame analysis.

    Args:
        camera_id: Camera identifier
        is_multi_frame: Whether this is multi-frame analysis
        frame_count: Number of frames (if multi-frame)

    Returns:
        Appropriate prompt for pause analysis
    """
    if is_multi_frame:
        return f"分析 {camera_id} 暂停前的{frame_count}帧图像序列。请描述场景的演变和任何重要的变化或事件。"
    else:
        return f"分析 {camera_id} 前视暂停帧。请描述图像中的主要内容、物体和场景。"
