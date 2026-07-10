"""
Settings manager for OmniUAV application.
Handles loading, saving, and accessing application settings.
"""
import json
import os
from typing import Any, Dict
from pathlib import Path


class SettingsManager:
    """Manages application settings with JSON persistence."""

    DEFAULT_SETTINGS = {
        # LLM Settings
        "llm": {
            "api_key": "",
            "base_url": "https://api.openai.com/v1",
            "model": "gpt-4o-mini",
            "timeout": 30,
            "auto_analyze_on_pause": True,
        },
        # Detection Settings
        "detection": {
            "model": "fasterrcnn_resnet50",  # fasterrcnn_resnet50, fasterrcnn_mobilenet, fcos_resnet50, retinanet_resnet50
            "confidence_threshold": 0.5,
            "device": "cuda",  # cuda, cpu, mps
            "enabled_by_default": False,
        },
        # Tracking Settings
        "tracking": {
            "padding": 2.5,
            "features": "color",  # gray, color
            "kernel": "linear",  # linear, gaussian
            "lambda_r": 1e-4,
            "enabled_by_default": False,
        },
        # Cross-Camera Tracking Settings
        "cross_camera": {
            "enabled": False,
            "similarity_threshold": 0.7,
            "max_distance": 100,  # pixels in 3D space
            "feature_type": "color_histogram",  # color_histogram, deep_features
        },
        # Multi-Frame LLM Analysis Settings
        "multi_frame_llm": {
            "enabled": False,
            "frame_count": 5,  # Number of frames to analyze together
            "frame_interval": 10,  # Frames between samples
            "analysis_type": "motion",  # motion, behavior, scene_change
        },
        # UI Settings
        "ui": {
            "theme": "dark",
            "default_view": "grid",  # single, grid
            "show_fps": True,
            "show_confidence": True,
        },
        # Data Settings
        "data": {
            "default_data_dir": "examples",
            "output_dir": "outputs",
            "paused_frames_dir": "paused_frames",
            "save_detections": False,
            "save_tracking": False,
        }
    }

    def __init__(self, config_file: str = "configs/config.json"):
        """Initialize settings manager.

        Args:
            config_file: Path to configuration file
        """
        self.config_file = Path(config_file)
        self.settings: Dict[str, Any] = {}
        self.load()

    def load(self):
        """Load settings from file, or create default if not exists."""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    loaded_settings = json.load(f)
                # Merge with defaults to ensure all keys exist
                self.settings = self._merge_settings(self.DEFAULT_SETTINGS, loaded_settings)
            except Exception as e:
                print(f"Error loading settings: {e}")
                self.settings = self.DEFAULT_SETTINGS.copy()
        else:
            self.settings = self.DEFAULT_SETTINGS.copy()
            self.save()

    def save(self):
        """Save current settings to file."""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.settings, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving settings: {e}")

    def get(self, category: str, key: str, default: Any = None) -> Any:
        """Get a setting value.

        Args:
            category: Settings category (e.g., 'llm', 'detection')
            key: Setting key within category
            default: Default value if not found

        Returns:
            Setting value or default
        """
        return self.settings.get(category, {}).get(key, default)

    def set(self, category: str, key: str, value: Any):
        """Set a setting value.

        Args:
            category: Settings category
            key: Setting key within category
            value: Value to set
        """
        if category not in self.settings:
            self.settings[category] = {}
        self.settings[category][key] = value

    def get_category(self, category: str) -> Dict[str, Any]:
        """Get all settings in a category.

        Args:
            category: Settings category

        Returns:
            Dictionary of settings in category
        """
        return self.settings.get(category, {}).copy()

    def set_category(self, category: str, values: Dict[str, Any]):
        """Set all settings in a category.

        Args:
            category: Settings category
            values: Dictionary of settings to set
        """
        self.settings[category] = values.copy()

    def reset_to_defaults(self):
        """Reset all settings to defaults."""
        self.settings = self.DEFAULT_SETTINGS.copy()
        self.save()

    def _merge_settings(self, defaults: Dict, loaded: Dict) -> Dict:
        """Recursively merge loaded settings with defaults.

        Args:
            defaults: Default settings dictionary
            loaded: Loaded settings dictionary

        Returns:
            Merged settings dictionary
        """
        result = defaults.copy()
        for key, value in loaded.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_settings(result[key], value)
            else:
                result[key] = value
        return result


# Global settings instance
_settings_instance = None


def get_settings() -> SettingsManager:
    """Get global settings instance."""
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = SettingsManager()
    return _settings_instance
