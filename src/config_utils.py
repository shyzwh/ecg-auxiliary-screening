# 读配置文件并提供跨平台路径兼容。
import json
import os
from pathlib import Path

from src.logger import logger


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = {
    "risk_threshold_medium": 0.4,
    "risk_threshold_high": 0.7,
    "low_confidence_threshold": 0.8,
    "default_duration_sec": 30,
    "model_path": "models/ecg_risk_xgb_model.json",
    "cnn_model_path": "models/cnn_model.h5",
    "scaler_path": "models/ecg_scaler.pkl",
    "data_dir": "data/mitbih/mit-bih-arrhythmia-database-1.0.0",
    "cnn_beats_path": "cnn_beats.npy",
    "cnn_labels_path": "cnn_labels.npy",
    "cnn_beats_real_path": "cnn_beats_real.npy",
    "cnn_labels_real_path": "cnn_labels_real.npy",
    "storage_path": "storage/records.json",
    "waveform_dir": "storage/waveforms",
    "upload_dir": "uploads",
    "report_dir": "reports",
    "suggestions_path": "config/suggestions.json",
    "log_dir": "logs",
    "notch_freq": "auto",
    "show_shap": True,
    "show_r_peaks": True,
    "bandpass_low": 0.5,
    "bandpass_high": 40.0,
    "medfilt_kernel": 3,
    "qtc_threshold_male": 440,
    "qtc_threshold_female": 460,
    "qtc_threshold_default": 450,
    "qtc_threshold_mild_high": 500,
}


def normalize_path_string(path):
    # 统一删除Windows反斜杠，兼容Linux和Streamlit Cloud。
    if path is None:
        return ""
    return str(path).replace("\\", "/")


def resolve_path(path):
    # 解析相对路径为绝对路径，兼容Windows和Linux。
    path = normalize_path_string(path)
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def resolve_config_paths(config):
    # 解析配置中的文件路径，确保跨平台兼容。
    result = config.copy()
    for key in (
        "model_path",
        "cnn_model_path",
        "scaler_path",
        "data_dir",
        "cnn_beats_path",
        "cnn_labels_path",
        "cnn_beats_real_path",
        "cnn_labels_real_path",
        "storage_path",
        "upload_dir",
        "report_dir",
        "log_dir",
    ):
        if key in result:
            result[key] = normalize_path_string(str(resolve_path(result[key])))
    return result


def load_config(config_path="config.json"):
    # 从config.json读取配置。
    config_path = normalize_path_string(resolve_path(config_path))
    if not os.path.exists(config_path):
        logger.warning("config.json不存在，使用默认配置")
        return DEFAULT_CONFIG.copy()

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        for key, value in DEFAULT_CONFIG.items():
            if key not in config:
                config[key] = value

        logger.info("配置读取成功")
        return config
    except Exception as e:
        logger.error(f"配置读取失败：{e}")
        return DEFAULT_CONFIG.copy()


def save_config(config, config_path="config.json"):
    # 保存配置到config.json，并保持相对路径可移植。
    try:
        config_path = normalize_path_string(resolve_path(config_path))
        portable_config = config.copy()
        for key in (
            "model_path",
            "cnn_model_path",
            "scaler_path",
            "data_dir",
            "cnn_beats_path",
            "cnn_labels_path",
            "cnn_beats_real_path",
            "cnn_labels_real_path",
            "storage_path",
            "upload_dir",
            "report_dir",
            "log_dir",
        ):
            value = portable_config.get(key)
            if value is None:
                continue
            candidate = resolve_path(value).expanduser()
            try:
                portable_config[key] = normalize_path_string(
                    str(candidate.resolve().relative_to(PROJECT_ROOT))
                )
            except ValueError:
                portable_config[key] = normalize_path_string(str(candidate))

        Path(config_path).parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(portable_config, f, ensure_ascii=False, indent=2)
        logger.info("配置保存成功")
        return "success", "配置保存成功"
    except Exception as e:
        logger.error(f"配置保存失败：{e}")
        return "error", f"配置保存失败：{e}"
