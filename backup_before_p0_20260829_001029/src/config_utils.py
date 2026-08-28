# 读 config.json，写 config.json。并且带默认值保护。
import os
import json
from pathlib import Path
from .logger import logger


PROJECT_ROOT = Path(__file__).resolve().parent.parent


DEFAULT_CONFIG = {
    "risk_threshold_medium": 0.4,
    "risk_threshold_high": 0.7,
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
    "upload_dir": "uploads",
    "report_dir": "reports",
    "log_dir": "logs",
    "notch_freq": "auto",
    "show_shap": True,
    "show_r_peaks": True,
    "bandpass_low": 0.5,
    "bandpass_high": 40.0,
    "medfilt_kernel": 3
}


def resolve_path(path):
    """将配置中的相对路径解析到项目根目录，绝对路径仍可由配置覆盖。"""
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def resolve_config_paths(config):
    """解析所有文件系统配置项，配置文件本身仍保持可移植的相对路径。"""
    result = config.copy()
    for key in (
        "model_path", "cnn_model_path", "scaler_path", "data_dir",
        "cnn_beats_path", "cnn_labels_path", "cnn_beats_real_path",
        "cnn_labels_real_path", "storage_path", "upload_dir", "report_dir",
        "log_dir",
    ):
        if key in result:
            result[key] = str(resolve_path(result[key]))
    return result


def load_config(config_path="config.json"):
    """读取配置文件，如果文件缺失或损坏，返回默认配置"""
    config_path = resolve_path(config_path)
    if not os.path.exists(config_path):
        logger.warning("config.json不存在，使用默认配置")
        return DEFAULT_CONFIG.copy()

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        # 补全缺失字段
        for key, value in DEFAULT_CONFIG.items():
            if key not in config:
                config[key] = value

        logger.info("配置读取成功")
        return config
    except Exception as e:
        logger.error(f"配置读取失败：{e}")
        return DEFAULT_CONFIG.copy()


def save_config(config, config_path="config.json"):
    """保存配置到文件"""
    try:
        config_path = resolve_path(config_path)
        portable_config = config.copy()
        for key in (
            "model_path", "cnn_model_path", "scaler_path", "data_dir",
            "cnn_beats_path", "cnn_labels_path", "cnn_beats_real_path",
            "cnn_labels_real_path", "storage_path", "upload_dir", "report_dir",
            "log_dir",
        ):
            value = portable_config.get(key)
            if value is None:
                continue
            candidate = resolve_path(value).expanduser()
            try:
                portable_config[key] = str(candidate.resolve().relative_to(PROJECT_ROOT))
            except ValueError:
                portable_config[key] = str(candidate)

        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(portable_config, f, ensure_ascii=False, indent=2)
        logger.info("配置保存成功")
        return "success", "配置保存成功"
    except Exception as e:
        logger.error(f"配置保存失败：{e}")
        return "error", f"配置保存失败：{e}"