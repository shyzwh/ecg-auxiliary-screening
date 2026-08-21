# 读 config.json，写 config.json。并且带默认值保护。
import os
import json
from .logger import logger


DEFAULT_CONFIG = {
    "risk_threshold_medium": 0.4,
    "risk_threshold_high": 0.7,
    "default_duration_sec": 30,
    "model_path": "models/ecg_risk_xgb_model.json",
    "cnn_model_path": "models/cnn_model.h5",
    "scaler_path": "models/ecg_scaler.pkl",
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


def load_config(config_path="config.json"):
    """读取配置文件，如果文件缺失或损坏，返回默认配置"""
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
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        logger.info("配置保存成功")
        return "success", "配置保存成功"
    except Exception as e:
        logger.error(f"配置保存失败：{e}")
        return "error", f"配置保存失败：{e}"