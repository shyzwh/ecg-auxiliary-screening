import os

import streamlit as st

from src.config_utils import DEFAULT_CONFIG


DEFAULT_UI_CONFIG = {
    **DEFAULT_CONFIG,
    "theme": "医疗蓝",
    "default_sex": "未指定",
    "qtc_threshold_male": 440,
    "qtc_threshold_female": 460,
    "qtc_threshold_default": 450,
    "llm_enabled": False,
    "llm_provider": "OpenAI 兼容接口",
    "llm_api_key": "",
    "llm_endpoint": "",
    "llm_model": "",
}


def normalize_config(config):
    """
    把配置项补齐到本地运行所需的绝对路径和默认值。

    参数:
        config: 当前配置字典

    返回:
        标准化后的配置字典
    """
    # 统一把相对路径转换成项目根目录下的绝对路径，避免不同工作目录下运行不一致
    result = DEFAULT_UI_CONFIG.copy()
    result.update(config or {})
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for key in ("model_path", "cnn_model_path", "scaler_path", "data_dir", "upload_dir", "waveform_dir", "report_dir", "suggestions_path", "log_dir"):
        value = result.get(key)
        if value and not os.path.isabs(str(value)):
            result[key] = os.path.join(base_dir, str(value))
    result["storage_path"] = os.path.join(base_dir, "storage", "ecg.db")
    return result


def init_state():
    """初始化 session state 中的前端和分析状态字段。"""
    defaults = {
        "analysis_result": None,
        "selected_file": None,
        "patient_name": "",
        "patient_age": 0,
        "patient_sex": "未指定",
        "patient_height": 170,
        "patient_weight": 65,
        "patient_history": [],
        "patient_medication": "",
        "selected_history_patient": None,
        "polished_report": "",
        "ai_diagnosis": None,
        "rag_answer": "",
        "use_custom_api": False,
        "custom_api_key": "",
        "custom_model": "glm-4-flash-250414",
        "custom_base_url": "https://open.bigmodel.cn/api/paas/v4/chat/completions",
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)
