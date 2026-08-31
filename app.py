"""ECG risk screening web app entry point.

This module coordinates ECG preprocessing, model inference, SHAP
explanation, report generation, history persistence, and UI rendering.
"""

import datetime
import html
import json
import os
import uuid

import joblib
import matplotlib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import xgboost as xgb

st.set_page_config(
    page_title="心电风险辅助筛查系统",
    page_icon="static/logo.jpg",  # 改这里
    layout="wide"
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.cnn_inference import predict_abnormal_beats
from src.config_utils import DEFAULT_CONFIG, load_config, resolve_config_paths, save_config
from src.data_loader import load_ecg
from src.feature_extract import extract_all_features, pan_tompkins
from src.inference import FEATURE_ORDER, explain_with_shap, predict_risk
from src.llm_client import GLM_MODEL_OPTIONS, get_glm_api_key, polish_report_with_glm, test_glm_connection
from src.preprocess import preprocess_ecg
from src.report_gen import DEFAULT_THRESHOLDS, generate_report, judge_feature


APP_VERSION = "V2.0"
RISK_COLORS = {"低危": "#52c41a", "中危": "#faad14", "高危": "#ff4d4f"}
RISK_ICONS = {"低危": "✓", "中危": "!", "高危": "!"}
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


# 注入全局样式，统一医疗级页面视觉和组件间距
def inject_styles(theme):
    """注入医疗级轻色 UI CSS，保持业务逻辑不变，仅调整视觉层。"""
    st.markdown(
        """
        <style>
        :root {
            --primary: #1A5CFF;
            --bg: #F5F7FA;
            --card: #FFFFFF;
            --card-soft: #F5F7FA;
            --card-green: #E8F5E9;
            --card-yellow: #FFF7E6;
            --card-red: #FDECEA;
            --text: #1F2937;
            --muted: #667085;
            --line: rgba(31,41,55,0.08);
            --shadow: 0 2px 8px rgba(0,0,0,0.06);
            --radius: 16px;
            --danger-text: #D93025;
            --safe-text: #2E7D32;
            --blue-soft: #E3F2FD;
            --warning-text: #B26A00;
        }
        html, body, [class*="css"] {
            font-family: "Microsoft YaHei", "PingFang SC", sans-serif;
            background: var(--bg);
            color: var(--text);
        }
        .stApp {
            background: var(--bg);
            color: var(--text);
        }
        [data-testid="stAppViewContainer"] {
            max-width: 1400px;
            margin: 0 auto;
        }
        [data-testid="stAppViewContainer"] > .main {
            padding-top: 1rem;
        }
        [data-testid="stSidebar"] {
            background: #FFFFFF;
            border-right: 1px solid var(--line);
            width: 270px !important;
            min-width: 270px !important;
        }
        .block-container {
            max-width: 1400px !important;
            padding-left: 1.2rem !important;
            padding-right: 1.2rem !important;
        }
        [data-testid="stSidebar"] * {
            color: #374151 !important;
        }
        [data-testid="stSidebar"] .stRadio label {
            padding: 10px 12px;
            border-radius: 12px;
            border: 1px solid transparent;
            transition: all 0.2s ease;
        }
        [data-testid="stSidebar"] .stRadio label:hover {
            background: var(--card-soft);
            border-color: var(--line);
        }
        .topbar,
        .glass-panel,
        .section-card,
        .metric-card,
        .history-card,
        .risk-hero,
        .feature-tag,
        .report-box,
        .disclaimer,
        .path-card {
            background: var(--card);
            border: 1px solid var(--line);
            border-radius: var(--radius);
            box-shadow: var(--shadow);
        }
        .topbar {
            padding: 18px 20px;
            margin-bottom: 18px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .topbar-title {
            font-size: 1.85rem;
            font-weight: 800;
            color: var(--text);
        }
        .topbar-meta {
            font-size: 12px;
            color: var(--muted);
            text-align: right;
        }
        .section-title {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            font-size: 1.1rem;
            font-weight: 800;
            color: var(--text);
            margin: 0 0 0.85rem 0;
            padding-left: 0.7rem;
            border-left: 4px solid var(--primary);
        }
        .section-card {
            padding: 1.15rem 1.2rem;
            margin-bottom: 1rem;
        }
        .glass-panel {
            padding: 1.15rem 1.2rem;
            margin-bottom: 1rem;
        }
        .metric-card {
            padding: 1rem 0.9rem;
            min-height: 122px;
            position: relative;
            overflow: hidden;
        }
        .metric-card::before {
            content: "";
            position: absolute;
            width: 54px;
            height: 54px;
            right: -10px;
            top: -12px;
            border-radius: 50%;
            background: rgba(26,92,255,0.08);
        }
        .metric-icon {
            font-size: 1.2rem;
            margin-bottom: 0.35rem;
            color: var(--primary);
            position: relative;
            z-index: 1;
        }
        .metric-label {
            position: relative;
            z-index: 1;
            font-size: 0.74rem;
            color: var(--muted);
            font-weight: 700;
            letter-spacing: 0.04em;
            text-transform: uppercase;
        }
        .metric-value {
            position: relative;
            z-index: 1;
            margin-top: 0.4rem;
            font-size: clamp(1.4rem, 2.2vw, 2rem);
            font-weight: 800;
            color: var(--text);
            font-family: "Consolas", "Courier New", monospace;
        }
        .metric-subtext {
            position: relative;
            z-index: 1;
            margin-top: 0.2rem;
            color: var(--muted);
            font-size: 0.75rem;
        }
        .risk-hero {
            padding: 1.2rem 1.2rem 1rem;
            margin-bottom: 1rem;
        }
        .risk-hero .risk-icon {
            font-size: 1.05rem;
            font-weight: 700;
            letter-spacing: 0.02em;
        }
        .risk-hero .risk-name {
            font-size: clamp(1.8rem, 2.4vw, 2.4rem);
            font-weight: 800;
            margin-top: 0.5rem;
        }
        .risk-hero .risk-score {
            margin-top: 0.4rem;
            color: var(--muted);
            font-size: 0.9rem;
        }
        .path-card {
            padding: 1rem 1rem 0.9rem;
            min-height: 180px;
            border-top: 4px solid var(--primary);
        }
        .path-card.cnn { border-top-color: var(--primary); }
        .path-card.xgb { border-top-color: #7EC8B9; }
        .path-label {
            font-size: 0.72rem;
            font-weight: 800;
            letter-spacing: 0.04em;
            color: var(--primary);
            text-transform: uppercase;
        }
        .path-title {
            font-size: 1.15rem;
            font-weight: 800;
            margin: 0.6rem 0 0.5rem;
            color: var(--text);
        }
        .path-card .metric-value {
            margin-top: 0.3rem;
            font-size: 1.7rem;
        }
        .hint {
            color: var(--muted);
            font-size: 0.8rem;
            line-height: 1.6;
        }
        .compact-progress {
            margin-top: 0.5rem;
        }
        .compact-progress .stProgress > div > div {
            border-radius: 999px;
            background: linear-gradient(90deg, var(--primary) 0%, #6CA4FF 100%);
            height: 10px;
        }
        .feature-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 0.8rem;
            margin-top: 0.8rem;
        }
        .feature-tag {
            padding: 0.8rem 0.75rem;
            border-radius: 12px;
            background: var(--card-soft);
        }
        .feature-tag.abnormal {
            background: #FDECEA;
            border-color: rgba(217,48,37,0.12);
        }
        .feature-tag.safe {
            background: var(--card-soft);
        }
        .feature-tag .name {
            font-size: 0.7rem;
            color: var(--muted);
            margin-bottom: 0.35rem;
        }
        .feature-tag .value {
            font-size: 0.96rem;
            font-weight: 700;
            color: var(--text);
        }
        .feature-tag.abnormal .value {
            color: var(--danger-text);
        }
        .feature-tag.abnormal .hint {
            color: var(--danger-text);
            font-weight: 700;
        }
        .alert-box {
            background: var(--blue-soft);
            border: 1px solid rgba(26,92,255,0.12);
            border-radius: 12px;
            color: #0F3F7B;
            padding: 0.75rem 0.9rem;
            margin-top: 0.8rem;
            line-height: 1.6;
        }
        .report-box {
            background: #FFFFFF;
            border: 1px solid var(--line);
            border-radius: 14px;
            padding: 1.1rem 1.1rem;
            white-space: pre-wrap;
            line-height: 1.8;
            font-family: "Consolas", "Microsoft YaHei", monospace;
            color: var(--text);
            max-height: 420px;
            overflow: auto;
        }
        .report-box strong { color: var(--text); }
        .report-panel {
            background: #F7F9FC;
            border: 1px solid var(--line);
            border-radius: 14px;
            padding: 0.85rem;
            height: 100%;
        }
        .report-panel .stDownloadButton > button,
        .report-panel .stButton > button {
            width: 100%;
            margin-bottom: 0.45rem;
        }
        .keyword-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
            margin-top: 0.55rem;
        }
        .keyword-tag {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            padding: 0.28rem 0.6rem;
            border-radius: 999px;
            background: #EAF2FF;
            color: #234DA4;
            border: 1px solid rgba(26,92,255,0.12);
            font-size: 0.72rem;
            font-weight: 700;
        }
        .history-card-header {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            gap: 0.8rem;
            margin-bottom: 0.6rem;
        }
        .risk-badge-mini {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            padding: 0.28rem 0.58rem;
            border-radius: 999px;
            font-size: 0.7rem;
            font-weight: 700;
        }
        .risk-badge-mini.low { background: #E8F5E9; color: #2E7D32; }
        .risk-badge-mini.medium { background: #FFF7E6; color: #B26A00; }
        .risk-badge-mini.high { background: #FDECEA; color: #D93025; }
        .stForm > div {
            gap: 0.7rem;
        }
        .disclaimer {
            background: #F3F4F6;
            border: 1px solid rgba(31,41,55,0.05);
            border-radius: 12px;
            padding: 0.8rem 0.9rem;
            color: #4B5563;
            line-height: 1.6;
            margin-top: 0.8rem;
        }
        .history-card {
            padding: 1rem 0.9rem;
            margin-bottom: 0.9rem;
        }
        .risk-pill {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-width: 80px;
            padding: 0.35rem 0.7rem;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 700;
        }
        .risk-pill.low { background: var(--card-green); color: var(--safe-text); }
        .risk-pill.medium { background: var(--card-yellow); color: var(--warning-text); }
        .risk-pill.high { background: var(--card-red); color: var(--danger-text); }
        .stTabs [role="tablist"] {
            gap: 0.5rem;
            margin-bottom: 0.8rem;
        }
        .stTabs [role="tab"] {
            border-radius: 12px;
            background: var(--card-soft);
            color: #374151;
            border: 1px solid transparent;
        }
        .stTabs [role="tab"][aria-selected="true"] {
            background: var(--blue-soft);
            color: var(--primary);
            border-color: rgba(26,92,255,0.12);
        }
        .stRadio > div {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
        }
        .stRadio label {
            background: var(--card-soft);
            border: 1px solid var(--line);
            border-radius: 12px;
            padding: 0.6rem 0.8rem;
            font-weight: 600;
        }
        .stRadio label:hover {
            background: var(--blue-soft);
        }
        [data-testid="stExpander"] summary {
            background: var(--card-soft);
            border-radius: 12px;
            padding: 0.75rem 0.9rem;
            font-weight: 700;
            color: var(--text);
        }
        [data-testid="stExpander"] .streamlit-expanderContent {
            background: #FAFBFC;
            border-radius: 0 0 12px 12px;
            padding: 0.8rem 0.9rem;
        }
        .stButton > button {
            border-radius: 10px;
            border: 1px solid rgba(26,92,255,0.25);
            color: var(--primary);
            background: #fff;
            font-weight: 700;
            padding: 0.6rem 1rem;
        }
        .stButton > button:hover {
            border-color: var(--primary);
            background: #F5F8FF;
            box-shadow: 0 4px 12px rgba(26,92,255,0.12);
        }
        .stDownloadButton > button {
            width: 100%;
        }
        .footer {
            color: #7A869B;
            font-size: 12px;
            text-align: center;
            padding: 16px 0 8px;
        }
        @media (max-width: 768px) {
            .topbar {
                display: block;
            }
            .topbar-meta {
                margin-top: 0.5rem;
                text-align: left;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def normalize_config(config):
    result = DEFAULT_UI_CONFIG.copy()
    result.update(config or {})
    return result


def model_status(config):
    return os.path.exists(config.get("model_path", "")), os.path.exists(config.get("cnn_model_path", ""))


def render_header(page, config):
    xgb_ok, cnn_ok = model_status(config)
    status = "模型就绪" if xgb_ok and cnn_ok else "部分模型不可用"
    color = "#389e0d" if xgb_ok and cnn_ok else "#cf1322"
    st.markdown(f"""
    <div class="topbar"><div><div class="topbar-title">♥ 心电风险可解释辅助筛查</div><div class="hint">当前页面：{html.escape(page)}</div></div>
    <div class="topbar-meta"><span style="color:{color};font-weight:800">● {status}</span><br>{APP_VERSION} · 阈值 {config['risk_threshold_medium']:.2f}/{config['risk_threshold_high']:.2f}</div></div>
    """, unsafe_allow_html=True)


def render_sidebar():
    with st.sidebar:
        st.markdown("# ♥ 心电筛查")
        st.caption("辅助分析工作台")
        st.markdown("---")
        pages = ["心电分析", "历史记录", "病情统计", "病例教学", "系统设置", "关于项目"]
        display_map = {
            "心电分析": "♥  心电分析",
            "历史记录": "▣  历史记录",
            "病情统计": "📊 病情统计",
            "病例教学": "📚 病例教学",
            "系统设置": "⚙  系统设置",
            "关于项目": "ⓘ  关于项目",
        }
        default_page = st.session_state.get("current_page", "心电分析")
        safe_default = default_page if default_page in pages else "心电分析"
        selected = st.radio(
            "功能导航",
            pages,
            index=pages.index(safe_default),
            format_func=lambda p: display_map.get(p, p),
            label_visibility="collapsed",
        )
        st.session_state["current_page"] = selected
        st.markdown("---")
        st.caption(f"{APP_VERSION} · 本地运行")
        st.caption("● JSON 存储正常")
    return st.session_state["current_page"]


# 初始化会话状态，保证页面切换和表单状态稳定
def init_state():
    defaults = {
        "analysis_result": None,
        "selected_file": None,
        "selected_file_name": None,
        "history_detail": None,
        "selected_patient": None,
        "patient_name": "",
        "patient_age": None,
        "patient_sex": "未指定",
        "uploaded_filename": None,
        "current_page": "心电分析",
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def mask_patient_name(name):
    """仅在页面/导出层做脱敏，不修改原始磁盘存储数据。"""
    if name is None:
        return "未命名患者"
    text = str(name).strip()
    if not text:
        return "未命名患者"
    if len(text) == 1:
        return text
    return f"{text[0]}{'*' * (len(text) - 1)}"


def extract_abnormal_keywords(features, sex="未指定"):
    """从已有特征数据与 judge_feature 逻辑提取真正的异常关键词。"""
    if not isinstance(features, dict):
        return []
    keywords = []
    for feature_name, value in features.items():
        try:
            severity, _ = judge_feature(feature_name, float(value), DEFAULT_THRESHOLDS, sex=sex)
        except Exception:
            continue
        if "异常" in severity:
            label_map = {
                "HR": "#心率增快",
                "RR_mean": "#RR间期",
                "RR_std": "#RR波动",
                "SDNN": "#SDNN",
                "RMSSD": "#RMSSD",
                "PR": "#PR间期",
                "QRS": "#QRS增宽",
                "QT": "#QT延长",
                "QTc": "#QTc延长",
                "ST_shift": "#ST压低",
                "T_amp": "#T波幅值",
                "P_amp": "#P波幅值",
            }
            display = label_map.get(feature_name, f"#{feature_name}")
            keywords.append(display)
    return keywords[:4]


def sanitize_uploaded_file(uploaded):
    if uploaded is None:
        return False, "请先选择 ECG 文件。"
    try:
        contents = uploaded.getvalue()
    except Exception:
        return False, "文件读取失败，请检查文件格式（支持CSV/TXT/DAT），确认文件未损坏"
    if contents is None or len(contents) == 0:
        return False, "文件为空，请重新上传有效的 ECG 数据。"
    if len(contents) > 10 * 1024 * 1024:
        return False, "文件过大，请上传小于 10MB 的 ECG 数据。"
    return True, ""


def validate_patient_form(patient_name, patient_age, patient_sex):
    name_text = (patient_name or "").strip()
    if name_text:
        if len(name_text) > 50:
            return False, "患者姓名最大长度为 50 个字符。"
    elif patient_name is not None and patient_name.strip() == "":
        return False, "患者姓名不能全为空白字符。"
    try:
        age_value = float(patient_age)
    except (TypeError, ValueError):
        return False, "患者年龄必须为 0-120 之间的数字。"
    if not np.isfinite(age_value) or age_value < 0 or age_value > 120:
        return False, "患者年龄需为 0-120 之间的数字。"
    if patient_sex not in ["男", "女", "未指定"]:
        return False, "性别仅支持“男”“女”“未指定”。"
    return True, ""


def build_export_report(report_text, patient_name=None):
    if not report_text:
        return report_text
    masked_name = mask_patient_name(patient_name)
    prefix = f"患者姓名：{masked_name}\n\n"
    if patient_name is not None and str(patient_name).strip():
        return prefix + report_text
    return report_text


@st.cache_data
def load_case_data():
    try:
        with open("config/cases.json", "r", encoding="utf-8") as file:
            data = json.load(file)
        if isinstance(data, list):
            return data
    except (OSError, json.JSONDecodeError):
        return []
    return []


def get_history(path):
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)
        return data if isinstance(data, list) else []
    except (OSError, json.JSONDecodeError):
        return []


def extract_record_age(record):
    if not isinstance(record, dict):
        return None
    candidates = [record.get("Age"), record.get("age"), record.get("patient_age"), record.get("年龄"), record.get("patientAge")]
    for candidate in candidates:
        if candidate is None or candidate == "" or str(candidate).strip() == "":
            continue
        try:
            value = float(candidate)
            if np.isfinite(value) and 0 <= value <= 120:
                return value
        except (TypeError, ValueError):
            continue
    return None


def normalize_history_record(record):
    if not isinstance(record, dict):
        return {}
    normalized_record = {}
    for key, value in record.items():
        clean_key = str(key).strip()
        if isinstance(value, (list, dict)):
            value = str(value)
        normalized_record[clean_key] = value

    features = normalized_record.get("特征", normalized_record.get("features", {}))
    if not isinstance(features, dict):
        features = {}
    age_value = extract_record_age(normalized_record)
    sex_value = normalized_record.get("Sex") or normalized_record.get("sex") or normalized_record.get("性别") or "未指定"
    normalized = {
        "record_id": normalized_record.get("record_id") or str(uuid.uuid4().hex),
        "时间": normalized_record.get("时间") or normalized_record.get("time") or "",
        "文件名": normalized_record.get("文件名") or normalized_record.get("file_name") or normalized_record.get("filename") or "",
        "风险等级": normalized_record.get("风险等级") or normalized_record.get("risk_level") or "未知",
        "风险评分": normalized_record.get("风险评分") if normalized_record.get("风险评分") is not None else (normalized_record.get("risk_score") if normalized_record.get("risk_score") is not None else 0.0),
        "总心拍数": normalized_record.get("总心拍数") if normalized_record.get("总心拍数") is not None else (normalized_record.get("total_beats") if normalized_record.get("total_beats") is not None else 0),
        "异常心拍数": normalized_record.get("异常心拍数") if normalized_record.get("异常心拍数") is not None else (normalized_record.get("abnormal_count") if normalized_record.get("abnormal_count") is not None else 0),
        "备注": normalized_record.get("备注") or normalized_record.get("note") or "",
        "Age": age_value,
        "age": age_value,
        "patient_age": age_value,
        "Sex": sex_value,
        "sex": sex_value,
        "报告": normalized_record.get("报告") or normalized_record.get("report") or "",
        "特征": features,
        "patient_name": normalized_record.get("patient_name") or normalized_record.get("姓名") or normalized_record.get("备注") or normalized_record.get("文件名") or "未命名患者",
    }
    return normalized


def load_normalized_history(path):
    records = get_history(path)
    normalized = []
    if not records:
        return []
    for record in records:
        try:
            normalized_record = normalize_history_record(record)
            if normalized_record:
                normalized.append(normalized_record)
        except Exception:
            continue
    return normalized


def save_history(records, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(records, file, ensure_ascii=False, indent=2)


def sanitize_history_records(records):
    """强制清理历史记录，避免嵌套列和键名不一致导致 DataFrame 崩溃。"""
    cleaned_records = []
    try:
        for record in records:
            if not isinstance(record, dict):
                continue
            normalized = {}
            for key, value in record.items():
                clean_key = str(key).strip()
                if isinstance(value, (list, dict)):
                    value = str(value)
                normalized[clean_key] = value
            cleaned_records.append(normalized)
        return [normalize_history_record(record) for record in cleaned_records]
    except Exception as exc:
        print("历史记录清理错误：", exc)
        return []


def risk_probabilities(features, config, risk_num, score):
    """界面层读取真实概率，不改变 src.inference 的公共接口。"""
    try:
        model = xgb.XGBClassifier()
        model.load_model(config["model_path"])
        values = np.array([[features[name] for name in FEATURE_ORDER]], dtype=float)
        scaler_path = config.get("scaler_path", "models/ecg_scaler.pkl")
        if os.path.exists(scaler_path):
            values = joblib.load(scaler_path).transform(values)
        probabilities = model.predict_proba(values)[0]
        return [float(probabilities[i]) if i < len(probabilities) else 0.0 for i in range(3)]
    except Exception:
        fallback = [0.0, 0.0, 0.0]
        fallback[risk_num] = max(float(score), 0.0)
        return fallback


def run_analysis(file_path, file_name, config):
    progress = st.progress(0, text="准备开始分析")
    labels = ["上传中", "预处理中", "特征提取中", "CNN 推理中", "XGBoost 推理中", "SHAP 分析中", "生成报告中"]

    def update(index):
        progress.progress(index / 7, text=f"✓ {labels[index - 1]} · 正在处理下一步" if index else "准备开始分析")

    try:
        if not file_path or not os.path.exists(file_path):
            return None, "文件读取失败，请检查文件格式（支持CSV/TXT/DAT），确认文件未损坏"
        file_size = os.path.getsize(file_path)
        if file_size == 0:
            return None, "文件为空，请重新上传有效的 ECG 数据。"
        if file_size > 10 * 1024 * 1024:
            return None, "文件过大，请上传小于 10MB 的 ECG 数据。"
        if not os.path.exists(config.get("model_path", "")):
            return None, "模型文件缺失，请在系统设置中检查模型路径"
        if not os.path.exists(config.get("cnn_model_path", "")):
            return None, "模型文件缺失，请在系统设置中检查模型路径"

        update(1)
        status, signal, fs, message = load_ecg(file_path)
        if status != "success":
            return None, "文件读取失败，请检查文件格式（支持CSV/TXT/DAT），确认文件未损坏"
        update(2)
        status, clean_signal, message = preprocess_ecg(signal, fs)
        if status != "success":
            return None, "预处理失败：信号质量过低，请检查 ECG 数据或重新上传。"
        update(3)
        status, r_peaks, message = pan_tompkins(clean_signal, fs)
        if status != "success":
            return None, "预处理失败：R 峰检测失败，请检查信号质量。"
        status, features, message = extract_all_features(clean_signal, r_peaks, fs)
        if status != "success":
            return None, "分析失败：特征提取失败，请检查 ECG 数据质量。"
        features = {key: 0.0 if value is None else value for key, value in features.items()}
        update(4)
        cnn_status, abnormal_positions, cnn_message = predict_abnormal_beats(clean_signal, r_peaks, fs, config["cnn_model_path"])
        abnormal_positions = [] if abnormal_positions is None else list(abnormal_positions)
        update(5)
        xgb_status, risk_level, risk_num, score, risk_message = predict_risk(features, config["model_path"])
        if xgb_status != "success":
            return None, "模型推理失败，请检查模型文件是否完整。"
        risk_probs = risk_probabilities(features, config, risk_num, score)
        update(6)
        shap_status, shap_result, shap_message = explain_with_shap(features, config["model_path"])
        update(7)
        total_beats = len(r_peaks)
        abnormal_count = len(abnormal_positions)
        report_cnn_status = "abnormal" if abnormal_count else "normal"
        report_status, report_text, report_data = generate_report(risk_num=risk_num, risk_score=score, risk_probs=risk_probs, features=features, abnormal_count=abnormal_count, total_beats=total_beats, sex=config.get("default_sex", "未指定"), cnn_status=report_cnn_status)
        progress.empty()
        return {"file_name": file_name, "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "signal": clean_signal, "fs": fs, "r_peaks": r_peaks, "features": features, "abnormal_positions": abnormal_positions, "cnn_status": cnn_status, "cnn_message": cnn_message, "xgb_status": xgb_status, "risk_level": risk_level, "risk_num": risk_num, "score": score, "risk_probs": risk_probs, "shap_status": shap_status, "shap_result": shap_result, "shap_message": shap_message, "report_status": report_status, "report_text": report_text, "report_data": report_data}, "分析完成"
    except FileNotFoundError:
        progress.empty()
        return None, "模型文件缺失，请在系统设置中检查模型路径"
    except Exception:
        progress.empty()
        return None, "分析失败：数据处理或模型执行异常，请检查文件与配置后重试。"


def plot_layout(fig, height=360):
    fig.update_layout(height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=20, r=20, t=45, b=25), font=dict(family="Microsoft YaHei"), hoverlabel=dict(bgcolor="white", font_size=13))
    fig.update_xaxes(showgrid=True, gridcolor="rgba(120,140,160,.22)", griddash="dot")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(120,140,160,.22)", griddash="dot")
    return fig


def _get_shap_plot_data(shap_dict, feature_values=None):
    """准备 SHAP 图表所需的特征值、目标类别贡献值和基线值。"""
    import shap

    feature_names = [
        name
        for name, item in shap_dict.items()
        if name not in {"__narrative__", "target_class"} and isinstance(item, dict)
    ]
    values = np.array([[float(shap_dict[name]["value"]) for name in feature_names]]) if feature_values is None else np.asarray(feature_values, dtype=float)

    model = xgb.XGBClassifier()
    model.load_model("models/ecg_risk_xgb_model.json")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(values)

    target_class = shap_dict.get("target_class", 0)
    if not isinstance(target_class, int):
        target_class = 0

    if isinstance(shap_values, list):
        if not shap_values:
            raise ValueError("SHAP 模型未返回可用的类别贡献值")
        if target_class >= len(shap_values):
            target_class = len(shap_values) - 1
        target_shap = np.asarray(shap_values[target_class])
        expected_value = explainer.expected_value[target_class]
        base_value = float(expected_value)
    elif np.asarray(shap_values).ndim == 3:
        class_count = np.asarray(shap_values).shape[2]
        if target_class >= class_count:
            target_class = class_count - 1
        target_shap = np.asarray(shap_values)[:, :, target_class]
        base_value = float(np.asarray(explainer.expected_value)[target_class])
    else:
        target_shap = np.asarray(shap_values)
        base_value = float(np.asarray(explainer.expected_value).reshape(-1)[0])

    return feature_names, values, target_shap, base_value


@st.cache_data
def _load_shap_background(feature_names, fallback_values):
    """加载真实多行特征背景，保证依赖图具有可见的样本分布。"""
    try:
        background = pd.read_csv("training_data.csv")
        if all(name in background.columns for name in feature_names):
            matrix = background[list(feature_names)].apply(pd.to_numeric, errors="coerce").dropna().to_numpy(dtype=float)
            if len(matrix) >= 2:
                return matrix
    except (OSError, ValueError, pd.errors.ParserError):
        pass
    base = np.asarray(fallback_values, dtype=float).reshape(1, -1)
    offsets = np.linspace(-0.02, 0.02, 32).reshape(-1, 1)
    scales = np.maximum(np.abs(base), 1.0)
    return base + offsets * scales


@st.cache_data
def build_shap_summary_plot(shap_dict):
    """生成 SHAP 全局蜂群图。"""
    import shap

    feature_names, values, _, _ = _get_shap_plot_data(shap_dict)
    values = _load_shap_background(tuple(feature_names), values)
    _, _, target_shap, _ = _get_shap_plot_data(shap_dict, values)
    fig, ax = plt.subplots(figsize=(8, 5))
    plt.sca(ax)
    try:
        shap.summary_plot(target_shap, values, feature_names=feature_names, plot_type="dot", show=False, ax=ax)
    except TypeError:
        shap.summary_plot(target_shap, values, feature_names=feature_names, plot_type="dot", show=False)
    plt.tight_layout()
    return fig


@st.cache_data
def build_shap_bar_plot(shap_dict):
    """生成 SHAP 特征贡献条形图。"""
    feature_names = [
        name for name, item in shap_dict.items()
        if name not in {"__narrative__", "target_class"} and isinstance(item, dict)
    ]
    contributions = [float(shap_dict[name]["shap_value"]) for name in feature_names]
    order = np.argsort(contributions)
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#1677ff" if contributions[index] < 0 else "#f5222d" for index in order]
    ax.barh([feature_names[index] for index in order], [contributions[index] for index in order], color=colors)
    ax.set_xlabel("SHAP 贡献")
    ax.set_ylabel("特征")
    ax.axvline(0, color="#667085", linewidth=0.8)
    fig.tight_layout()
    return fig


@st.cache_data
def build_shap_dependence_plot(shap_dict):
    """生成 QRS 与 ST_shift 的 SHAP 交互依赖图。"""
    import shap

    feature_names, values, _, _ = _get_shap_plot_data(shap_dict)
    values = _load_shap_background(tuple(feature_names), values)
    _, _, target_shap, _ = _get_shap_plot_data(shap_dict, values)
    if "QRS" not in feature_names or "ST_shift" not in feature_names:
        raise ValueError("缺少 QRS 或 ST_shift 特征，无法生成交互依赖图")
    fig, ax = plt.subplots(figsize=(8, 5))
    plt.sca(ax)
    try:
        shap.dependence_plot("QRS", target_shap, values, feature_names=feature_names, interaction_index="ST_shift", show=False, ax=ax)
    except TypeError:
        shap.dependence_plot("QRS", target_shap, values, feature_names=feature_names, interaction_index="ST_shift", show=False)
    plt.tight_layout()
    return fig


@st.cache_data
def build_shap_decision_plot(shap_dict):
    """生成 SHAP 单样本决策图。"""
    import shap

    feature_names, _, target_shap, base_value = _get_shap_plot_data(shap_dict)
    fig, ax = plt.subplots(figsize=(10, 5))
    plt.sca(ax)
    shap.decision_plot(base_value, target_shap[0], feature_names=feature_names, show=False)
    plt.tight_layout()
    return fig


def feature_group_block(title, feature_names, features, config):
    st.markdown(f"### {title}")
    cols = st.columns(len(feature_names))
    for idx, name in enumerate(feature_names):
        value = features.get(name)
        if value is None:
            continue
        severity, display_text = judge_feature(name, value, DEFAULT_THRESHOLDS, sex=config.get("default_sex", "未指定"))
        is_abnormal = "异常" in severity
        with cols[idx]:
            classes = "feature-tag abnormal" if is_abnormal else "feature-tag safe"
            badge = "异常" if is_abnormal else "正常"
            st.markdown(
                f'<div class="{classes}"><div class="name">{name}</div><div class="value">{display_text}</div><div class="hint" style="margin-top: 0.2rem;">{badge}</div></div>',
                unsafe_allow_html=True,
            )


def render_waveform(result, config):
    signal, fs = np.asarray(result["signal"]), result["fs"]
    indices = np.arange(0, len(signal), max(1, len(signal) // 12000))
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=indices / fs, y=signal[indices], mode="lines", name="清洗信号", line=dict(color="#1A5CFF", width=1.5)))
    peaks = result["r_peaks"]
    if config.get("show_r_peaks", True):
        figure.add_trace(go.Scatter(x=[peak / fs for peak in peaks], y=[signal[peak] for peak in peaks], mode="markers", name="R峰", marker=dict(color="#D93025", size=6)))
    abnormal = result["abnormal_positions"]
    if abnormal:
        figure.add_trace(go.Scatter(x=[peak / fs for peak in abnormal], y=[signal[peak] for peak in abnormal], mode="markers", name="异常心拍", marker=dict(color="#FFB020", size=10, symbol="x")))
    figure.update_layout(
        title="单导联 ECG",
        xaxis_title="时间（秒）",
        yaxis_title="电压",
        hovermode="x unified",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Microsoft YaHei", color="#1F2937"),
        margin=dict(l=20, r=20, t=40, b=25),
    )
    figure.update_xaxes(showgrid=True, gridcolor="rgba(120,140,160,.22)", griddash="dot")
    figure.update_yaxes(showgrid=True, gridcolor="rgba(120,140,160,.22)", griddash="dot")
    st.plotly_chart(plot_layout(figure, 330), use_container_width=True, key="waveform")


def render_ai_polish_panel(report_text):
    if not report_text:
        return

    model_options = GLM_MODEL_OPTIONS
    default_idx = next((idx for idx, item in enumerate(model_options) if item.get("default")), 0)
    selected_model = st.selectbox(
        "AI润色模型（完全免费）",
        options=[item["id"] for item in model_options],
        format_func=lambda model_id: next(item["label"] for item in model_options if item["id"] == model_id),
        index=default_idx,
        key="glm_model_select",
    )
    if st.button("AI润色", type="secondary", use_container_width=True):
        api_key = get_glm_api_key()
        if not api_key:
            st.warning("后台服务未配置，AI润色不可用")
            return
        try:
            with st.spinner("GLM 正在润色报告内容..."):
                polished = polish_report_with_glm(report_text, model_name=selected_model)
            st.session_state["ai_polished_report"] = polished
            st.success("AI润色已生成，可在下方选择最终版本。")
        except Exception:
            st.warning("AI润色不可用，已使用离线建议")
            st.session_state["ai_polished_report"] = report_text

    offline_text = report_text
    ai_text = st.session_state.get("ai_polished_report", offline_text)
    if ai_text == offline_text and not get_glm_api_key():
        return

    version_choice = st.radio("选择最终版本", ["离线建议", "AI润色版本"], index=0, horizontal=True, key="ai_report_version")
    offline_col, ai_col = st.columns(2)
    with offline_col:
        st.markdown("### 离线建议")
        st.markdown(f'<div class="report-box">{html.escape(offline_text)}</div>', unsafe_allow_html=True)
    with ai_col:
        st.markdown("### AI润色结果")
        st.markdown(f'<div class="report-box">{html.escape(ai_text)}</div>', unsafe_allow_html=True)

    if version_choice == "AI润色版本":
        st.session_state["final_report_version"] = "AI"
    else:
        st.session_state["final_report_version"] = "offline"


# 渲染分析结果页面，展示双通路结论和报告内容
def render_results(result, config):
    abnormal_positions = result.get("abnormal_positions") or []
    abnormal_count = len(abnormal_positions)
    risk = result["risk_level"]
    risk_color = {"低危": "#E8F5E9", "中危": "#FFF7E6", "高危": "#FDECEA"}.get(risk, "#F5F7FA")
    risk_text_color = {"低危": "#2E7D32", "中危": "#B26A00", "高危": "#D93025"}.get(risk, "#1F2937")
    patient_name = st.session_state.get("patient_name", "")
    st.markdown('<div class="section-title">核心筛查结论</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="risk-hero" style="background:{risk_color}; border-left: 6px solid {risk_text_color};"><div class="risk-icon" style="color:{risk_text_color};">{RISK_ICONS.get(risk, "!")} 风险结论</div><div class="risk-name" style="color:{risk_text_color};">{risk}</div><div class="risk-score">模型置信评分 {result["score"]:.2f}</div></div>', unsafe_allow_html=True)
    left_panel, right_panel = st.columns([1, 2])
    with left_panel:
        st.markdown('<div class="section-title">双通路分析结果</div>', unsafe_allow_html=True)
        cnn_title = "已完成" if result["cnn_status"] == "success" else "不可用"
        cnn_hint = f'异常心拍 {abnormal_count} 个' if result["cnn_status"] == "success" else result["cnn_message"]
        path_cnn, path_xgb = st.columns(2)
        with path_cnn:
            st.markdown(f'<div class="path-card cnn"><div class="path-label">形态通路 · 1D-CNN</div><div class="path-title">{cnn_title}</div><div class="metric-value">{abnormal_count}</div><div class="hint">{html.escape(cnn_hint)}</div></div>', unsafe_allow_html=True)
        with path_xgb:
            st.markdown(f'<div class="path-card xgb"><div class="path-label">数值通路 · XGBoost</div><div class="path-title">风险分级：{risk}</div><div class="metric-value">{result["score"]:.2f}</div><div class="hint">模型概率：低危 {result["risk_probs"][0]:.1%} · 中危 {result["risk_probs"][1]:.1%} · 高危 {result["risk_probs"][2]:.1%}</div></div>', unsafe_allow_html=True)
        st.markdown('<div class="alert-box">两条通路独立输出，请医护人员结合波形、特征和临床信息综合判断。</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title" style="margin-top:1rem;">模型概率分布</div>', unsafe_allow_html=True)
        probability_columns = st.columns(3)
        for column, (label, probability) in zip(probability_columns, zip(["低危", "中危", "高危"], result["risk_probs"])):
            with column:
                st.markdown(
                    f'<div class="metric-card"><div class="metric-icon">{"●" if label == "低危" else "◆" if label == "中危" else "▲"}</div><div class="metric-label">{label}</div><div class="metric-value">{probability:.1%}</div></div>',
                    unsafe_allow_html=True,
                )
                st.progress(float(probability))
    with right_panel:
        st.markdown('<div class="section-title" style="margin-top:1rem;">波形监测</div>', unsafe_allow_html=True)
        render_waveform(result, config)
    columns = st.columns(4)
    metrics = [("心率", f'{result["features"].get("HR", 0):.1f}', "次/分"), ("总心拍", str(len(result["r_peaks"])), "个"), ("异常心拍", str(abnormal_count), "个"), ("采样率", f'{result["fs"]:.0f}', "Hz")]
    metrics_icons = ["❤", "▣", "⚠", "⏱"]
    for column, ((label, value, unit), icon) in zip(columns, zip(metrics, metrics_icons)):
        with column:
            st.markdown(f'<div class="metric-card"><div class="metric-icon">{icon}</div><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-subtext">{unit}</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title" style="margin-top:1rem;">12 项临床特征</div>', unsafe_allow_html=True)
    groups = {
        "节律稳定性": ["HR", "RR_mean", "RR_std", "SDNN", "RMSSD"],
        "传导功能": ["PR", "QRS"],
        "复极状态": ["QT", "QTc", "ST_shift", "T_amp"],
        "波形形态": ["P_amp"],
    }
    for title, names in groups.items():
        feature_group_block(title, names, result["features"], config)
    if config.get("show_shap", True):
        st.markdown('<div class="section-title" style="margin-top:1rem;">SHAP 可解释性分析</div>', unsafe_allow_html=True)
        if result["shap_status"] == "success":
            shap_view = st.radio("SHAP视图", ["📊特征贡献条形图", "🐝全局蜂群图", "🔄特征交互依赖图", "📈单样本决策力图"], index=0, key="shap_view", horizontal=True)
            if shap_view == "📊特征贡献条形图":
                st.pyplot(build_shap_bar_plot(result["shap_result"]), use_container_width=True)
            elif shap_view == "🐝全局蜂群图":
                st.pyplot(build_shap_summary_plot(result["shap_result"]), use_container_width=True)
            elif shap_view == "🔄特征交互依赖图":
                st.pyplot(build_shap_dependence_plot(result["shap_result"]), use_container_width=True)
            else:
                st.pyplot(build_shap_decision_plot(result["shap_result"]), use_container_width=True)
            st.markdown('<div class="alert-box">SHAP 仅解释 XGBoost 的 12 项数值特征；结果应与波形形态和临床诊断结合解读。</div>', unsafe_allow_html=True)
        else:
            st.warning(f'SHAP 暂不可用：{result["shap_message"]}')
    st.markdown('<div class="section-title" style="margin-top:1rem;">筛查报告</div>', unsafe_allow_html=True)
    if result["report_status"] == "success":
        sanitized_report = build_export_report(result["report_text"], patient_name)
        report_col, action_col = st.columns([3, 1])
        with report_col:
            st.markdown(f'<div class="report-box" style="padding: 1.1rem; background: #fff; border: 1px solid rgba(31,41,55,0.08); border-radius: 14px;">{html.escape(sanitized_report)}</div>', unsafe_allow_html=True)
        with action_col:
            st.markdown('<div class="report-panel">', unsafe_allow_html=True)
            st.download_button("下载 TXT", sanitized_report, "心电筛查报告.txt", "text/plain", use_container_width=True)
            st.download_button("下载报告", sanitized_report, "心电筛查报告-副本.txt", "text/plain", use_container_width=True)
            st.download_button("打印版", sanitized_report, "心电筛查报告-打印版.txt", "text/plain", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        render_ai_polish_panel(sanitized_report)
        st.markdown('<div class="disclaimer">免责声明：本系统仅用于辅助筛查参考，不替代执业医师诊断；异常结果应结合病史、体检和其他检查信息进行判定。</div>', unsafe_allow_html=True)
    else:
        st.error(result["report_text"])


# 渲染心电分析页面，处理表单、上传和分析入口
def analysis_page(config):
    st.title("心电分析")
    st.caption("上传单导联 ECG，完成预处理、双通路分析与可解释报告生成。")
    st.markdown('<div class="disclaimer">所有患者数据仅保存在本地，不会对外上传。</div>', unsafe_allow_html=True)
    result = st.session_state.get("analysis_result")
    if result:
        a, b, c = st.columns([1.2, 1.2, 1.2])
        with a:
            if st.button("重新分析", type="primary", use_container_width=True):
                st.session_state["analysis_result"] = None
                st.rerun()
        with b:
            if st.button("清空重置", use_container_width=True):
                for key in ["analysis_result", "selected_file", "selected_file_name", "patient_name", "patient_age", "patient_sex", "uploaded_filename", "record_note"]:
                    st.session_state.pop(key, None)
                st.rerun()
        with c:
            if st.button("清空结果", use_container_width=True):
                st.session_state["analysis_result"] = None
                st.session_state["selected_file"] = None
                st.session_state["selected_file_name"] = None
                st.rerun()
        render_results(result, config)
        render_save_record(result, config)
        return

    st.markdown('<div class="section-card"><div class="section-title">患者信息</div>', unsafe_allow_html=True)
    patient_name = st.text_input("患者姓名（可选）", value=st.session_state.get("patient_name", ""), max_chars=50, key="patient_name")
    if patient_name is not None and str(patient_name).strip() == "":
        st.caption("患者姓名为空时将按匿名记录展示。")
    patient_age = st.number_input("患者年龄", min_value=0, max_value=120, value=int(st.session_state.get("patient_age") if st.session_state.get("patient_age") is not None else 0), step=1, key="patient_age")
    patient_sex = st.selectbox("性别", ["男", "女", "未指定"], index=["男", "女", "未指定"].index(st.session_state.get("patient_sex", "未指定")), key="patient_sex")
    valid_patient, patient_message = validate_patient_form(patient_name, patient_age, patient_sex)
    if not valid_patient and (patient_name is not None or patient_age is not None or patient_sex is not None):
        st.caption(patient_message)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-card"><div class="section-title">上传心电数据</div>', unsafe_allow_html=True)
    upload, demo = st.columns([3, 1])
    with upload:
        uploaded = st.file_uploader("选择 CSV / TXT / DAT 文件", type=["csv", "txt", "dat"], key="ecg_upload", label_visibility="collapsed")
    with demo:
        demo_clicked = st.button("加载示例数据", use_container_width=True)
    file_path, file_name = None, None
    if demo_clicked:
        file_path, file_name = "uploads/100_30s.csv", "100_30s.csv（示例）"
    elif uploaded is not None:
        ok, file_message = sanitize_uploaded_file(uploaded)
        if not ok:
            st.warning(file_message)
        else:
            try:
                os.makedirs(config["upload_dir"], exist_ok=True)
                file_name = os.path.basename(uploaded.name)
                file_path = os.path.join(config["upload_dir"], file_name)
                with open(file_path, "wb") as file:
                    file.write(uploaded.getbuffer())
            except Exception:
                st.error("文件读取失败，请检查文件格式（支持CSV/TXT/DAT），确认文件未损坏")
    if file_path:
        st.session_state["selected_file"] = (file_path, file_name)
        st.session_state["selected_file_name"] = file_name
        st.session_state["uploaded_filename"] = file_name
    elif st.session_state.get("selected_file"):
        file_name = st.session_state.get("selected_file_name") or st.session_state["selected_file"][1]
    st.markdown("</div>", unsafe_allow_html=True)
    with st.expander("高级设置", expanded=False):
        st.caption("参数保存到配置文件；现有 src 预处理接口保持不变。")
    selected = st.session_state.get("selected_file")
    if not selected:
        st.markdown('<div class="glass-card" style="padding:52px;text-align:center;margin-top:18px"><div style="font-size:46px">♥</div><h3>开始一次清晰的心电筛查</h3><p class="hint">上传心电文件或点击“加载示例数据”开始分析</p></div>', unsafe_allow_html=True)
        return
    st.success(f"已选择：{selected[1]}")
    if st.button("开始分析", type="primary", use_container_width=True):
        if not valid_patient:
            st.caption(patient_message)
            return
        result, message = run_analysis(selected[0], selected[1], config)
        if result:
            st.session_state["analysis_result"] = result
            st.rerun()
        else:
            st.error(message)


# 渲染保存记录模块，处理历史数据落库
def render_save_record(result, config):
    st.markdown('<div class="section-card"><div class="section-title">保存分析记录</div>', unsafe_allow_html=True)
    note = st.text_input("备注（可选）", key="record_note", placeholder="患者编号或检查日期")
    if st.button("保存到历史记录", type="primary"):
        path = config.get("storage_path", "storage/records.json")
        try:
            history = get_history(path)
            patient_age = st.session_state.get("patient_age")
            patient_sex = st.session_state.get("patient_sex", "未指定")
            if patient_age is None or str(patient_age).strip() == "":
                age_value = None
            else:
                try:
                    age_value = float(patient_age)
                except (TypeError, ValueError):
                    age_value = None
            if age_value is not None and (age_value < 0 or age_value > 120):
                age_value = None
            record = {
                "record_id": uuid.uuid4().hex,
                "时间": result["created_at"],
                "文件名": result["file_name"],
                "风险等级": result["risk_level"],
                "风险评分": result["score"],
                "总心拍数": len(result["r_peaks"]),
                "异常心拍数": len(result["abnormal_positions"]),
                "备注": note,
                "Age": age_value,
                "age": age_value,
                "patient_age": age_value,
                "Sex": patient_sex,
                "sex": patient_sex,
                "报告": result.get("report_text", ""),
                "特征": result.get("features", {}),
            }
            history.append(record)
            save_history(history, path)
            st.success("已保存到历史记录")
        except Exception:
            st.error("保存失败：历史记录写入异常，请检查存储目录权限。")
    st.markdown("</div>", unsafe_allow_html=True)


# 渲染历史记录页面，按患者聚合并展示筛查时间线
def render_history_page(config):
    st.title("历史记录")
    path = config.get("storage_path", "storage/records.json")
    records = load_normalized_history(path)
    if not records:
        st.info("暂无历史记录。完成一次分析后，可在分析页保存记录。")
        return

    frame = pd.DataFrame(records)
    for column in ["风险等级", "文件名", "时间", "备注", "Sex", "Age", "patient_name"]:
        if column not in frame:
            frame[column] = ""
    frame["patient_name"] = frame.apply(lambda row: str(row.get("patient_name") or row.get("备注") or row.get("文件名") or "未命名患者"), axis=1)
    patient_names = [name for name in dict.fromkeys(frame["patient_name"].tolist()) if str(name).strip()]
    if not patient_names:
        st.info("暂无有效患者记录。")
        return

    st.caption(f"共 {len(records)} 条记录，按患者归档")
    patient_columns = st.columns(3)
    for index, name in enumerate(patient_names):
        patient_records = frame[frame["patient_name"] == name]
        if patient_records.empty:
            continue
        latest = patient_records.sort_values("时间", na_position="first").iloc[-1]
        risk = str(latest.get("风险等级", "未知"))
        risk_class = "low" if risk == "低危" else "medium" if risk == "中危" else "high" if risk == "高危" else "medium"
        masked_name = mask_patient_name(name)
        features = latest.get("特征", {}) if isinstance(latest.get("特征", {}), dict) else {}
        sex_value = str(latest.get("Sex", "未指定") or "未指定")
        keywords = extract_abnormal_keywords(features, sex_value)
        keyword_html = "".join(f'<span class="keyword-tag">{html.escape(tag)}</span>' for tag in keywords) if keywords else '<span class="keyword-tag">#待更新</span>'
        with patient_columns[index % 3]:
            st.markdown(f"<div class='history-card'><div class='history-card-header'><div><strong>{html.escape(masked_name)}</strong><div class='keyword-row'>{keyword_html}</div></div><span class='risk-badge-mini {risk_class}'>{risk}</span></div><div>{html.escape(str(len(patient_records)))} 条记录 · 最近筛查 {html.escape(str(latest.get('时间', '未知')))}</div></div>", unsafe_allow_html=True)
            if st.button("查看历史", key=f"patient_{index}_{name}", use_container_width=True):
                st.session_state["selected_patient"] = name
                st.rerun()

    selected_patient = st.session_state.get("selected_patient")
    if not selected_patient or selected_patient not in patient_names:
        return

    st.markdown(f"### {html.escape(mask_patient_name(selected_patient))} 的筛查时间线")
    selected_rows = frame[frame["patient_name"] == selected_patient].sort_values("时间", ascending=False)
    for row_index, (_, selected) in enumerate(selected_rows.iterrows()):
        title = f"{selected.get('时间', '未知时间')} · {selected.get('风险等级', '未知')} · {selected.get('文件名', '')}"
        with st.expander(title, expanded=row_index == 0):
            d1, d2, d3 = st.columns(3)
            risk_label = str(selected.get("风险等级", "未知"))
            try:
                risk_score = float(selected.get("风险评分", 0) or 0)
            except (TypeError, ValueError):
                risk_score = 0.0
            d1.markdown(f'<div class="history-card"><div class="metric-label">风险等级</div><div class="metric-value" style="font-size:1.2rem;">{risk_label}</div></div>', unsafe_allow_html=True)
            d2.markdown(f'<div class="history-card"><div class="metric-label">风险评分</div><div class="metric-value" style="font-size:1.2rem;">{risk_score:.2f}</div></div>', unsafe_allow_html=True)
            d3.markdown(f'<div class="history-card"><div class="metric-label">异常心拍</div><div class="metric-value" style="font-size:1.2rem;">{selected.get("异常心拍数", 0)}</div></div>', unsafe_allow_html=True)
            feature_map = selected.get("特征", {}) if isinstance(selected.get("特征", {}), dict) else {}
            if feature_map:
                st.dataframe(pd.DataFrame([feature_map]).T.rename(columns={0: "数值"}), use_container_width=True)
            report = selected.get("报告", "") or f'风险等级：{selected.get("风险等级", "未知")}\n风险评分：{selected.get("风险评分", "")}\n完整图表需重新分析。'
            masked_report = build_export_report(report, selected.get("备注") or selected_patient)
            st.markdown(f'<div class="report-box">{html.escape(masked_report)}</div>', unsafe_allow_html=True)
            confirm_delete = st.checkbox("确认删除？勾选后点击删除按钮", key=f"confirm_delete_{row_index}_{selected.get('record_id', row_index)}")
            delete_col, download_col = st.columns([1, 2])
            with delete_col:
                if st.button("删除记录", key=f"delete_{row_index}_{selected.get('record_id', row_index)}", use_container_width=True, disabled=not confirm_delete):
                    try:
                        records = get_history(path)
                        if isinstance(records, list):
                            filtered = [item for item in records if item.get("record_id") != selected.get("record_id")]
                            save_history(filtered, path)
                            st.success("已删除该条历史记录。")
                            st.rerun()
                    except Exception:
                        st.error("删除失败：历史记录删除异常，请稍后重试。")
            with download_col:
                st.download_button("下载脱敏报告", masked_report, f"心电筛查报告-{row_index + 1}.txt", "text/plain", key=f"download_{row_index}", use_container_width=True)
            st.caption("历史记录保留报告和特征；完整波形与 SHAP 图表需重新分析。")


# 渲染病例教学页面，展示典型 ECG 案例和教学说明
def render_teaching_page():
    st.title("病例教学")
    cases = load_case_data()
    if not cases:
        st.warning("案例数据暂不可用，请检查 config/cases.json 配置。")
        return
    left_panel, right_panel = st.columns([1, 2])
    with left_panel:
        selected_case = st.selectbox(
            "选择典型病例",
            options=cases,
            format_func=lambda item: item.get("title", "未命名病例"),
            key="teaching_case_select",
        )
    with right_panel:
        if not selected_case:
            st.info("暂无案例数据")
            return
        try:
            category = selected_case.get("category", "其他")
            difficulty = selected_case.get("difficulty", "初级")
            risk_level = selected_case.get("risk_level", "中危")
            st.markdown(f"## {selected_case.get('title', '病例')}")
            st.markdown(f"**分类**：{category}   ·   **难度**：{difficulty}   ·   **风险等级**：{risk_level}")
            st.markdown(selected_case.get("description", "无背景描述"))

            param_container = st.container(border=True)
            with param_container:
                st.markdown("### 参数展示")
                feature_dict = selected_case.get("ecg_features", {}) or {}
                if feature_dict:
                    feature_items = list(feature_dict.items())
                    feature_cols = st.columns(len(feature_items))
                    for idx, (name, value) in enumerate(feature_items):
                        with feature_cols[idx]:
                            st.markdown(
                                f'<div class="metric-card"><div class="metric-label">{name}</div><div class="metric-value" style="font-size:1.2rem;">{value}</div></div>',
                                unsafe_allow_html=True,
                            )

            analysis_container = st.container(border=True)
            with analysis_container:
                st.markdown("### 临床分析")
                st.info(selected_case.get("clinical_analysis", "无临床分析内容。"))

            learning_container = st.container(border=True)
            with learning_container:
                st.markdown("### 学习要点")
                learning_points = selected_case.get("learning_points", []) or []
                if learning_points:
                    for point in learning_points:
                        st.markdown(f"- {point}")
                else:
                    st.write("暂无学习要点")

            advice_container = st.container(border=True)
            with advice_container:
                st.markdown("### 处理建议")
                st.success(selected_case.get("treatment_advice", "暂无处理建议。"))

            explain_container = st.container(border=True)
            with explain_container:
                st.markdown("### SHAP 解释")
                st.markdown(f'<div class="alert-box">{selected_case.get("shap_explanation", "无 SHAP 解释。")}</div>', unsafe_allow_html=True)
        except Exception:
            st.error("病例教学数据加载失败，请检查配置文件格式。")


# 渲染病情统计页面，汇总风险趋势和异常特征分布
def render_statistics_page(config):
    st.title("病情统计")
    try:
        path = config.get("storage_path", "storage/records.json")
        raw_records = get_history(path)
        if not raw_records:
            st.info("暂无历史记录。完成一次分析后，返回主页面保存记录即可查看统计图表。")
            return

        records = sanitize_history_records(raw_records)
        if not records:
            st.info("暂无可解析的历史记录数据。")
            return

        try:
            df = pd.DataFrame(records)
        except Exception as exc:
            print("数据解析严重错误：", exc)
            df = None

        if df is None or df.empty:
            st.info("历史记录存在异常格式，已自动跳过错误数据。")
            return

        for column in ["风险等级", "文件名", "时间", "备注", "Sex", "Age", "patient_name"]:
            if column not in df.columns:
                df[column] = ""

        try:
            df["风险等级"] = df["风险等级"].fillna("未知").astype(str)
            df["日期"] = pd.to_datetime(df["时间"], errors="coerce").dt.strftime("%Y-%m-%d")
            risk_counts = df["风险等级"].fillna("未知").astype(str).value_counts().reindex(["低危", "中危", "高危"], fill_value=0)
            pie_fig = px.pie(
                values=risk_counts.values,
                names=risk_counts.index,
                title="风险等级分布",
                color_discrete_sequence=["#66BB6A", "#FFA726", "#EF5350"],
                opacity=0.85,
            )
            pie_fig.update_traces(marker=dict(line=dict(color="#FFFFFF", width=1)))
            pie_fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Microsoft YaHei", color="#1F2937"))
            st.plotly_chart(pie_fig, use_container_width=True)
        except Exception:
            st.warning("风险分布图暂不可用，已跳过此图表。")

        try:
            trend_df = df[df["日期"].notna()].groupby("日期").size().reset_index(name="筛查次数")
            if not trend_df.empty:
                trend_fig = go.Figure()
                trend_fig.add_trace(go.Scatter(x=trend_df["日期"], y=trend_df["筛查次数"], mode="lines+markers", line=dict(color="#1A5CFF", width=2), marker=dict(size=6)))
                trend_fig.update_layout(title="筛查趋势", xaxis_title="日期", yaxis_title="筛查次数", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Microsoft YaHei", color="#1F2937"), margin=dict(l=20, r=20, t=40, b=25))
                st.plotly_chart(trend_fig, use_container_width=True)
            else:
                st.info("暂无有效日期数据，无法绘制趋势图。")
        except Exception:
            st.warning("趋势图暂不可用，已跳过此图表。")

        try:
            abnormal_counter = {}
            for record in records:
                try:
                    feature_map = record.get("特征", {}) if isinstance(record.get("特征", {}), dict) else {}
                    sex = str(record.get("Sex", "未指定") or "未指定")
                    for feature_name, feature_value in feature_map.items():
                        try:
                            severity, _ = judge_feature(str(feature_name), float(feature_value), DEFAULT_THRESHOLDS, sex=sex)
                        except Exception:
                            continue
                        if "异常" in severity:
                            abnormal_counter[str(feature_name)] = abnormal_counter.get(str(feature_name), 0) + 1
                except Exception:
                    continue
            abnormal_df = pd.DataFrame({"特征": list(abnormal_counter.keys()), "异常次数": list(abnormal_counter.values())}).sort_values("异常次数", ascending=False)
            if not abnormal_df.empty:
                anomaly_fig = px.bar(abnormal_df, x="特征", y="异常次数", color_discrete_sequence=["#1A5CFF"])
                anomaly_fig.update_traces(marker_line_color="#1A5CFF", marker_line_width=0, opacity=0.9, marker=dict(color="#1A5CFF", line=dict(color="#1A5CFF", width=0)))
                anomaly_fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Microsoft YaHei", color="#1F2937"), xaxis_title="特征", yaxis_title="异常次数", legend=False, margin=dict(l=20, r=20, t=40, b=25), bargap=0.2)
                anomaly_fig.update_xaxes(showgrid=False)
                anomaly_fig.update_yaxes(showgrid=False)
                st.plotly_chart(anomaly_fig, use_container_width=True)
            else:
                st.info("暂无异常特征数据。")
        except Exception:
            st.warning("异常特征图暂不可用，已跳过此图表。")

        try:
            summary_cards = st.columns(5)
            with summary_cards[0]:
                st.metric("总筛查人数", len(records))
            with summary_cards[1]:
                st.metric("高危人数", int(df["风险等级"].eq("高危").sum()))
            with summary_cards[2]:
                st.metric("中危人数", int(df["风险等级"].eq("中危").sum()))
            with summary_cards[3]:
                st.metric("低危人数", int(df["风险等级"].eq("低危").sum()))

            age_values = []
            for record in records:
                try:
                    value = extract_record_age(record)
                    if value is None:
                        continue
                    age_values.append(float(value))
                except (TypeError, ValueError):
                    continue
            with summary_cards[4]:
                if age_values:
                    st.metric("平均年龄", f"{sum(age_values) / len(age_values):.1f} 岁")
                else:
                    st.metric("平均年龄", "暂无数据")
        except Exception:
            st.warning("汇总卡片部分数据异常，已跳过该模块。")

        try:
            male_count = 0
            female_count = 0
            for record in records:
                sex_value = str(record.get("Sex", "") or "").strip()
                if sex_value == "男":
                    male_count += 1
                elif sex_value == "女":
                    female_count += 1
            if male_count or female_count:
                st.caption(f"男女比例：男 {male_count} : 女 {female_count}")
            else:
                st.caption("男女比例：暂无数据")
        except Exception:
            st.caption("男女比例：暂无数据")

        try:
            high_risk_rows = []
            for record in records:
                try:
                    risk_level = str(record.get("风险等级", "未知"))
                    if risk_level != "高危":
                        continue
                    identifier = record.get("备注") or record.get("patient_name") or record.get("文件名") or "未命名患者"
                    feature_map = record.get("特征", {}) if isinstance(record.get("特征", {}), dict) else {}
                    abnormal_features = []
                    sex = str(record.get("Sex", "未指定") or "未指定")
                    for feature_name, feature_value in feature_map.items():
                        try:
                            severity, _ = judge_feature(str(feature_name), float(feature_value), DEFAULT_THRESHOLDS, sex=sex)
                        except Exception:
                            continue
                        if "异常" in severity:
                            abnormal_features.append(str(feature_name))
                    high_risk_rows.append({
                        "姓名": mask_patient_name(identifier),
                        "日期": str(record.get("时间", "未知")),
                        "风险等级": risk_level,
                        "关键异常特征": ", ".join(abnormal_features[:3]) if abnormal_features else "无明显异常特征",
                    })
                except Exception:
                    continue
            if high_risk_rows:
                st.markdown("### 高危病例列表")
                st.dataframe(pd.DataFrame(high_risk_rows), use_container_width=True)
            else:
                st.info("当前无高危病例记录。")
        except Exception:
            st.info("当前无高危病例记录。")
    except Exception:
        st.error("病情统计加载失败：数据格式异常或历史记录不可用。")


# 渲染系统设置页面，管理模型参数和显示配置
def settings_page(config):
    st.title("系统设置")
    with st.form("settings_form"):
        st.markdown("### 风险与显示"); c1, c2, c3 = st.columns(3); medium = c1.number_input("中危阈值", 0.0, 1.0, float(config["risk_threshold_medium"]), .05); high = c2.number_input("高危阈值", 0.0, 1.0, float(config["risk_threshold_high"]), .05); theme = c3.selectbox("主题", ["医疗蓝", "浅色", "深色"], index=["医疗蓝", "浅色", "深色"].index(config.get("theme", "医疗蓝")))
        st.markdown("### 模型与预处理"); model_path = st.text_input("XGBoost 模型路径", config["model_path"]); cnn_path = st.text_input("CNN 模型路径", config["cnn_model_path"]); band_low, band_high, notch = st.columns(3); low = band_low.number_input("带通下限（Hz）", value=float(config.get("bandpass_low", .5))); upper = band_high.number_input("带通上限（Hz）", value=float(config.get("bandpass_high", 40.0))); notch_freq = notch.selectbox("工频频率", ["auto", "50", "60"], index=["auto", "50", "60"].index(str(config.get("notch_freq", "auto")))); show_shap = st.checkbox("显示 SHAP 图", config.get("show_shap", True))
        st.markdown("### QTc 性别阈值"); q1, q2, q3 = st.columns(3); qtc_male = q1.number_input("男性（ms）", value=int(config.get("qtc_threshold_male", 440))); qtc_female = q2.number_input("女性（ms）", value=int(config.get("qtc_threshold_female", 460))); default_sex = q3.selectbox("默认性别", ["未指定", "男", "女"], index=["未指定", "男", "女"].index(config.get("default_sex", "未指定")))
        st.markdown("### GLM 智谱AI润色服务（完全免费）")
        st.info("当前系统采用安全的后台代理模式，API密钥由系统管理员统一配置，用户无需输入密钥即可使用AI润色功能。")
        for option in GLM_MODEL_OPTIONS:
            st.markdown(f"- {option['label']}：{option['description']}")
        submitted = st.form_submit_button("保存设置", type="primary")
    if submitted:
        if medium >= high: st.error("中危阈值必须小于高危阈值")
        else:
            config.update({"risk_threshold_medium": medium, "risk_threshold_high": high, "theme": theme, "model_path": model_path, "cnn_model_path": cnn_path, "bandpass_low": low, "bandpass_high": upper, "notch_freq": notch_freq, "show_shap": show_shap, "qtc_threshold_male": qtc_male, "qtc_threshold_female": qtc_female, "default_sex": default_sex}); status, message = save_config(config); st.success(message) if status == "success" else st.error(message); st.rerun()
    b1, b2 = st.columns(2)
    with b1:
        if st.button("测试连接"):
            ok, message = test_glm_connection()
            if ok:
                st.success(message)
            else:
                st.warning(message)
    with b2:
        if st.button("恢复默认设置"): save_config(DEFAULT_UI_CONFIG.copy()); st.success("已恢复默认设置，请重新加载页面。")


# 渲染关于页面，说明项目定位与技术栈
def about_page():
    st.title("关于项目")
    st.caption(f"版本 {APP_VERSION}")
    st.markdown('<div class="section-card"><div class="section-title">系统定位</div><p>本系统面向基层医护人员，用于单导联心电数据的风险辅助筛查与结果解释，不替代执业医师诊断。</p><p class="hint">数据处理、模型推理和报告均在本地完成，历史记录使用 JSON 文件保存。</p></div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1: st.markdown('<div class="path-card cnn"><div class="path-label">形态通路</div><div class="path-title">1D-CNN</div><p>按心拍切分波形，定位可能异常的心拍位置。</p></div>', unsafe_allow_html=True)
    with c2: st.markdown('<div class="path-card xgb"><div class="path-label">数值通路</div><div class="path-title">XGBoost + SHAP</div><p>基于 12 项 ECG/HRV 特征完成风险分级，并解释特征贡献。</p></div>', unsafe_allow_html=True)
    st.markdown("### 技术栈"); st.write("Streamlit · Plotly · NumPy · SciPy · WFDB · TensorFlow/Keras · XGBoost · SHAP"); st.warning("本系统仅供辅助筛查参考，不能替代执业医师诊断；异常结果请结合临床信息及时就医。")


# 入口函数，处理页面路由和应用启动流程
def main():
    init_state(); config = normalize_config(resolve_config_paths(load_config())); page = render_sidebar(); inject_styles(config.get("theme", "医疗蓝")); render_header(page, config)
    if page == "心电分析": analysis_page(config)
    elif page == "历史记录": render_history_page(config)
    elif page == "病情统计": render_statistics_page(config)
    elif page == "病例教学": render_teaching_page()
    elif page == "系统设置": settings_page(config)
    else: about_page()
    st.markdown('<div class="footer">辅助筛查工具 · 请由专业医护人员结合临床信息判断 · 本地 JSON 存储</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
