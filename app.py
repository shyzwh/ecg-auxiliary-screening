"""ECG辅助筛查主应用，负责上传、分析、历史和报告管理。"""

import datetime
import html
import json
import os
import uuid
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import xgboost as xgb

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

from src.cnn_inference import predict_abnormal_beats
from src.config_utils import DEFAULT_CONFIG, load_config, save_config
from src.data_loader import load_ecg
from src.feature_extract import extract_all_features, pan_tompkins
from src.inference import FEATURE_ORDER, explain_with_shap, predict_risk
from src.llm_client import polish_report_with_glm
from src.preprocess import preprocess_ecg
from src.report_gen import generate_report
from src.ui_components import (
    inject_global_css,
    render_metric_card,
    render_patient_card,
    render_risk_badge,
    render_section_title,
    render_upload_card,
)

st.set_page_config(
    page_title="ECG辅助筛查",
    page_icon="assets/favicon.jpg",
    layout="wide"
)

APP_VERSION = "v2.0.0"
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


# 初始化配置项
def normalize_config(config):
    result = DEFAULT_UI_CONFIG.copy()
    result.update(config or {})
    return result


# 初始化分析状态
def init_state():
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
        "use_custom_api": False,
        "custom_api_key": "",
        "custom_model": "glm-4-flash",
        "custom_base_url": "https://open.bigmodel.cn/api/paas/v4/chat/completions",
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


# 脱敏处理患者姓名
def mask_name(name):
    raw = str(name or "").strip()
    if not raw:
        return "未命名患者"
    if len(raw) == 1:
        return raw[0] + "*"
    if len(raw) == 2:
        return raw[0] + "**"
    return raw[0] + "**"


# 校验患者基本信息
def validate_patient_info(name, age, sex):
    if name is not None and len(str(name).strip()) > 50:
        return False, "姓名不能超过 50 个字符。"
    try:
        age_value = int(age)
    except Exception:
        return False, "年龄必须为整数。"
    if age_value < 0 or age_value > 120:
        return False, "年龄必须在 0-120 岁之间。"
    return True, "ok"


# 获取当前患者画像
def get_patient_profile():
    return {
        "name": st.session_state.get("patient_name", ""),
        "age": int(st.session_state.get("patient_age", 0) or 0),
        "sex": st.session_state.get("patient_sex", "未指定"),
        "height_cm": int(st.session_state.get("patient_height", 170) or 170),
        "weight_kg": float(st.session_state.get("patient_weight", 65) or 65.0),
        "past_history": list(st.session_state.get("patient_history", []) or []),
        "medication": st.session_state.get("patient_medication", ""),
    }


# 保存患者信息到会话状态
def save_patient_profile():
    st.session_state["patient_profile"] = get_patient_profile()


# 加载历史记录列表
def load_history(path):
    path = str(path).replace("\\", "/")
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, list) else []
    except (OSError, json.JSONDecodeError):
        return []


# 保存历史记录到JSON文件
def save_history(records, path):
    path = str(path).replace("\\", "/")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(records, handle, ensure_ascii=False, indent=2)


# 渲染侧边栏导航
def render_sidebar():
    with st.sidebar:
        st.markdown("# ♥ 心电筛查")
        st.caption("辅助分析工作台")
        st.markdown("---")
        page = st.radio(
            "功能导航",
            ["心电分析", "历史记录", "病例教学", "病情统计", "系统设置", "关于项目"],
            label_visibility="collapsed",
        )
        st.markdown("---")
        st.caption(f"{APP_VERSION} · 本地运行")
        st.caption("所有患者数据仅保存在本地，不会对外上传。")
    return page


# 渲染页面标题栏
def render_header(page, config):
    st.markdown(
        """
        <div style="padding:0.9rem 1.2rem;border-radius:18px;background:rgba(255,255,255,0.7);backdrop-filter:blur(20px);border:1px solid rgba(255,255,255,0.8);box-shadow:0 18px 30px rgba(17,39,73,0.08);margin-bottom:1rem;">
            <div style="font-size:1.7rem;font-weight:800;color:#0b2d7a;">♥ 心电风险可解释辅助筛查</div>
            <div style="font-size:0.8rem;color:#667085;">当前页面：{page} · 版本 {version} · 阈值 {med:.2f}/{high:.2f}</div>
        </div>
        """.format(page=html.escape(page), version=APP_VERSION, med=float(config.get("risk_threshold_medium", 0.4)), high=float(config.get("risk_threshold_high", 0.7))),
        unsafe_allow_html=True,
    )


# 计算三级风险概率
def risk_probabilities(features, config, risk_num, score):
    try:
        model = xgb.XGBClassifier()
        model_path = str(config["model_path"]).replace("\\", "/")
        model.load_model(model_path)
        arr = np.array([[features[name] for name in FEATURE_ORDER]], dtype=float)
        scaler_path = str(config.get("scaler_path", "models/ecg_scaler.pkl")).replace("\\", "/")
        if os.path.exists(scaler_path):
            arr = joblib.load(scaler_path).transform(arr)
        probs = model.predict_proba(arr)[0]
        return [float(probs[i]) for i in range(len(probs))]
    except Exception:
        fallback = [0.0, 0.0, 0.0]
        fallback[int(risk_num)] = max(float(score), 0.0)
        return fallback


# 执行完整心电分析流程
def run_analysis(file_path, file_name, config):
    progress = st.progress(0, text="准备开始分析")
    labels = ["上传中", "预处理中", "特征提取中", "CNN 推理中", "XGBoost 推理中", "SHAP 分析中", "生成报告中"]

    def update(index):
        if index == 0:
            progress.progress(0, text="准备开始分析")
        else:
            progress.progress(index / 7, text=f"✓ {labels[index - 1]} · 正在处理下一步")

    update(1)
    status, signal, fs, message = load_ecg(file_path)
    if status != "success":
        progress.empty(); return None, message

    update(2)
    status, clean_signal, message = preprocess_ecg(signal, fs)
    if status != "success":
        progress.empty(); return None, message

    update(3)
    status, r_peaks, message = pan_tompkins(clean_signal, fs)
    if status != "success":
        progress.empty(); return None, message

    status, features, message = extract_all_features(clean_signal, r_peaks, fs)
    if status != "success":
        progress.empty(); return None, message
    features = {key: 0.0 if value is None else value for key, value in features.items()}

    update(4)
    cnn_status, abnormal_positions, cnn_confidence, cnn_message = predict_abnormal_beats(clean_signal, r_peaks, fs, config["cnn_model_path"])
    abnormal_positions = abnormal_positions or []

    update(5)
    xgb_status, risk_level, risk_num, score, risk_message = predict_risk(features, config["model_path"], config=config, sex=st.session_state.get("patient_sex", config.get("default_sex", "未指定")))
    if xgb_status != "success":
        progress.empty(); return None, risk_message
    risk_probs = risk_probabilities(features, config, risk_num, score)

    update(6)
    shap_status, shap_result, shap_message = explain_with_shap(features, config["model_path"])

    update(7)
    report_status, report_text, report_data = generate_report(
        risk_num=risk_num,
        risk_score=score,
        risk_probs=risk_probs,
        features=features,
        abnormal_count=len(abnormal_positions),
        total_beats=len(r_peaks),
        sex=st.session_state.get("patient_sex", config.get("default_sex", "未指定")),
        cnn_status=cnn_status,
    )
    progress.empty()

    patient_info = get_patient_profile()
    return {
        "file_name": file_name,
        "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "patient_info": patient_info,
        "signal": clean_signal,
        "fs": fs,
        "r_peaks": r_peaks,
        "features": features,
        "abnormal_positions": abnormal_positions,
        "cnn_status": cnn_status,
        "cnn_confidence": cnn_confidence,
        "cnn_message": cnn_message,
        "risk_level": risk_level,
        "risk_num": risk_num,
        "score": score,
        "risk_probs": risk_probs,
        "shap_status": shap_status,
        "shap_result": shap_result,
        "shap_message": shap_message,
        "report_status": report_status,
        "report_text": report_text,
        "report_data": report_data,
    }, "分析完成"


# 设置绘图布局样式
def plot_layout(fig, height=360):
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=40, b=20),
        font=dict(family="Microsoft YaHei"),
        hoverlabel=dict(bgcolor="white", font_size=13),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(120,140,160,.22)", griddash="dot")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(120,140,160,.22)", griddash="dot")
    return fig


# 渲染单导联心电波形
def render_waveform(result):
    signal = np.asarray(result["signal"], dtype=float)
    fs = float(result.get("fs", 1) or 1)
    x = np.arange(len(signal)) / fs
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=signal, mode="lines", name="清洗信号", line=dict(color="#1677ff", width=1.6)))
    if result.get("r_peaks"):
        peaks = np.asarray(result["r_peaks"], dtype=int)
        fig.add_trace(go.Scatter(x=peaks / fs, y=signal[peaks], mode="markers", name="R峰", marker=dict(color="#ff4d4f", size=7)))
    if result.get("abnormal_positions"):
        abnormal = np.asarray(result["abnormal_positions"], dtype=int)
        fig.add_trace(go.Scatter(x=abnormal / fs, y=signal[abnormal], mode="markers", name="异常心拍", marker=dict(color="#fa8c16", size=9, symbol="x")))
    fig.update_layout(title="单导联 ECG", xaxis_title="时间（秒）", yaxis_title="电压")
    st.plotly_chart(plot_layout(fig, 420), use_container_width=True)


# 渲染特征指标卡片
def render_feature_card(name, value, severity):
    card_bg = "rgba(255, 235, 238, 0.75)" if "异常" in severity else "rgba(255,255,255,0.8)"
    text_color = "#b42318" if "异常" in severity else "#18324f"
    st.markdown(
        f"<div style='background:{card_bg};padding:0.8rem 0.9rem;border-radius:14px;border:1px solid rgba(181, 30, 24, 0.08);color:{text_color};min-height:86px;'><div style='font-size:0.78rem;font-weight:700;opacity:0.8;'>{name}</div><div style='font-size:1.15rem;font-weight:800;margin-top:0.45rem;'>{float(value):.3f}</div><div style='font-size:0.72rem;font-weight:700;margin-top:0.25rem;'>{severity}</div></div>",
        unsafe_allow_html=True,
    )


# 渲染临床特征分组
def render_clinical_feature_groups(result):
    from src.report_gen import judge_feature, DEFAULT_THRESHOLDS

    groups = {
        "节律稳定性": ["HR", "RR_mean", "RR_std", "SDNN", "RMSSD"],
        "传导功能": ["PR", "QRS"],
        "复极状态": ["QT", "QTc", "ST_shift", "T_amp"],
        "波形形态": ["P_amp"],
    }
    for label, names in groups.items():
        st.subheader(label)
        cols = st.columns(len(names))
        for col, name in zip(cols, names):
            with col:
                value = result.get("features", {}).get(name, 0)
                severity, display_text = judge_feature(name, value, DEFAULT_THRESHOLDS, sex=result.get("patient_info", {}).get("sex", "未指定"))
                render_feature_card(name, value, severity)
        st.markdown("---")


# 渲染报告分段块
def render_report_section(title, body, bg_color="#f5f7ff", text_color="#18324f"):
    st.markdown(
        f"<div style='background:{bg_color};padding:1rem 1.2rem;border-radius:16px;border:1px solid rgba(24,50,79,0.08);color:{text_color};margin:0.7rem 0 1rem 0;'><div style='font-size:1.05rem;font-weight:800;margin-bottom:0.5rem;'>{title}</div><div style='white-space:pre-wrap;line-height:1.8;'>{html.escape(body)}</div></div>",
        unsafe_allow_html=True,
    )


# 渲染心电分析页面
def analysis_page(config):
    st.title("心电分析")
    st.caption("上传单导联 ECG，完成预处理、双通路分析与可解释报告生成。所有患者数据仅保存在本地，不会对外上传。")
    result = st.session_state.get("analysis_result")

    left, right = st.columns([1, 2])
    with left:
        with st.expander("管理个人信息", expanded=False):
            st.session_state["patient_name"] = st.text_input("姓名", value=st.session_state.get("patient_name", ""), max_chars=50)
            st.session_state["patient_age"] = st.number_input("年龄", min_value=0, max_value=120, value=int(st.session_state.get("patient_age", 0)))
            st.session_state["patient_sex"] = st.selectbox("性别", ["未指定", "男", "女"], index=["未指定", "男", "女"].index(st.session_state.get("patient_sex", "未指定")))
            st.session_state["patient_height"] = st.number_input("身高（cm）", min_value=50, max_value=250, value=int(st.session_state.get("patient_height", 170)))
            st.session_state["patient_weight"] = st.number_input("体重（kg）", min_value=10.0, max_value=300.0, value=float(st.session_state.get("patient_weight", 65.0)))
            history_options = ["高血压", "糖尿病", "冠心病", "无"]
            st.session_state["patient_history"] = st.multiselect("既往病史", history_options, default=list(st.session_state.get("patient_history", []) or []))
            st.session_state["patient_medication"] = st.text_input("用药情况", value=st.session_state.get("patient_medication", ""), placeholder="如：阿托伐他汀、降压药")
            if st.button("保存个人信息", use_container_width=True):
                save_patient_profile()
                st.success("个人信息已保存")
        valid, msg = validate_patient_info(st.session_state.get("patient_name", ""), st.session_state.get("patient_age", 0), st.session_state.get("patient_sex", "未指定"))
        if not valid:
            st.error(msg)
        render_patient_card(mask_name(st.session_state.get("patient_name", "")), st.session_state.get("patient_age", 0), st.session_state.get("patient_sex", "未指定"))
        uploaded, demo = render_upload_card()
        if demo:
            st.session_state["selected_file"] = (os.path.join("uploads", "100_30s.csv").replace("\\", "/"), "100_30s.csv（示例）")
        elif uploaded is not None:
            file_name = os.path.basename(uploaded.name)
            file_path = os.path.join(config.get("upload_dir", "uploads"), file_name)
            file_path = file_path.replace("\\", "/")
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, "wb") as handle:
                handle.write(uploaded.getbuffer())
            st.session_state["selected_file"] = (file_path, file_name)
        if st.session_state.get("selected_file"):
            st.success(f"已选择：{st.session_state['selected_file'][1]}")
        if st.button("开始分析", type="primary", use_container_width=True):
            file_info = st.session_state.get("selected_file")
            if not file_info:
                st.error("请先上传 ECG 文件或加载示例数据。")
            elif not valid:
                st.error(msg)
            else:
                result_obj, msg = run_analysis(file_info[0], file_info[1], config)
                if result_obj is not None:
                    st.session_state["analysis_result"] = result_obj
                    st.rerun()
                else:
                    st.error(msg)

    with right:
        if result is None:
            st.info("请在左侧上传 ECG 文件并完成分析。")
            return

        tabs = st.tabs(["核心结论与波形", "临床特征", "SHAP可解释性", "筛查报告"])
        with tabs[0]:
            st.subheader("核心结论")
            risk_color = RISK_COLORS.get(result["risk_level"], "#8c8c8c")
            st.markdown(
                f'<div style="padding:1rem;border-radius:18px;background:linear-gradient(135deg,{risk_color},#0b2d7a);color:white;box-shadow:0 12px 26px rgba(0,0,0,.12);">'
                f'<h3>{result["risk_level"]}</h3><p>风险评分：{result["score"]:.2f}</p></div>',
                unsafe_allow_html=True,
            )
            c1, c2, c3, c4 = st.columns(4)
            with c1: render_metric_card("心率", f"{result['features'].get('HR', 0):.1f}", "次/分")
            with c2: render_metric_card("总心拍", str(len(result["r_peaks"])), "个")
            with c3: render_metric_card("异常心拍", str(len(result["abnormal_positions"])), "个")
            with c4: render_metric_card("采样率", f"{result['fs']:.0f}", "Hz")
            st.markdown("---")
            render_waveform(result)

        with tabs[1]:
            render_section_title("临床特征")
            render_clinical_feature_groups(result)

        with tabs[2]:
            render_section_title("SHAP可解释性")
            if result.get("shap_status") != "success":
                st.warning(result.get("shap_message", "SHAP 暂不可用"))
            else:
                option = st.radio(
                    "SHAP图选择",
                    ["📊特征贡献条形图", "🐝全局蜂群图", "🔄QRS-ST_shift交互图", "📈单样本决策力图"],
                    horizontal=True,
                )
                if option == "📊特征贡献条形图":
                    shap_df = pd.DataFrame(
                        [{"特征": name, "贡献": float(info.get("shap_value", 0.0))} for name, info in result["shap_result"].items()]
                    ).sort_values("贡献")
                    fig = px.bar(shap_df, x="贡献", y="特征", orientation="h", color="贡献", color_continuous_scale=["#1677ff", "#f5222d"])
                    st.plotly_chart(plot_layout(fig, 420), use_container_width=True)
                elif option == "🐝全局蜂群图":
                    training_path = os.path.join(BASE_DIR, "training_data.csv")
                    if os.path.exists(training_path):
                        train_df = pd.read_csv(training_path)
                        if set(FEATURE_ORDER).issubset(train_df.columns):
                            train_data = train_df[FEATURE_ORDER].apply(pd.to_numeric, errors="coerce").dropna()
                            if len(train_data) >= 2:
                                import shap
                                model = xgb.XGBClassifier(); model.load_model(os.path.join(BASE_DIR, "models", "ecg_risk_xgb_model.json"))
                                explainer = shap.TreeExplainer(model)
                                shap_values = explainer(train_data.to_numpy(), check_additivity=False)
                                target_class = int(result.get("risk_num", 0))
                                values = np.asarray(shap_values.values)
                                if values.ndim == 3:
                                    values = values[..., target_class]
                                fig, ax = plt.subplots(figsize=(10, 5))
                                shap.summary_plot(values, train_data.to_numpy(), feature_names=FEATURE_ORDER, show=False)
                                st.pyplot(fig)
                            else:
                                st.error("training_data.csv 样本过少，无法生成蜂群图。")
                        else:
                            st.error("training_data.csv 缺少完整 ECG 特征列。")
                    else:
                        st.error("未找到 training_data.csv，无法渲染全局蜂群图。")
                elif option == "🔄QRS-ST_shift交互图":
                    training_path = os.path.join(BASE_DIR, "training_data.csv")
                    if os.path.exists(training_path):
                        train_df = pd.read_csv(training_path)
                        if {"QRS", "ST_shift"}.issubset(train_df.columns):
                            train_data = train_df[FEATURE_ORDER].apply(pd.to_numeric, errors="coerce").dropna()
                            if len(train_data) >= 2:
                                import shap
                                model = xgb.XGBClassifier(); model.load_model(os.path.join(BASE_DIR, "models", "ecg_risk_xgb_model.json"))
                                explainer = shap.TreeExplainer(model)
                                shap_values = explainer(train_data.to_numpy(), check_additivity=False)
                                values = np.asarray(shap_values.values)
                                target_class = int(result.get("risk_num", 0))
                                if values.ndim == 3:
                                    values = values[..., target_class]
                                fig, ax = plt.subplots(figsize=(8, 5))
                                shap.dependence_plot("QRS", values, train_data.to_numpy(), feature_names=FEATURE_ORDER, interaction_index="ST_shift", show=False, ax=ax)
                                st.pyplot(fig)
                            else:
                                st.error("训练数据样本不足，交互图为空白。")
                        else:
                            st.error("训练数据缺少 QRS/ST_shift 列，无法渲染交互图。")
                    else:
                        st.error("未找到 training_data.csv，无法渲染交互图。")
                else:
                    training_path = os.path.join(BASE_DIR, "training_data.csv")
                    if os.path.exists(training_path):
                        train_df = pd.read_csv(training_path)
                        if set(FEATURE_ORDER).issubset(train_df.columns):
                            import shap
                            train_data = train_df[FEATURE_ORDER].apply(pd.to_numeric, errors="coerce").dropna().head(100)
                            model = xgb.XGBClassifier(); model.load_model(os.path.join(BASE_DIR,"models", "ecg_risk_xgb_model.json"))
                            explainer = shap.TreeExplainer(model)
                            shap_values = explainer(train_data.to_numpy(), check_additivity=False)
                            values = np.asarray(shap_values.values)
                            target_class = int(result.get("risk_num", 0))
                            if values.ndim == 3:
                                values = values[..., target_class]
                            base_value = float(np.asarray(shap_values.base_values).reshape(-1)[0]) if hasattr(shap_values, 'base_values') else 0.0
                            fig = plt.figure(figsize=(10, 5))
                            shap.decision_plot(base_value, values, feature_names=FEATURE_ORDER, show=False)
                            st.pyplot(fig)
                        else:
                            st.error("training_data.csv 缺少完整特征列，无法生成决策力图。")
                    else:
                        st.error("未找到 training_data.csv，无法渲染决策力图。")

        with tabs[3]:
            render_section_title("筛查报告")
            report_data = result.get("report_data", {})
            offline_report = result.get("report_text", "")
            if st.button("AI润色", use_container_width=True):
                try:
                    custom_api_key = st.session_state.get("custom_api_key", "") or None
                    custom_model = st.session_state.get("custom_model", "glm-4-flash") or "glm-4-flash"
                    custom_base_url = st.session_state.get("custom_base_url", "") or None
                    polished = polish_report_with_glm(offline_report, api_key=custom_api_key, model=custom_model, base_url=custom_base_url)
                    st.session_state["polished_report"] = polished
                    st.success("已生成 AI 润色版本")
                except Exception as exc:
                    st.session_state["polished_report"] = offline_report
                    st.warning(f"AI 润色失败，已回退到离线建议：{exc}")
            padded = st.session_state.get("polished_report", offline_report)
            version = st.radio("选择版本", ["离线建议", "AI润色版"], horizontal=True)
            left_col, right_col = st.columns(2)
            with left_col:
                st.markdown("### 离线建议")
                risk_summary = report_data.get("risk_summary") or ""
                etiology_analysis = report_data.get("etiology_analysis") or ""
                lifestyle_advice = report_data.get("lifestyle_advice") or ""
                doctor_communication = report_data.get("doctor_communication") or ""
                disclaimer = report_data.get("disclaimer") or "本报告仅供参考，不能替代专业医生诊断。"
                render_report_section("风险结论", risk_summary or offline_report.split("综合建议：", 1)[0][:300], "#f6f7fb")
                render_report_section("病因分析", etiology_analysis or "病因分析：请结合具体参数评估。", "#fff2f0")
                render_report_section("生活建议", lifestyle_advice or "生活建议：请保持规律作息并复查。", "#edfff2")
                render_report_section("医生沟通话术", doctor_communication or "医生沟通话术：请结合临床症状进行复核。", "#eef6ff")
                render_report_section("免责声明", disclaimer, "#f3f4f6")
            with right_col:
                st.markdown("### AI润色版")
                ai_text = padded if version == "AI润色版" else offline_report
                ai_report_data = {k: v for k, v in report_data.items() if isinstance(v, str)}
                risk_summary_ai = ai_report_data.get("risk_summary") or ai_text
                etiology_analysis_ai = ai_report_data.get("etiology_analysis") or ai_text
                lifestyle_advice_ai = ai_report_data.get("lifestyle_advice") or ai_text
                doctor_communication_ai = ai_report_data.get("doctor_communication") or ai_text
                disclaimer_ai = ai_report_data.get("disclaimer") or "本报告仅供参考，不能替代专业医生诊断。"
                render_report_section("风险结论", risk_summary_ai, "#f6f7fb")
                render_report_section("病因分析", etiology_analysis_ai, "#fff2f0")
                render_report_section("生活建议", lifestyle_advice_ai, "#edfff2")
                render_report_section("医生沟通话术", doctor_communication_ai, "#eef6ff")
                render_report_section("免责声明", disclaimer_ai, "#f3f4f6")
            if version == "离线建议":
                st.download_button("下载离线报告", offline_report, "心电筛查报告.txt", "text/plain", use_container_width=True)
            else:
                st.download_button("下载 AI 润色版", padded, "心电筛查报告_ai.txt", "text/plain", use_container_width=True)

            if st.button("保存到历史记录", type="primary", use_container_width=True):
                history_path = config.get("storage_path", "storage/records.json")
                records = load_history(history_path)
                records.append({
                    "record_id": uuid.uuid4().hex,
                    "时间": result["created_at"],
                    "文件名": result["file_name"],
                    "风险等级": result["risk_level"],
                    "风险评分": float(result["score"]),
                    "总心拍数": len(result["r_peaks"]),
                    "异常心拍数": len(result["abnormal_positions"]),
                    "备注": st.session_state.get("record_note", ""),
                    "报告": result.get("report_text", ""),
                    "特征": result.get("features", {}),
                    "patient_name": st.session_state.get("patient_name", ""),
                    "patient_info": result.get("patient_info", {}),
                })
                save_history(records, history_path)
                st.success("已保存到历史记录")


# 渲染历史记录卡片
def render_patient_history_cards(records):
    grouped = {}
    for item in records:
        patient_name = str((item.get("patient_info") or {}).get("name") or item.get("patient_name") or "未命名患者").strip() or "未命名患者"
        grouped.setdefault(patient_name, []).append(item)
    if not grouped:
        return

    names = list(grouped.keys())
    for idx in range(0, len(names), 3):
        cols = st.columns(3)
        for i, name in enumerate(names[idx:idx + 3]):
            with cols[i]:
                series = grouped[name]
                latest = sorted(series, key=lambda row: row.get("时间", ""), reverse=True)[0]
                risk = latest.get("风险等级", "未知")
                tags = []
                for row in series:
                    if row.get("异常心拍数", 0) > 0:
                        tags.append("异常心拍")
                    if row.get("风险等级") in {"中危", "高危"}:
                        tags.append(row["风险等级"])
                st.markdown(
                    f"""
                    <div class="glass-panel" style="min-height:0;">
                        <h3>{mask_name(name)}</h3>
                        <div style="color:#667085; font-size:0.8rem;">记录数：{len(series)}</div>
                        <div style="color:#667085; font-size:0.8rem;">最近时间：{latest.get('时间', '')}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                render_risk_badge(risk)
                if tags:
                    st.write(" ".join(f"<span style='display:inline-block;padding:4px 8px;border-radius:999px;background:#eef5ff;color:#0b2d7a;font-size:12px;margin:0 4px 4px 0;'>{tag}</span>" for tag in sorted(set(tags))[:3]), unsafe_allow_html=True)
                if st.button("查看历史", key=f"view_history_{name}_{idx}_{i}", use_container_width=True):
                    st.session_state["selected_history_patient"] = name
                    st.rerun()


# 渲染历史记录页面
def history_page(config):
    records = load_history(config.get("storage_path", "storage/records.json"))
    st.title("历史记录")
    st.caption("患者姓名仅在展示层脱敏处理，原始资料保存在本地 JSON 文件中。")
    if not records:
        st.info("暂无历史记录。完成一次分析后，可在分析页保存记录。")
        return

    render_patient_history_cards(records)
    st.markdown("---")
    selected_patient = st.session_state.get("selected_history_patient")
    if selected_patient:
        patient_records = [
            item for item in records
            if str((item.get("patient_info") or {}).get("name") or item.get("patient_name") or "未命名患者").strip() == selected_patient
        ]
        if patient_records:
            patient_records = sorted(patient_records, key=lambda row: row.get("时间", ""), reverse=True)
            for row in patient_records:
                with st.expander(f"{row.get('时间', '')} · {row.get('文件名', '')}", expanded=False):
                    st.write(f"风险等级：{row.get('风险等级', '未知')} | 风险评分：{float(row.get('风险评分', 0)):.2f}")
                    if row.get("备注"):
                        st.caption(f"备注：{row.get('备注')}")
                    if row.get("特征"):
                        feature_df = pd.DataFrame([row["特征"]]).T.reset_index().rename(columns={"index": "特征", 0: "数值"})
                        st.dataframe(feature_df, use_container_width=True, hide_index=True)
                    report_text = row.get("报告", "") or ""
                    st.markdown(f'<div class="report-box">{html.escape(report_text)}</div>', unsafe_allow_html=True)
                    record_id = row.get("record_id") or f"record_{row.get('时间', '')}_{idx}"
                    st.download_button("导出报告", report_text, "心电筛查历史报告.txt", "text/plain", key=f"download_{record_id}", use_container_width=True)
                    if st.button("删除该记录", key=f"delete_record_{row.get('record_id') or idx}", use_container_width=True):
                        if st.session_state.get(f"delete_confirm_{row.get('record_id')}"):
                            records = [item for item in records if item.get("record_id") != row.get("record_id")]
                            save_history(records, config.get("storage_path", "storage/records.json"))
                            st.session_state["selected_history_patient"] = None
                            st.session_state.pop(f"delete_confirm_{row.get('record_id')}", None)
                            st.rerun()
                        else:
                            st.session_state[f"delete_confirm_{row.get('record_id')}"] = True
                            st.warning("再次点击确认删除。")


# 渲染系统设置页面
def settings_page(config):
    st.title("系统设置")
    with st.form("settings_form"):
        st.markdown("### 风险与显示")
        c1, c2, c3 = st.columns(3)
        medium = c1.number_input("中危阈值", 0.0, 1.0, float(config.get("risk_threshold_medium", 0.4)), 0.05)
        high = c2.number_input("高危阈值", 0.0, 1.0, float(config.get("risk_threshold_high", 0.7)), 0.05)
        theme = c3.selectbox("主题", ["医疗蓝", "浅色", "深色"], index=["医疗蓝", "浅色", "深色"].index(config.get("theme", "医疗蓝")))
        st.markdown("### 模型与预处理")
        model_path = st.text_input("XGBoost 模型路径", config.get("model_path", "models/ecg_risk_xgb_model.json"))
        cnn_model_path = st.text_input("CNN 模型路径", config.get("cnn_model_path", "models/cnn_model.h5"))
        show_shap = st.checkbox("显示 SHAP 图", value=config.get("show_shap", True))

        st.markdown("### 大模型API设置")
        st.caption("系统默认使用管理员配置的后台API。如需使用自己的API密钥，可在下方填写。")
        st.session_state["use_custom_api"] = st.checkbox("使用自定义API", value=st.session_state.get("use_custom_api", False))
        st.session_state["custom_api_key"] = st.text_input("API密钥", value=st.session_state.get("custom_api_key", ""), type="password")
        st.session_state["custom_model"] = st.text_input("模型名称", value=st.session_state.get("custom_model", "glm-4-flash"))
        st.session_state["custom_base_url"] = st.text_input("API接口地址", value=st.session_state.get("custom_base_url", "https://open.bigmodel.cn/api/paas/v4/chat/completions"))
        submitted = st.form_submit_button("保存设置", type="primary")

    if st.button("测试连接", use_container_width=True):
        try:
            from src.llm_client import test_glm_connection
            ok, msg = test_glm_connection(
                model_name=st.session_state.get("custom_model", "glm-4-flash"),
                api_key=st.session_state.get("custom_api_key", ""),
                base_url=st.session_state.get("custom_base_url", ""),
            )
            if ok:
                st.success(msg)
            else:
                st.warning(msg)
        except Exception as exc:
            st.warning(f"测试失败：{exc}")

    if submitted:
        if medium >= high:
            st.error("中危阈值必须小于高危阈值")
        else:
            config.update({
                "risk_threshold_medium": medium,
                "risk_threshold_high": high,
                "theme": theme,
                "model_path": model_path,
                "cnn_model_path": cnn_model_path,
                "show_shap": show_shap,
            })
            save_config(config)
            st.success("设置保存成功")
            st.rerun()


# 渲染病例教学页面
def cases_page():
    cases_path = os.path.join(BASE_DIR, "config", "cases.json").replace("\\", "/")
    if not os.path.exists(cases_path):
        st.warning("未找到案例数据文件 config/cases.json")
        return
    with open(cases_path, "r", encoding="utf-8") as handle:
        cases = json.load(handle)
    st.title("病例教学")
    selected_title = st.selectbox("选择病例", [case["title"] for case in cases])
    selected = next(case for case in cases if case["title"] == selected_title)
    st.caption(f"类别：{selected.get('category', '未分类')} · 难度：{selected.get('difficulty', '未知')} · 风险：{selected.get('risk_level', '未知')}")
    st.write(selected.get("description", ""))
    c1, c2 = st.columns(2)
    with c1:
        render_section_title("ECG 特征")
        st.json(selected.get("ecg_features", {}))
    with c2:
        render_section_title("临床分析")
        st.write(selected.get("clinical_analysis", ""))
    render_section_title("学习要点")
    for item in selected.get("learning_points", []):
        st.markdown(f"- {item}")
    render_section_title("处理建议")
    st.write(selected.get("treatment_advice", ""))


# 渲染病情统计页面
def statistics_page():
    records_path = os.path.join(BASE_DIR, "storage", "records.json").replace("\\", "/")
    records = load_history(records_path)
    st.title("病情统计")
    if not records:
        st.info("暂无筛查记录，完成一次分析后即可查看统计。")
        return

    frame = pd.DataFrame(records)
    if "风险等级" not in frame:
        frame["风险等级"] = "未知"
    summary = frame["风险等级"].value_counts().reindex(["低危", "中危", "高危"], fill_value=0)
    pie = px.pie(names=summary.index.tolist(), values=summary.values.tolist(), color=summary.index.tolist(), color_discrete_map=RISK_COLORS)
    st.plotly_chart(plot_layout(pie, 300), use_container_width=True)


# 渲染项目说明页面
def about_page():
    st.title("关于项目")
    st.markdown("本系统面向基层医护人员，支持单导联 ECG 风险辅助筛查、双通路推理和可解释分析，不替代执业医师诊断。")
    st.warning("所有患者数据仅保存在本地，不会对外上传。")


# 执行主程序入口
def main():
    inject_global_css()
    init_state()
    config = normalize_config(load_config())
    page = render_sidebar()
    render_header(page, config)
    if page == "心电分析":
        analysis_page(config)
    elif page == "历史记录":
        history_page(config)
    elif page == "病例教学":
        cases_page()
    elif page == "病情统计":
        statistics_page()
    elif page == "系统设置":
        settings_page(config)
    else:
        about_page()


if __name__ == "__main__":
    main()
