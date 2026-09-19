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
from scipy.interpolate import make_interp_spline

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

from src.config_utils import DEFAULT_CONFIG, load_config, save_config
from src.inference import FEATURE_ORDER
from src.llm_client import answer_with_rag, generate_ai_diagnosis, get_last_ai_error, polish_report_with_glm
from service.analysis_service import run_analysis, risk_probabilities
from service.config_service import init_state, normalize_config
from service.history_service import delete_history_record, load_history, save_history
from service.patient_service import (
    get_patient_profile,
    mask_name,
    save_patient_profile,
    validate_patient_info,
)
from src.ui_components import (
    inject_global_css,
    render_metric_card,
    render_feature_card as render_ui_feature_card,
    render_info_card as render_ui_info_card,
    render_patient_card,
    render_risk_badge,
    render_section_title,
    render_stat_card as render_ui_stat_card,
    render_upload_card,
    render_warning_card as render_ui_warning_card,
)

st.set_page_config(
    page_title="ECG辅助筛查",
    page_icon="assets/favicon.jpg",
    layout="wide"
)

APP_VERSION = "v2.1.0"
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


# 渲染侧边栏导航
# 只负责页面入口和版本信息展示，不处理业务状态更新

def render_sidebar():
    """
    渲染侧边栏导航和运行信息。

    返回:
        当前页面名称
    """
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
        # 统一展示当前应用版本，避免各页面版本号漂移
        st.caption(f"{APP_VERSION} · 本地运行")
        st.caption("所有患者数据仅保存在本地，不会对外上传。")
    return page


# 渲染页面标题栏
# 需要附带当前页面、版本和阈值信息，便于快速复核

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


# 设置绘图布局样式
# 让所有 Plotly 图表保持统一风格，减少页面审美差异

def plot_layout(fig, height=360):
    """
    统一设置图表样式，保证所有绘图使用相同网格和底色。

    参数:
        fig: Plotly 图对象
        height: 图表高度

    返回:
        格式化后的图对象
    """
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
# 直接从分析结果中输出清洗信号、R峰和异常心拍位置

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
# 使用异常状态决定背景色，快速识别高风险特征

def render_feature_card(name, value, severity):
    """
    按特征名称和值渲染一张卡片。

    参数:
        name: 特征名称
        value: 特征数值
        severity: 判读状态，如 正常 / 轻度异常
    """
    card_bg = "rgba(255, 235, 238, 0.75)" if "异常" in severity else "rgba(255,255,255,0.8)"
    text_color = "#b42318" if "异常" in severity else "#18324f"
    st.markdown(
        f"<div style='background:{card_bg};padding:0.8rem 0.9rem;border-radius:14px;border:1px solid rgba(181, 30, 24, 0.08);color:{text_color};min-height:86px;'><div style='font-size:0.78rem;font-weight:700;opacity:0.8;'>{name}</div><div style='font-size:1.15rem;font-weight:800;margin-top:0.45rem;'>{float(value):.3f}</div><div style='font-size:0.72rem;font-weight:700;margin-top:0.25rem;'>{severity}</div></div>",
        unsafe_allow_html=True,
    )


# 渲染临床特征分组
# 根据节律、传导、复极和波形四类组织展示特征

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
# 按统一风格输出结构化说明，便于患者和医生查看

def render_report_section(title, body, bg_color="#f5f7ff", text_color="#18324f"):
    """
    按卡片样式渲染报告段落。

    参数:
        title: 段落标题
        body: 段落内容
        bg_color: 背景色
        text_color: 文本色
    """
    st.markdown(
        f"<div style='background:{bg_color};padding:1rem 1.2rem;border-radius:16px;border:1px solid rgba(24,50,79,0.08);color:{text_color};margin:0.7rem 0 1rem 0;'><div style='font-size:1.05rem;font-weight:800;margin-bottom:0.5rem;'>{title}</div><div style='white-space:pre-wrap;line-height:1.8;'>{html.escape(body)}</div></div>",
        unsafe_allow_html=True,
    )


def get_llm_request_settings():
    """只在启用自定义 API 时返回前端覆盖项，否则使用后台默认配置。"""
    if not st.session_state.get("use_custom_api", False):
        return None, None, None
    api_key = st.session_state.get("custom_api_key", "").strip() or None
    model = st.session_state.get("custom_model", "").strip() or None
    base_url = st.session_state.get("custom_base_url", "").strip() or None
    return api_key, model, base_url


# 渲染心电分析页面
# 负责上传、校验、推理和报告展示的主流程

def analysis_page(config):
    st.title("心电分析")
    st.caption("上传单导联 ECG，完成预处理、双通路分析与可解释报告生成。所有患者数据仅保存在本地，不会对外上传。")
    result = st.session_state.get("analysis_result")

    left = st.container()
    right = st.container()
    with left:
        with st.expander("管理个人信息", expanded=True):
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
        symptoms = st.text_area(
            "症状描述（可选）",
            value=st.session_state.get("symptoms", ""),
            placeholder="如：最近一周胸闷、心悸，运动后加重",
            max_chars=200,
        )
        st.session_state["symptoms"] = symptoms
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
                progress = st.progress(0, text="准备开始分析")
                labels = ["上传中", "预处理中", "特征提取中", "CNN 推理中", "XGBoost 推理中", "SHAP 分析中", "生成报告中"]

                def update_progress(index):
                    if index == 0:
                        progress.progress(0, text="准备开始分析")
                    else:
                        progress.progress(index / 7, text=f"✓ {labels[index - 1]} · 正在处理下一步")

                result_obj, msg = run_analysis(
                    file_info[0],
                    file_info[1],
                    config,
                    patient_info=get_patient_profile(),
                    progress_callback=update_progress,
                    symptoms=st.session_state.get("symptoms", ""),
                )
                progress.empty()
                if result_obj is not None:
                    st.session_state["analysis_result"] = result_obj
                    st.rerun()
                else:
                    st.error(msg)

    with right:
        tabs = st.tabs(["核心结论与波形", "临床特征", "SHAP可解释性", "筛查报告"])
        if result is None:
            with tabs[0]:
                st.info("请先上传数据并分析。")
            return

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
            render_ui_info_card("波形概览", "清洗后的单导联ECG波形、R峰和异常心拍定位结果。", "⌁")
            render_waveform(result)

        with tabs[1]:
            render_section_title("临床特征")
            render_ui_info_card("12项临床特征", "基于心率、传导、复极和节律稳定性指标展示判读结果。", "▥")
            render_clinical_feature_groups(result)

        with tabs[2]:
            render_section_title("SHAP可解释性")
            render_ui_info_card("模型解释", "通过特征贡献、全局分布、交互关系和决策力图辅助理解XGBoost输出。", "◈")
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
            render_ui_info_card("报告版本", "可切换查看离线建议或AI润色版，下载和保存功能保持可用。", "▤")
            if st.button("AI润色", use_container_width=True):
                try:
                    custom_api_key = st.session_state.get("custom_api_key", "") or None
                    custom_model = st.session_state.get("custom_model", "glm-4-flash-250414") or "glm-4-flash-250414"
                    custom_base_url = st.session_state.get("custom_base_url", "") or None
                    polished = polish_report_with_glm(offline_report, api_key=custom_api_key, model=custom_model, base_url=custom_base_url, symptoms=result.get("symptoms", ""))
                    if polished is None:
                        st.session_state["polished_report"] = offline_report
                        st.warning("AI服务暂时不可用，请稍后重试；已回退到离线建议")
                    else:
                        st.session_state["polished_report"] = polished
                        st.success("已生成 AI 润色版本")
                except Exception as exc:
                    st.session_state["polished_report"] = offline_report
                    st.warning(f"AI 润色失败，已回退到离线建议：{exc}")
            padded = st.session_state.get("polished_report", offline_report)
            version = st.radio("选择版本", ["离线建议", "AI润色版"], horizontal=True)
            if version == "离线建议":
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
                st.download_button("下载离线报告", offline_report, "心电筛查报告.txt", "text/plain", use_container_width=True)
            else:
                st.markdown("### AI润色版")
                ai_report_data = {k: v for k, v in report_data.items() if isinstance(v, str)}
                risk_summary_ai = ai_report_data.get("risk_summary") or padded
                etiology_analysis_ai = ai_report_data.get("etiology_analysis") or padded
                lifestyle_advice_ai = ai_report_data.get("lifestyle_advice") or padded
                doctor_communication_ai = ai_report_data.get("doctor_communication") or padded
                disclaimer_ai = ai_report_data.get("disclaimer") or "本报告仅供参考，不能替代专业医生诊断。"
                render_report_section("风险结论", risk_summary_ai, "#f6f7fb")
                render_report_section("病因分析", etiology_analysis_ai, "#fff2f0")
                render_report_section("生活建议", lifestyle_advice_ai, "#edfff2")
                render_report_section("医生沟通话术", doctor_communication_ai, "#eef6ff")
                render_report_section("免责声明", disclaimer_ai, "#f3f4f6")
                st.download_button("下载 AI 润色版", padded, "心电筛查报告_ai.txt", "text/plain", use_container_width=True)

            st.markdown("---")
            render_section_title("AI智能解读")
            st.caption("AI解读用于辅助理解筛查结果，不能替代执业医师诊断。")
            if st.button("生成AI智能解读", use_container_width=True):
                llm_api_key, llm_model, llm_base_url = get_llm_request_settings()
                diagnosis = generate_ai_diagnosis(
                    result,
                    api_key=llm_api_key,
                    model=llm_model,
                    base_url=llm_base_url,
                    symptoms=result.get("symptoms", ""),
                )
                if diagnosis is None:
                    st.session_state["ai_diagnosis"] = None
                    st.error("AI服务暂时不可用，请稍后重试")
                else:
                    st.session_state["ai_diagnosis"] = diagnosis
                    st.success("已生成AI智能解读")
            diagnosis = st.session_state.get("ai_diagnosis")
            if diagnosis:
                render_report_section("风险总结", diagnosis.get("risk_summary", ""), "#f6f7fb")
                render_report_section("病因分析", diagnosis.get("etiology_analysis", ""), "#fff2f0")
                render_report_section("SHAP归因解读", diagnosis.get("shap_interpretation", ""), "#eef6ff")
                render_report_section("生活建议", diagnosis.get("lifestyle_advice", ""), "#edfff2")
                render_report_section("紧急情况提示", diagnosis.get("emergency_warning", ""), "#fff1f0")
                feature_explanations = diagnosis.get("feature_explanations") or []
                if feature_explanations:
                    st.markdown("#### 特征解释")
                    st.dataframe(pd.DataFrame(feature_explanations), use_container_width=True, hide_index=True)

            render_section_title("问AI")
            question = st.text_input("请输入关于本次心电结果或常见心电疾病的问题", key="rag_question")
            if st.button("提问", key="ask_rag", use_container_width=True):
                llm_api_key, llm_model, llm_base_url = get_llm_request_settings()
                answer = answer_with_rag(question, api_key=llm_api_key, model=llm_model, base_url=llm_base_url)
                if answer:
                    st.session_state["rag_answer"] = answer
                else:
                    st.session_state["rag_answer"] = "AI服务暂时不可用，请稍后重试；请先参考离线报告，并咨询专业医护人员。"
            if st.session_state.get("rag_answer"):
                render_report_section("AI回答", st.session_state["rag_answer"], "#eef6ff")

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
                    "symptoms": result.get("symptoms", ""),
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
                with st.container(border=True):
                    st.markdown(f"### {mask_name(name)}")
                    st.caption(f"记录数：{len(series)} · 最近时间：{latest.get('时间', '暂无数据')}")
                    render_risk_badge(risk)
                    if tags:
                        tag_html = " ".join(
                            f"<span style='display:inline-block;padding:4px 8px;border-radius:999px;background:#eef5ff;color:#0b2d7a;font-size:12px;margin:0 4px 4px 0;'>{html.escape(tag)}</span>"
                            for tag in sorted(set(tags))[:3]
                        )
                        st.markdown(tag_html, unsafe_allow_html=True)
                    else:
                        st.caption("暂无异常关键词")
                    if st.button("查看历史", key=f"view_history_{name}_{idx}_{i}", use_container_width=True):
                        st.session_state["selected_history_patient"] = name
                        st.rerun()


# 渲染历史记录页面
def history_page(config):
    records = load_history(config.get("storage_path", "storage/records.json"))
    st.title("历史记录")
    st.caption("患者姓名仅在展示层脱敏处理，原始资料保存在本地 SQLite 数据库中。")
    if not records:
        render_section_title("暂无记录", "ℹ")
        render_ui_info_card("开始一次心电筛查", "完成分析并点击保存到历史记录后，患者记录会显示在这里。", "▤")
        return

    render_section_title("患者记录", "👥")
    render_patient_history_cards(records)
    st.markdown("---")
    selected_patient = st.session_state.get("selected_history_patient")
    if selected_patient:
        render_section_title(f"{mask_name(selected_patient)} 的筛查时间线", "↗")
        patient_records = [
            item for item in records
            if str((item.get("patient_info") or {}).get("name") or item.get("patient_name") or "未命名患者").strip() == selected_patient
        ]
        if patient_records:
            patient_records = sorted(patient_records, key=lambda row: row.get("时间", ""), reverse=True)
            for row in patient_records:
                record_id = row.get("record_id") or f"record_{row.get('时间', '')}_{row.get('文件名', '')}"
                risk = row.get("风险等级", "未知")
                score = float(row.get("风险评分", 0) or 0)
                with st.container(border=True):
                    header_cols = st.columns([2, 1])
                    with header_cols[0]:
                        st.markdown(f"**{row.get('时间', '暂无日期')} · {row.get('文件名', '暂无文件名')}**")
                    with header_cols[1]:
                        render_risk_badge(risk, score)
                    feature_values = row.get("特征") if isinstance(row.get("特征"), dict) else {}
                    hr = feature_values.get("HR", "暂无数据")
                    abnormal_count = row.get("异常心拍数", "暂无数据")
                    st.markdown(f"**关键指标**　心率：{hr}　异常心拍数：{abnormal_count}")
                    st.markdown(f"**症状描述**　{row.get('symptoms') or '未填写'}")
                    if row.get("备注"):
                        st.caption(f"备注：{row.get('备注')}")
                    report_text = row.get("报告", "") or "暂无报告"
                    st.markdown("**完整报告**")
                    st.markdown(report_text)
                    st.download_button("导出报告", report_text, "心电筛查历史报告.txt", "text/plain", key=f"download_{record_id}", use_container_width=True)
                    if st.button("删除该记录", key=f"delete_record_{record_id}", use_container_width=True):
                        if st.session_state.get(f"delete_confirm_{record_id}"):
                            delete_history_record(row.get("record_id"), config.get("storage_path", "storage/records.json"))
                            st.session_state["selected_history_patient"] = None
                            st.session_state.pop(f"delete_confirm_{record_id}", None)
                            st.rerun()
                        else:
                            st.session_state[f"delete_confirm_{record_id}"] = True
                            st.warning("再次点击确认删除。")
                st.divider()


# 渲染系统设置页面
def settings_page(config):
    st.title("系统设置")
    with st.form("settings_form"):
        with st.container(border=True):
            render_section_title("基础设置", "⚙")
            c1, c2, c3 = st.columns(3)
            medium = c1.number_input("中危阈值", 0.0, 1.0, float(config.get("risk_threshold_medium", 0.4)), 0.05)
            high = c2.number_input("高危阈值", 0.0, 1.0, float(config.get("risk_threshold_high", 0.7)), 0.05)
            theme_options = ["医疗蓝", "清新绿", "暖阳橙"]
            current_theme = config.get("theme", "医疗蓝")
            theme = c3.selectbox("主题", theme_options, index=theme_options.index(current_theme) if current_theme in theme_options else 0)

        with st.container(border=True):
            render_section_title("模型路径", "◉")
            model_path = st.text_input("XGBoost 模型路径", config.get("model_path", "models/ecg_risk_xgb_model.json"))
            cnn_model_path = st.text_input("CNN 模型路径", config.get("cnn_model_path", "models/cnn_model.h5"))
            show_shap = st.checkbox("显示 SHAP 图", value=config.get("show_shap", True))

        with st.container(border=True):
            render_section_title("大模型API", "✦")
            st.caption("系统默认使用管理员配置的后台API。如需使用自己的API密钥，可在下方填写。")
            st.session_state["use_custom_api"] = st.checkbox("使用自定义API", value=st.session_state.get("use_custom_api", False))
            st.session_state["custom_api_key"] = st.text_input("API密钥", value=st.session_state.get("custom_api_key", ""), type="password")
            st.session_state["custom_model"] = st.text_input("模型名称", value=st.session_state.get("custom_model", "glm-4-flash-250414"))
            st.session_state["custom_base_url"] = st.text_input("API接口地址", value=st.session_state.get("custom_base_url", "https://open.bigmodel.cn/api/paas/v4/chat/completions"))
        submitted = st.form_submit_button("保存设置", type="primary")

    if st.button("测试连接", use_container_width=True):
        try:
            from src.llm_client import test_glm_connection
            ok, msg = test_glm_connection(
                model_name=st.session_state.get("custom_model", "glm-4-flash-250414"),
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
            st.session_state["theme"] = theme
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
    selector_col, content_col = st.columns([1, 2])
    with selector_col:
        render_section_title("病例列表", "▤")
        selected_title = st.selectbox("选择病例", [case["title"] for case in cases], label_visibility="collapsed")
        selected = next(case for case in cases if case["title"] == selected_title)
        render_ui_info_card(
            "病例概览",
            f"类别：{selected.get('category', '未分类')}\n难度：{selected.get('difficulty', '未知')}\n风险：{selected.get('risk_level', '未知')}",
            "ℹ",
        )
    with content_col:
        render_ui_info_card("病例描述", selected.get("description", "暂无数据"), "▣")
        detail_cols = st.columns(2)
        with detail_cols[0]:
            ecg_text = json.dumps(selected.get("ecg_features", {}), ensure_ascii=False, indent=2)
            render_ui_info_card("ECG 特征", f"<pre>{html.escape(ecg_text)}</pre>", "⌁")
        with detail_cols[1]:
            render_ui_info_card("临床分析", selected.get("clinical_analysis", "暂无数据"), "✚")
        points = selected.get("learning_points", []) or []
        numbered_points = "<br>".join(f"{index}. {html.escape(str(item))}" for index, item in enumerate(points, 1)) or "暂无数据"
        render_ui_info_card("学习要点", numbered_points, "✓")
        render_ui_info_card("处理建议", selected.get("treatment_advice", "暂无数据"), "➜", "ecg-green-card")


# 渲染病情统计页面
def statistics_page():
    records = load_history()
    st.title("病情统计")
    if not records:
        st.info("暂无筛查记录，完成一次分析后即可查看统计。")
        return

    from src.report_gen import DEFAULT_THRESHOLDS, judge_feature

    def patient_info(record):
        info = record.get("patient_info")
        return info if isinstance(info, dict) else {}

    def display_name(record):
        info = patient_info(record)
        return mask_name(info.get("name") or record.get("patient_name") or "")

    def abnormal_features(record):
        info = patient_info(record)
        sex = info.get("sex", "未指定")
        features = record.get("特征")
        if not isinstance(features, dict):
            return []
        abnormal = []
        for name, value in features.items():
            try:
                severity, display_text = judge_feature(name, float(value), DEFAULT_THRESHOLDS, sex=sex)
            except (TypeError, ValueError):
                continue
            if "异常" in severity:
                abnormal.append(f"{name}：{display_text}")
        return abnormal

    risk_levels = [str(record.get("风险等级") or "未知") for record in records]
    risk_counts = {level: risk_levels.count(level) for level in ["低危", "中危", "高危"]}
    unique_patients = set()
    ages = []
    sex_counts = {"男": 0, "女": 0}
    for record in records:
        info = patient_info(record)
        identity = info.get("name") or record.get("patient_name") or "未命名患者"
        unique_patients.add(str(identity))
        try:
            age = float(info.get("age"))
            if 0 <= age <= 120:
                ages.append(age)
        except (TypeError, ValueError):
            pass
        sex = str(info.get("sex") or "")
        if sex in sex_counts:
            sex_counts[sex] += 1

    render_section_title("患者摘要", "▥")
    summary_cols = st.columns(4)
    with summary_cols[0]:
        render_ui_stat_card("总筛查人数", len(unique_patients), "人")
    with summary_cols[1]:
        render_ui_stat_card("高危数", risk_counts["高危"], "例")
    with summary_cols[2]:
        render_ui_stat_card("平均年龄", f"{sum(ages) / len(ages):.1f}" if ages else "暂无数据", "岁")
    total_sex = sex_counts["男"] + sex_counts["女"]
    sex_ratio = f"{sex_counts['男']} : {sex_counts['女']}" if total_sex else "暂无数据"
    with summary_cols[3]:
        render_ui_stat_card("男女比例", sex_ratio, "男 : 女")

    render_section_title("高危病例", "!")
    render_ui_info_card("高危病例列表", "姓名已脱敏，关键异常特征由阈值规则判读。", "▤")
    high_risk_rows = []
    for record in records:
        if record.get("风险等级") != "高危":
            continue
        high_risk_rows.append({
            "姓名": display_name(record),
            "日期": record.get("时间") or "暂无数据",
            "风险等级": record.get("风险等级") or "暂无数据",
            "关键异常特征": "；".join(abnormal_features(record)) or "暂无数据",
        })
    if high_risk_rows:
        st.dataframe(pd.DataFrame(high_risk_rows), use_container_width=True, hide_index=True)
    else:
        st.info("暂无高危病例。")

    render_section_title("风险分布", "◔")
    summary = pd.Series(risk_counts).reindex(["低危", "中危", "高危"], fill_value=0)
    pie = px.pie(names=summary.index.tolist(), values=summary.values.tolist(), color=summary.index.tolist(), color_discrete_map=RISK_COLORS)
    render_ui_info_card("风险等级分布", "按历史筛查记录统计", "◔")
    st.plotly_chart(plot_layout(pie, 300), use_container_width=True)

    feature_counts = {}
    for record in records:
        for item in abnormal_features(record):
            feature_name = item.split("：", 1)[0]
            feature_counts[feature_name] = feature_counts.get(feature_name, 0) + 1
    render_section_title("异常特征频率", "▥")
    if feature_counts:
        feature_frame = pd.DataFrame(sorted(feature_counts.items(), key=lambda item: item[1], reverse=True), columns=["特征", "异常次数"])
        feature_fig = px.bar(feature_frame, x="特征", y="异常次数", color="异常次数", color_continuous_scale=["#1677ff", "#f5222d"])
        render_ui_info_card("异常特征频率", "轻度异常和显著异常均计入统计", "▥")
        st.plotly_chart(plot_layout(feature_fig, 320), use_container_width=True)
    else:
        st.info("暂无异常特征数据。")

    trend_rows = []
    for record in records:
        raw_date = str(record.get("时间") or "")[:10]
        try:
            date_value = datetime.datetime.strptime(raw_date, "%Y-%m-%d").date().isoformat()
        except ValueError:
            date_value = "暂无日期"
        trend_rows.append({"日期": date_value, "风险等级": record.get("风险等级") or "未知"})
    trend_frame = pd.DataFrame(trend_rows).groupby(["日期", "风险等级"]).size().reset_index(name="筛查次数")
    render_section_title("筛查趋势", "↗")
    if not trend_frame.empty:
        trend_fig = go.Figure()
        for risk_level, group in trend_frame.groupby("风险等级"):
            dated = group[group["日期"] != "暂无日期"].copy()
            dated["日期"] = pd.to_datetime(dated["日期"], errors="coerce")
            dated = dated.dropna(subset=["日期"]).sort_values("日期")
            color = RISK_COLORS.get(risk_level, "#6c757d")
            if len(dated) >= 3:
                origin = dated["日期"].min()
                x = (dated["日期"] - origin).dt.total_seconds().to_numpy() / 86400.0
                y = dated["筛查次数"].to_numpy(dtype=float)
                x_smooth = np.linspace(x.min(), x.max(), 100)
                spline = make_interp_spline(x, y, k=min(3, len(x) - 1))
                y_smooth = spline(x_smooth)
                date_smooth = origin + pd.to_timedelta(x_smooth, unit="D")
                trend_fig.add_trace(go.Scatter(x=date_smooth, y=y_smooth, mode="lines", name=risk_level, line=dict(color=color)))
                trend_fig.add_trace(go.Scatter(x=dated["日期"], y=dated["筛查次数"], mode="markers", name=f"{risk_level}数据", marker=dict(color=color), showlegend=False))
            else:
                trend_fig.add_trace(go.Scatter(x=dated["日期"], y=dated["筛查次数"], mode="lines+markers", name=risk_level, line=dict(color=color)))
        if (trend_frame["日期"] == "暂无日期").any():
            trend_fig.add_annotation(text="含暂无日期记录", xref="paper", x=1, yref="paper", y=1.08, showarrow=False)
        trend_fig.update_layout(xaxis_title="日期", yaxis_title="筛查次数")
        render_ui_info_card("筛查趋势", "按日期和风险等级统计筛查次数", "↗")
        st.plotly_chart(plot_layout(trend_fig, 320), use_container_width=True)
    else:
        st.info("暂无趋势数据。")


# 渲染项目说明页面
def about_page():
    st.title("关于项目")
    render_section_title("系统简介", "♥")
    render_ui_info_card("双通路AI心电筛查", "本系统面向基层医护人员，是一套基于双通路AI的心电风险辅助筛查系统。系统服务于基层医护、校园体检和社区门诊场景，提供AI初筛、结果可解释和医生终审支持。")
    intro_cols = st.columns(3)
    intro_items = [("目标用户", "基层医护、校园体检、社区门诊", ""), ("核心价值", "AI初筛 + 可解释 + 医生终审", ""), ("使用方式", "上传单导联ECG，查看风险和复核依据", "")]
    for col, (title, desc, icon) in zip(intro_cols, intro_items):
        with col:
            render_ui_feature_card(title, desc, icon)

    render_section_title("核心功能", "▦")
    feature_cols = st.columns(3)
    features = [
        ("双通路推理", "1D-CNN异常定位 + XGBoost风险分级", ""),
        ("SHAP可解释", "支持条形图、蜂群图、交互图、决策力图", ""),
        ("智能诊断解读", "GLM大模型生成风险总结和病因分析", ""),
        ("疾病知识库", "RAG检索10类常见心电疾病", ""),
        ("病例教学", "内置典型病例和学习要点", ""),
        ("病情统计", "风险分布、筛查趋势和异常特征频率", ""),
    ]
    for index, (title, description, icon) in enumerate(features):
        with feature_cols[index % 3]:
            render_ui_feature_card(title, description, icon)

    render_section_title("技术架构", "⌘")
    architecture_cols = st.columns(4)
    architecture = [("前端展示层", "Streamlit负责页面交互、图表和报告展示", "▣"), ("业务服务层", "service/负责分析、历史、患者和配置业务", "⚙"), ("算法模型层", "src/负责预处理、特征、CNN、XGBoost和SHAP", "◉"), ("数据存储层", "SQLite保存记录，.npy文件保存波形数据", "▤")]
    for col, (title, desc, icon) in zip(architecture_cols, architecture):
        with col:
            render_ui_info_card(title, desc, icon)

    render_section_title("双通路说明", "⇄")
    path_cols = st.columns(2)
    with path_cols[0]:
        render_ui_info_card("数值通路（XGBoost）", "12项临床特征 → 三级风险分级", "▥", "ecg-blue-card")
    with path_cols[1]:
        render_ui_info_card("形态通路（1D-CNN）", "心拍级分析 → 异常心拍定位", "◉", "ecg-green-card")
    render_ui_info_card("判读原则", "两条通路独立输出结果，不自动融合，最终由医生结合临床信息综合判断。", "!", "ecg-blue-card")

    render_section_title("模型信息", "◈")
    model_cols = st.columns(2)
    with model_cols[0]:
        render_ui_stat_card("XGBoost · 三级风险分级", "0.90", "留出集准确率", "▥")
        render_ui_info_card("训练规模", "训练样本：47条", "#")
    with model_cols[1]:
        render_ui_stat_card("CNN · 正常/异常二分类", "0.98", "测试准确率", "◉")
        render_ui_info_card("模型规模", "参数量：43,521", "#")
    render_ui_warning_card("诚实标注：高危敏感性未验证，原因是留出集无高危样本。")

    render_section_title("数据来源", "⌁")
    render_ui_info_card("训练数据", "MIT-BIH心律失常数据库", "▤")

    render_section_title("版本信息", "ℹ")
    render_ui_stat_card("当前版本", "V2.1.0", "2026年9月", "•")

    render_section_title("免责声明", "!")
    render_ui_warning_card("本系统为辅助筛查工具，不替代执业医师诊断。所有分析结果仅供参考，请由专业医护人员结合完整病史和必要检查进行终审。")


# 执行主程序入口
def main():
    init_state()
    config = normalize_config(load_config())
    st.session_state["theme"] = config.get("theme", st.session_state.get("theme", "医疗蓝"))
    inject_global_css(st.session_state.get("theme", "医疗蓝"))
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
