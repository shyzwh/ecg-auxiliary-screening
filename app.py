import datetime
import html
import json
import os
import uuid

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
import xgboost as xgb

from src.cnn_inference import predict_abnormal_beats
from src.config_utils import DEFAULT_CONFIG, load_config, resolve_config_paths, save_config
from src.data_loader import load_ecg
from src.feature_extract import extract_all_features, pan_tompkins
from src.inference import FEATURE_ORDER, explain_shap_visualizations, explain_with_shap, predict_risk
from src.preprocess import preprocess_ecg
from src.report_gen import generate_pdf_report, generate_report


st.set_page_config(
    page_title="心电风险辅助筛查系统",
    page_icon="assets/favicon.jpg",
    layout="wide"
)

APP_VERSION = "v1.1.0"
RISK_COLORS = {"低危": "#52c41a", "中危": "#faad14", "高危": "#ff4d4f"}
RISK_ICONS = {"低危": "✓", "中危": "!", "高危": "!"}
DEFAULT_UI_CONFIG = {
    **DEFAULT_CONFIG,
    "theme": "医疗蓝",
    "default_sex": "未指定",
    "qtc_threshold_male": 440,
    "qtc_threshold_female": 460,
    "qtc_threshold_default": 450,
    "qtc_threshold_mild_high": 500,
    "llm_enabled": False,
    "llm_provider": "OpenAI 兼容接口",
    "llm_api_key": "",
    "llm_endpoint": "",
    "llm_model": "",
}


def inject_styles(theme):
    """注入页面级样式；使用 Streamlit 容器模拟固定的视觉层级。"""
    if theme == "浅色":
        background = "linear-gradient(135deg,#f8fbff 0%,#eef5ff 100%)"
    elif theme == "深色":
        background = "linear-gradient(135deg,#101a2b 0%,#172b46 100%)"
    else:
        background = "linear-gradient(135deg,#f0f7ff 0%,#e6f0ff 100%)"
    st.markdown(f"""
    <style>
    :root {{ --primary:#1677ff; --cyan:#13c2c2; --ink:#18324f; --muted:#667085; }}
    html,body,[class*="css"] {{ font-family:"Microsoft YaHei","PingFang SC",sans-serif; }}
    .stApp {{ background:{background}; color:var(--ink); }}
    [data-testid="stAppViewContainer"] > .main {{ padding-top:1rem; }}
    [data-testid="stSidebar"] {{ background:linear-gradient(180deg,#1677ff 0%,#0958d9 100%); }}
    [data-testid="stSidebar"] * {{ color:#fff; }}
    [data-testid="stSidebar"] .stRadio label {{ padding:9px 12px; border-radius:8px; transition:.2s; }}
    [data-testid="stSidebar"] .stRadio label:hover {{ background:rgba(255,255,255,.16); }}
    .topbar,.glass-card,.metric-card,.section-card {{ background:rgba(255,255,255,.72); backdrop-filter:blur(12px); -webkit-backdrop-filter:blur(12px); border:1px solid rgba(255,255,255,.72); border-radius:12px; box-shadow:0 4px 20px rgba(0,0,0,.08); animation:rise .45s ease both; }}
    .topbar {{ padding:14px 20px; margin-bottom:20px; display:flex; justify-content:space-between; align-items:center; }}
    .topbar-title {{ font-size:24px; font-weight:800; color:#0b2d7a; }} .topbar-meta {{ color:#667085; font-size:12px; text-align:right; }}
    .section-card {{ padding:20px; margin:18px 0; }} .section-title {{ border-left:4px solid var(--primary); padding-left:10px; font-size:16px; font-weight:800; color:#0b2d7a; margin-bottom:14px; }}
    .metric-card {{ padding:16px; min-height:104px; }} .metric-label {{ color:var(--muted); font-size:13px; }} .metric-value {{ font:800 28px Consolas,monospace; color:#18324f; margin-top:9px; }}
    .risk-hero {{ color:#fff; padding:24px; border-radius:12px; animation:pop .3s ease both; box-shadow:0 8px 24px rgba(0,0,0,.14); }} .risk-hero .risk-icon {{ font-size:30px; font-weight:900; }} .risk-hero .risk-name {{ font-size:32px; font-weight:900; margin-top:6px; }}
    .path-card {{ padding:20px; min-height:220px; border-radius:12px; background:rgba(255,255,255,.78); box-shadow:0 4px 20px rgba(0,0,0,.08); border-top:4px solid var(--primary); }} .path-card.cnn {{ border-color:#1677ff; }} .path-card.xgb {{ border-color:#13c2c2; }}
    .path-label {{ font-size:12px; font-weight:800; color:#1677ff; }} .xgb .path-label {{ color:#08979c; }} .path-title {{ font-size:18px; font-weight:800; margin:8px 0 18px; }}
    .hint {{ color:#667085; font-size:12px; line-height:1.6; }} .report-box {{ background:rgba(248,250,252,.92); border:1px solid #dbe4ef; border-radius:8px; padding:18px; white-space:pre-wrap; font:14px/1.8 Consolas,"Microsoft YaHei",monospace; max-height:520px; overflow:auto; }}
    .footer {{ color:#7b8794; font-size:12px; text-align:center; padding:20px 0 8px; }}
    @keyframes rise {{ from {{ opacity:0; transform:translateY(8px); }} to {{ opacity:1; transform:translateY(0); }} }} @keyframes pop {{ from {{ opacity:0; transform:scale(.95); }} to {{ opacity:1; transform:scale(1); }} }}
    .stButton > button {{ border-radius:8px; padding:8px 20px; font-size:14px; transition:.2s; border:1px solid #1677ff; }} .stButton > button:hover {{ transform:translateY(-2px); box-shadow:0 5px 12px rgba(22,119,255,.2); }}
    </style>
    """, unsafe_allow_html=True)


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
        labels = ["♥  心电分析", "▣  历史记录", "⚙  系统设置", "ⓘ  关于项目"]
        selected = st.radio("功能导航", labels, label_visibility="collapsed")
        st.markdown("---")
        st.caption(f"{APP_VERSION} · 本地运行")
        st.caption("● JSON 存储正常")
    return selected.split("  ", 1)[-1]


def init_state():
    defaults = {"analysis_result": None, "selected_file": None, "history_detail": None}
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def get_history(path):
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)
        return data if isinstance(data, list) else []
    except (OSError, json.JSONDecodeError):
        return []


def save_history(records, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(records, file, ensure_ascii=False, indent=2)


def serialize_value(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: serialize_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [serialize_value(item) for item in value]
    return value


def build_input_metadata(file_path, file_name):
    metadata = {"file_name": file_name, "file_path": os.path.abspath(file_path)}
    try:
        stat = os.stat(file_path)
        metadata.update({"file_size": stat.st_size, "file_modified_at": datetime.datetime.fromtimestamp(stat.st_mtime).isoformat()})
    except OSError:
        metadata["file_size"] = None
    return metadata


def save_complete_record(result, config, patient_info=None):
    record_id = result.get("record_id") or uuid.uuid4().hex
    waveform_dir = config.get("waveform_dir", os.path.join(os.path.dirname(config["storage_path"]), "waveforms"))
    os.makedirs(waveform_dir, exist_ok=True)
    waveform_path = os.path.join(waveform_dir, f"{record_id}.npy")
    np.save(waveform_path, np.asarray(result["signal"], dtype=float))
    relative_waveform_path = os.path.relpath(waveform_path, os.path.dirname(config["storage_path"]))
    record = {key: serialize_value(value) for key, value in result.items() if key != "signal"}
    record.update({
        "record_id": record_id,
        "patient_info": serialize_value(patient_info or result.get("patient_info", {})),
        "input_metadata": serialize_value(result.get("input_metadata", {})),
        "waveform_path": relative_waveform_path,
        "config_snapshot": serialize_value(result.get("config_snapshot", config)),
    })
    history = get_history(config["storage_path"])
    history = [item for item in history if item.get("record_id") != record_id]
    history.append(record)
    save_history(history, config["storage_path"])
    return record_id


def restore_complete_record(record, config):
    restored = dict(record)
    waveform_path = record.get("waveform_path")
    if not waveform_path:
        return restored
    storage_dir = os.path.dirname(config["storage_path"])
    path = waveform_path if os.path.isabs(waveform_path) else os.path.join(storage_dir, waveform_path)
    try:
        restored["signal"] = np.load(path)
    except (OSError, ValueError):
        restored["signal"] = np.array([])
    return restored


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


def run_analysis(file_path, file_name, config, sex=None, patient_info=None):
    progress = st.progress(0, text="准备开始分析")
    labels = ["上传中", "预处理中", "特征提取中", "CNN 推理中", "XGBoost 推理中", "SHAP 分析中", "生成报告中"]
    def update(index):
        progress.progress(index / 7, text=f"✓ {labels[index - 1]} · 正在处理下一步" if index else "准备开始分析")
    update(1)
    status, signal, fs, message = load_ecg(file_path)
    if status != "success": return None, message
    update(2)
    status, clean_signal, message = preprocess_ecg(signal, fs)
    if status != "success": return None, message
    update(3)
    status, r_peaks, message = pan_tompkins(clean_signal, fs)
    if status != "success": return None, message
    status, features, message = extract_all_features(clean_signal, r_peaks, fs)
    if status != "success": return None, message
    features = {key: 0.0 if value is None else value for key, value in features.items()}
    update(4)
    cnn_status, abnormal_positions, cnn_confidence, cnn_message = predict_abnormal_beats(clean_signal, r_peaks, fs, config["cnn_model_path"])
    abnormal_positions = abnormal_positions or []
    update(5)
    analysis_sex = sex or config.get("default_sex", "未指定")
    xgb_status, risk_level, risk_num, score, risk_message = predict_risk(features, config["model_path"], config, analysis_sex)
    if xgb_status != "success": return None, risk_message
    risk_probs = risk_probabilities(features, config, risk_num, score)
    update(6)
    shap_status, shap_result, shap_message = explain_with_shap(features, config["model_path"])
    shap_visual_status, shap_visualizations, shap_visual_message = explain_shap_visualizations(features, config["model_path"], os.path.join(os.path.dirname(config["model_path"]), "..", "training_data.csv"))
    update(7)
    total_beats = len(r_peaks)
    abnormal_count = len(abnormal_positions)
    cnn_state = "abnormal" if cnn_status == "success" and abnormal_count else "normal"
    xgb_confidence = float(score)
    confidence_threshold = float(config.get("low_confidence_threshold", 0.8))
    if cnn_confidence < confidence_threshold or xgb_confidence < confidence_threshold:
        review_status = "低置信度"
    elif cnn_state == "abnormal" and risk_level in ("高危", "中危"):
        review_status = "双通路一致-高风险"
    elif cnn_state == "abnormal" and risk_level == "低危":
        review_status = "通路冲突-波形异常"
    elif cnn_state == "normal" and risk_level == "高危":
        review_status = "通路冲突-风险偏高"
    else:
        review_status = "双通路一致-低风险"
    report_status, report_text, report_data = generate_report(risk_num=risk_num, risk_score=score, risk_probs=risk_probs, features=features, abnormal_count=abnormal_count, total_beats=total_beats, config=config, sex=analysis_sex, cnn_status=cnn_state, review_status=review_status, cnn_confidence=cnn_confidence, xgb_confidence=xgb_confidence)
    progress.empty()
    return {"record_id": uuid.uuid4().hex, "patient_info": patient_info or {}, "input_metadata": build_input_metadata(file_path, file_name), "file_name": file_name, "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "signal": clean_signal, "fs": fs, "r_peaks": r_peaks, "features": features, "abnormal_positions": abnormal_positions, "cnn_status": cnn_status, "cnn_state": cnn_state, "cnn_confidence": cnn_confidence, "cnn_message": cnn_message, "xgb_status": xgb_status, "risk_level": risk_level, "risk_num": risk_num, "score": score, "xgb_confidence": xgb_confidence, "risk_probs": risk_probs, "review_status": review_status, "shap_status": shap_status, "shap_result": shap_result, "shap_message": shap_message, "shap_visual_status": shap_visual_status, "shap_visualizations": shap_visualizations, "shap_visual_message": shap_visual_message, "report_status": report_status, "report_text": report_text, "report_data": report_data, "config_snapshot": serialize_value(config)}, "分析完成"


def plot_layout(fig, height=360):
    fig.update_layout(height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=20, r=20, t=45, b=25), font=dict(family="Microsoft YaHei"), hoverlabel=dict(bgcolor="white", font_size=13))
    fig.update_xaxes(showgrid=True, gridcolor="rgba(120,140,160,.22)", griddash="dot")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(120,140,160,.22)", griddash="dot")
    return fig


def render_waveform(result, config):
    st.markdown('<div class="section-card"><div class="section-title">心电波形与异常定位</div>', unsafe_allow_html=True)
    signal, fs = np.asarray(result["signal"]), result["fs"]
    indices = np.arange(0, len(signal), max(1, len(signal) // 12000))
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=indices / fs, y=signal[indices], mode="lines", name="清洗信号", line=dict(color="#1677ff", width=1.5)))
    peaks = result["r_peaks"]
    if config.get("show_r_peaks", True): figure.add_trace(go.Scatter(x=[peak / fs for peak in peaks], y=[signal[peak] for peak in peaks], mode="markers", name="R峰", marker=dict(color="#ff4d4f", size=6)))
    abnormal = result["abnormal_positions"]
    if abnormal: figure.add_trace(go.Scatter(x=[peak / fs for peak in abnormal], y=[signal[peak] for peak in abnormal], mode="markers", name="异常心拍", marker=dict(color="#fa8c16", size=10, symbol="x")))
    figure.update_layout(title="单导联 ECG", xaxis_title="时间（秒）", yaxis_title="电压", hovermode="x unified")
    st.plotly_chart(plot_layout(figure, 420), use_container_width=True, key="waveform")
    st.markdown("</div>", unsafe_allow_html=True)


def render_results(result, config):
    risk, color = result["risk_level"], RISK_COLORS.get(result["risk_level"], "#8c8c8c")
    st.markdown('<div class="section-card"><div class="section-title">核心筛查结论</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="risk-hero" style="background:linear-gradient(135deg,{color},#18324f)"><div class="risk-icon">{RISK_ICONS.get(risk,"!")} 风险结论</div><div class="risk-name">{risk}</div><div>模型置信评分 {result["score"]:.2f}</div></div>', unsafe_allow_html=True)
    columns = st.columns(4)
    metrics = [("心率", f'{result["features"].get("HR", 0):.1f}', "次/分"), ("总心拍", str(len(result["r_peaks"])), "个"), ("异常心拍", str(len(result["abnormal_positions"])), "个"), ("采样率", f'{result["fs"]:.0f}', "Hz")]
    for column, (label, value, unit) in zip(columns, metrics):
        with column: st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="hint">{unit}</div></div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-card"><div class="section-title">双通路分析结果</div>', unsafe_allow_html=True)
    cnn_title = "已完成" if result["cnn_status"] == "success" else "不可用"
    cnn_hint = f'异常心拍 {len(result["abnormal_positions"])} 个' if result["cnn_status"] == "success" else result["cnn_message"]
    path_cnn, path_xgb = st.columns(2)
    with path_cnn:
        st.markdown(f'<div class="path-card cnn"><div class="path-label">形态通路 · 1D-CNN</div><div class="path-title">{cnn_title}</div><div class="metric-value">{len(result["abnormal_positions"])} <span style="font:14px Microsoft YaHei">异常心拍</span></div><div class="hint">置信度：{result.get("cnn_confidence", 0.0):.1%}<br>{html.escape(cnn_hint)}</div></div>', unsafe_allow_html=True)
    with path_xgb:
        st.markdown(f'<div class="path-card xgb"><div class="path-label">数值通路 · XGBoost</div><div class="path-title">风险分级：{risk}</div><div class="metric-value">{result["score"]:.2f} <span style="font:14px Microsoft YaHei">风险评分</span></div><div class="hint">置信度：{result.get("xgb_confidence", result["score"]):.1%}<br>模型概率：低危 {result["risk_probs"][0]:.1%} · 中危 {result["risk_probs"][1]:.1%} · 高危 {result["risk_probs"][2]:.1%}</div></div>', unsafe_allow_html=True)
    st.info("两条通路独立输出，请医护人员结合波形、特征和临床信息综合判断。")
    review_status = result.get("review_status", "未生成")
    if review_status in ("低置信度", "通路冲突-波形异常", "通路冲突-风险偏高"):
        st.warning(f"复核状态：{review_status}")
    else:
        st.success(f"复核状态：{review_status}")
    pie = px.pie(names=["低危", "中危", "高危"], values=result["risk_probs"], color=["低危", "中危", "高危"], color_discrete_map=RISK_COLORS, hole=.48)
    pie.update_traces(textinfo="percent+label")
    st.plotly_chart(plot_layout(pie, 300), use_container_width=True, key="risk_pie")
    st.markdown("</div>", unsafe_allow_html=True)
    render_waveform(result, config)
    st.markdown('<div class="section-card"><div class="section-title">12 项临床特征</div>', unsafe_allow_html=True)
    st.dataframe(pd.DataFrame([result["features"]]).T.rename(columns={0: "数值"}), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
    if config.get("show_shap", True):
        st.markdown('<div class="section-card"><div class="section-title">SHAP 可解释性分析</div><div class="hint">SHAP 仅解释 XGBoost 的 12 项数值特征。</div>', unsafe_allow_html=True)
        if result["shap_status"] == "success":
            render_shap_result(result["shap_result"], result.get("shap_visualizations"))
        else: st.warning(f'SHAP 暂不可用：{result["shap_message"]}')
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown('<div class="section-card"><div class="section-title">筛查报告</div>', unsafe_allow_html=True)
    if result["report_status"] == "success":
        st.markdown(f'<div class="report-box">{html.escape(result["report_text"])}</div>', unsafe_allow_html=True)
        d1, d2, d3 = st.columns(3)
        with d1: st.download_button("下载 TXT", result["report_text"], "心电筛查报告.txt", "text/plain", use_container_width=True)
        with d2:
            pdf_status, pdf_content, pdf_message = generate_pdf_report(result)
            if pdf_status == "success":
                st.download_button("下载 PDF", pdf_content, "心电筛查报告.pdf", "application/pdf", use_container_width=True)
            else:
                st.warning(pdf_message)
        with d3:
            if st.button("复制到剪贴板", use_container_width=True): components.html(f"<script>navigator.clipboard.writeText({json.dumps(result['report_text'])});</script><p style='font:12px sans-serif'>已请求复制，请检查浏览器提示。</p>", height=35)
        if st.button("打印友好视图", use_container_width=True): components.html(f"<script>const w=window.open('','_blank');w.document.write('<pre style=\"white-space:pre-wrap;font:14px/1.8 Consolas\">'+{json.dumps(html.escape(result['report_text']))}+'</pre>');w.document.close();w.print();</script><p style='font:12px sans-serif'>已打开打印窗口（若被拦截请允许弹窗）。</p>", height=35)
    else: st.error(result["report_text"])
    st.markdown("</div>", unsafe_allow_html=True)


def render_shap_result(shap_result, visualizations=None):
    if not shap_result:
        return
    shap_df = pd.DataFrame([{"特征": name, "贡献": info["shap_value"]} for name, info in shap_result.items()]).sort_values("贡献")
    shap_fig = px.bar(shap_df, x="贡献", y="特征", orientation="h", color="贡献", color_continuous_scale=["#1677ff", "#f5222d"])
    st.plotly_chart(plot_layout(shap_fig, 440), use_container_width=True)
    if not visualizations:
        st.info("当前记录缺少详细 SHAP 数据，暂无法生成全局、交互和决策图。")
        return
    target_label = visualizations.get("target_label", "未知类别")
    st.caption(f"SHAP 解释目标类别：{target_label}（类别 {visualizations.get('target_class', '未知')}）")
    global_data = visualizations.get("global")
    if global_data:
        global_frame = pd.DataFrame(global_data["shap_values"], columns=FEATURE_ORDER)
        importance = global_frame.abs().mean().sort_values()
        global_fig = px.bar(x=importance.values, y=importance.index, orientation="h", labels={"x": "平均绝对 SHAP 值", "y": "特征"}, title=f"全局特征重要性（{global_data['sample_count']} 个样本）")
        st.plotly_chart(plot_layout(global_fig, 420), use_container_width=True)
    else:
        st.info("全局特征重要性图暂不可用：训练样本不足或数据不可读。")
    interaction = visualizations.get("interaction")
    if interaction:
        interaction_fig = px.scatter(x=interaction["x"], y=interaction["y"], color=interaction["color"], labels={"x": "标准化 QRS", "y": "QRS 的 SHAP 值", "color": "标准化 ST_shift"}, title="QRS 与 ST_shift 交互分析")
        st.plotly_chart(plot_layout(interaction_fig, 420), use_container_width=True)
    else:
        st.info("QRS 与 ST_shift 交互图暂不可用：缺少足够样本。")
    decision = visualizations.get("decision")
    if decision:
        base_value = float(visualizations.get("base_value", 0.0))
        cumulative = np.cumsum([base_value] + decision["contributions"])
        decision_fig = go.Figure(go.Scatter(x=list(range(len(cumulative))), y=cumulative, mode="lines+markers", text=["base"] + decision["features"], hovertemplate="%{text}<br>模型输出：%{y:.4f}<extra></extra>"))
        decision_fig.update_layout(title="单样本 SHAP 决策路径", xaxis_title="贡献步骤", yaxis_title="模型输出")
        st.plotly_chart(plot_layout(decision_fig, 420), use_container_width=True)


def analysis_page(config):
    st.title("心电分析")
    st.caption("上传单导联 ECG，完成预处理、双通路分析与可解释报告生成。")
    result = st.session_state.get("analysis_result")
    if result:
        a, b = st.columns(2)
        with a:
            if st.button("重新分析", type="primary", use_container_width=True): st.session_state["analysis_result"] = None; st.rerun()
        with b:
            if st.button("清空结果", use_container_width=True): st.session_state["analysis_result"] = None; st.session_state["selected_file"] = None; st.rerun()
        render_results(result, config); render_save_record(result, config); return
    st.markdown('<div class="section-card"><div class="section-title">上传心电数据</div>', unsafe_allow_html=True)
    patient_name = st.text_input("患者姓名", placeholder="可填写脱敏编号")
    patient_age = st.number_input("患者年龄", min_value=0, max_value=150, value=0, step=1)
    sex_options = ["未指定", "男", "女"]
    default_sex = config.get("default_sex", "未指定")
    sex = st.selectbox("本次分析性别", sex_options, index=sex_options.index(default_sex) if default_sex in sex_options else 0)
    upload, demo = st.columns([3, 1])
    with upload: uploaded = st.file_uploader("选择 CSV / TXT / DAT 文件", type=["csv", "txt", "dat"], label_visibility="collapsed")
    with demo: demo_clicked = st.button("加载示例数据", use_container_width=True)
    file_path, file_name = None, None
    if demo_clicked:
        file_path = os.path.join(config["data_dir"], "100.dat")
        file_name = "MIT-BIH 100.dat（示例）"
    elif uploaded:
        os.makedirs(config["upload_dir"], exist_ok=True); file_name = os.path.basename(uploaded.name); file_path = os.path.join(config["upload_dir"], file_name)
        with open(file_path, "wb") as file: file.write(uploaded.getbuffer())
    if file_path: st.session_state["selected_file"] = (file_path, file_name)
    st.markdown("</div>", unsafe_allow_html=True)
    with st.expander("高级设置", expanded=False): st.caption("参数保存到配置文件；现有 src 预处理接口保持不变。")
    selected = st.session_state.get("selected_file")
    if not selected:
        st.markdown('<div class="glass-card" style="padding:52px;text-align:center;margin-top:18px"><div style="font-size:46px">♥</div><h3>开始一次清晰的心电筛查</h3><p class="hint">上传心电文件或点击“加载示例数据”开始分析</p></div>', unsafe_allow_html=True); return
    st.success(f"已选择：{selected[1]}")
    if st.button("开始分析", type="primary", use_container_width=True):
        patient_info = {"name": patient_name, "age": int(patient_age) if patient_age else None, "sex": sex}
        result, message = run_analysis(selected[0], selected[1], config, sex, patient_info)
        if result: st.session_state["analysis_result"] = result; st.rerun()
        else: st.error(message)


def render_save_record(result, config):
    st.markdown('<div class="section-card"><div class="section-title">保存分析记录</div>', unsafe_allow_html=True)
    note = st.text_input("备注（可选）", key="record_note", placeholder="患者编号或检查日期")
    if st.button("保存到历史记录", type="primary"):
        result["备注"] = note
        record_id = save_complete_record(result, config, result.get("patient_info", {}))
        st.session_state["analysis_result"] = result
        st.success(f"已保存到历史记录：{record_id}")
    st.markdown("</div>", unsafe_allow_html=True)


def history_page(config):
    st.title("历史记录")
    path, records = config.get("storage_path", "storage/records.json"), get_history(config.get("storage_path", "storage/records.json"))
    if not records: st.info("暂无历史记录。完成一次分析后，可在分析页保存记录。"); return
    frame = pd.DataFrame(records)
    for column in ["风险等级", "文件名", "时间"]:
        if column not in frame: frame[column] = ""
    c1, c2, c3, c4 = st.columns(4); c1.metric("总记录", len(frame)); c2.metric("低危", int((frame["风险等级"] == "低危").sum())); c3.metric("中危", int((frame["风险等级"] == "中危").sum())); c4.metric("高危", int((frame["风险等级"] == "高危").sum()))
    f1, f2 = st.columns(2); risk_filter = f1.selectbox("风险等级", ["全部", "低危", "中危", "高危"]); keyword = f2.text_input("搜索文件名", placeholder="输入关键词")
    filtered = frame.copy()
    if risk_filter != "全部": filtered = filtered[filtered["风险等级"] == risk_filter]
    if keyword: filtered = filtered[filtered["文件名"].astype(str).str.contains(keyword, regex=False, na=False)]
    filtered = filtered.iloc[::-1]
    visible = ["时间", "文件名", "风险等级", "风险评分", "总心拍数", "异常心拍数", "备注"]
    for column in visible:
        if column not in filtered: filtered[column] = ""
    st.dataframe(filtered[visible], use_container_width=True, hide_index=True)
    st.download_button("导出筛选记录 CSV", filtered.to_csv(index=False).encode("utf-8-sig"), "心电筛查历史记录.csv", "text/csv")
    if len(filtered):
        options = list(filtered.index); selected_index = st.selectbox("选择记录查看详情", options, format_func=lambda index: f'{filtered.loc[index, "时间"]} · {filtered.loc[index, "文件名"]}'); selected = restore_complete_record(records[selected_index], config)
        st.markdown('<div class="section-card"><div class="section-title">记录详情</div>', unsafe_allow_html=True)
        risk_level = selected.get("risk_level", selected.get("风险等级", "未知"))
        risk_score = selected.get("score", selected.get("风险评分", 0))
        abnormal_count = len(selected.get("abnormal_positions", [])) or selected.get("异常心拍数", 0)
        d1, d2, d3 = st.columns(3); d1.metric("风险等级", risk_level); d2.metric("风险评分", f'{float(risk_score):.2f}'); d3.metric("异常心拍", abnormal_count)
        st.write(f'文件：{selected.get("file_name", selected.get("文件名", ""))} · 时间：{selected.get("created_at", selected.get("时间", ""))}')
        patient_info = selected.get("patient_info", {})
        if patient_info:
            st.write(f'患者：{patient_info.get("name", "")} · 年龄：{patient_info.get("age", "")} · 性别：{patient_info.get("sex", "未指定")}')
        if selected.get("input_metadata"):
            st.caption(f'输入文件：{selected["input_metadata"].get("file_name", selected.get("文件名", ""))} · 大小：{selected["input_metadata"].get("file_size", "未知")} 字节')
        if selected.get("signal") is not None and len(selected.get("signal", [])):
            render_waveform(selected, config)
        features = selected.get("features", selected.get("特征", {}))
        if features: st.dataframe(pd.DataFrame([features]).T.rename(columns={0: "数值"}), use_container_width=True)
        if selected.get("shap_status") == "success":
            st.subheader("SHAP 结果")
            render_shap_result(selected.get("shap_result", {}), selected.get("shap_visualizations"))
        report = selected.get("report_text", selected.get("报告", "")) or f'风险等级：{risk_level}\n风险评分：{risk_score}\n完整图表需重新分析。'
        st.markdown(f'<div class="report-box">{html.escape(report)}</div>', unsafe_allow_html=True)
        st.download_button("导出该条报告 TXT", report, "心电筛查报告.txt", "text/plain")
        pdf_status, pdf_content, pdf_message = generate_pdf_report(selected)
        if pdf_status == "success":
            st.download_button("导出该条报告 PDF", pdf_content, "心电筛查报告.pdf", "application/pdf")
        else:
            st.warning(pdf_message)
        st.caption("历史记录已保存完整分析详情；波形文件按记录 ID 独立存储。")
        st.markdown("</div>", unsafe_allow_html=True)
        if st.button("删除当前记录"):
            record_id = selected.get("record_id")
            records = [item for item in records if item.get("record_id") != record_id] if record_id else records[:selected_index] + records[selected_index + 1:]
            save_history(records, path); st.rerun()


def settings_page(config):
    st.title("系统设置")
    with st.form("settings_form"):
        st.markdown("### 风险与显示"); c1, c2, c3 = st.columns(3); medium = c1.number_input("中危阈值", 0.0, 1.0, float(config["risk_threshold_medium"]), .05); high = c2.number_input("高危阈值", 0.0, 1.0, float(config["risk_threshold_high"]), .05); theme = c3.selectbox("主题", ["医疗蓝", "浅色", "深色"], index=["医疗蓝", "浅色", "深色"].index(config.get("theme", "医疗蓝")))
        st.markdown("### 模型与预处理"); model_path = st.text_input("XGBoost 模型路径", config["model_path"]); cnn_path = st.text_input("CNN 模型路径", config["cnn_model_path"]); band_low, band_high, notch = st.columns(3); low = band_low.number_input("带通下限（Hz）", value=float(config.get("bandpass_low", .5))); upper = band_high.number_input("带通上限（Hz）", value=float(config.get("bandpass_high", 40.0))); notch_freq = notch.selectbox("工频频率", ["auto", "50", "60"], index=["auto", "50", "60"].index(str(config.get("notch_freq", "auto")))); show_shap = st.checkbox("显示 SHAP 图", config.get("show_shap", True))
        st.markdown("### QTc 性别阈值"); q1, q2, q3 = st.columns(3); qtc_male = q1.number_input("男性正常上限（ms）", value=int(config.get("qtc_threshold_male", 440))); qtc_female = q2.number_input("女性正常上限（ms）", value=int(config.get("qtc_threshold_female", 460))); default_sex = q3.selectbox("默认性别", ["未指定", "男", "女"], index=["未指定", "男", "女"].index(config.get("default_sex", "未指定"))); qtc_default = st.number_input("未指定性别正常上限（ms）", value=int(config.get("qtc_threshold_default", 450))); qtc_mild_high = st.number_input("QTc 显著异常起点（ms）", value=int(config.get("qtc_threshold_mild_high", 500)))
        st.markdown("### 可选大模型接口（默认关闭）"); llm_enabled = st.checkbox("启用大模型配置", config.get("llm_enabled", False)); llm_provider = st.selectbox("服务商", ["OpenAI 兼容接口", "自定义服务"]); llm_key = st.text_input("API 密钥", config.get("llm_api_key", ""), type="password"); endpoint = st.text_input("API 端点", config.get("llm_endpoint", "")); llm_model = st.text_input("模型名称", config.get("llm_model", "")); submitted = st.form_submit_button("保存设置", type="primary")
    if submitted:
        if medium >= high: st.error("中危阈值必须小于高危阈值")
        else:
            config.update({"risk_threshold_medium": medium, "risk_threshold_high": high, "theme": theme, "model_path": model_path, "cnn_model_path": cnn_path, "bandpass_low": low, "bandpass_high": upper, "notch_freq": notch_freq, "show_shap": show_shap, "qtc_threshold_male": qtc_male, "qtc_threshold_female": qtc_female, "qtc_threshold_default": qtc_default, "qtc_threshold_mild_high": qtc_mild_high, "default_sex": default_sex, "llm_enabled": llm_enabled, "llm_provider": llm_provider, "llm_api_key": llm_key, "llm_endpoint": endpoint, "llm_model": llm_model}); status, message = save_config(config); st.success(message) if status == "success" else st.error(message); st.rerun()
    b1, b2 = st.columns(2)
    with b1:
        if st.button("测试连接"): st.info("已完成本地配置检查，分析流程不会调用大模型 API。")
    with b2:
        if st.button("恢复默认设置"): save_config(DEFAULT_UI_CONFIG.copy()); st.success("已恢复默认设置，请重新加载页面。")


def about_page():
    st.title("关于项目")
    st.markdown('<div class="section-card"><div class="section-title">系统定位</div><p>本系统面向基层医护人员，用于单导联心电数据的风险辅助筛查与结果解释，不替代执业医师诊断。</p><p class="hint">数据处理、模型推理和报告均在本地完成，历史记录使用 JSON 文件保存。</p></div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1: st.markdown('<div class="path-card cnn"><div class="path-label">形态通路</div><div class="path-title">1D-CNN</div><p>按心拍切分波形，定位可能异常的心拍位置。</p></div>', unsafe_allow_html=True)
    with c2: st.markdown('<div class="path-card xgb"><div class="path-label">数值通路</div><div class="path-title">XGBoost + SHAP</div><p>基于 12 项 ECG/HRV 特征完成风险分级，并解释特征贡献。</p></div>', unsafe_allow_html=True)
    st.markdown("### 技术栈"); st.write("Streamlit · Plotly · NumPy · SciPy · WFDB · TensorFlow/Keras · XGBoost · SHAP"); st.warning("本系统仅供辅助筛查参考，不能替代执业医师诊断；异常结果请结合临床信息及时就医。")


def main():
    init_state(); config = normalize_config(resolve_config_paths(load_config())); page = render_sidebar(); inject_styles(config.get("theme", "医疗蓝")); render_header(page, config)
    if page == "心电分析": analysis_page(config)
    elif page == "历史记录": history_page(config)
    elif page == "系统设置": settings_page(config)
    else: about_page()
    st.markdown('<div class="footer">辅助筛查工具 · 请由专业医护人员结合临床信息判断 · 本地 JSON 存储</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()