"""Reusable UI helpers for the medical ECG dashboard."""

import streamlit as st


# 注入全局样式，包含毛玻璃卡片、医疗蓝主题、悬停效果
def inject_global_css():
    st.markdown(
        """
        <style>
        /* 定义医疗蓝色系与通用主题颜色 */
        :root {
            --medical-blue: #1677ff;
            --medical-deep: #0b2d7a;
            --medical-soft: #edf5ff;
            --glass-bg: rgba(255,255,255,0.72);
            --glass-border: rgba(255,255,255,0.9);
            --text: #18324f;
            --muted: #667085;
            --success: #2ecc71;
            --warning: #f39c12;
            --danger: #e74c3c;
            --shadow: 0 12px 28px rgba(23, 77, 155, 0.12);
        }
        html, body, [data-testid="stAppViewContainer"] {
            background: linear-gradient(135deg, #edf6ff 0%, #eaf3ff 48%, #f6fbff 100%);
            color: var(--text);
            font-family: "Microsoft YaHei", "PingFang SC", sans-serif;
        }
        [data-testid="stAppViewContainer"] > .main { background: transparent; }
        [data-testid="stSidebar"] { background: linear-gradient(180deg, #0d63d6 0%, #0a4aad 100%); }
        [data-testid="stSidebar"] * { color: white; }
        /* 定义毛玻璃卡片样式 */
        .glass-panel {
            background: var(--glass-bg);
            backdrop-filter: blur(18px);
            -webkit-backdrop-filter: blur(18px);
            border: 1px solid var(--glass-border);
            border-radius: 18px;
            box-shadow: var(--shadow);
            padding: 1rem 1.1rem;
            margin-bottom: 1rem;
            min-height: 0;
            transition: transform 0.25s ease, box-shadow 0.25s ease;
        }
        .glass-panel:hover {
            transform: translateY(-2px);
            box-shadow: 0 18px 35px rgba(15, 90, 170, 0.16);
        }
        /* 定义分节标题样式 */
        .section-title {
            color: var(--medical-deep);
            font-size: 1.08rem;
            font-weight: 800;
            padding-left: 0.7rem;
            border-left: 4px solid var(--medical-blue);
            margin-bottom: 0.8rem;
        }
        .risk-badge {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            padding: 0.38rem 0.8rem;
            border-radius: 999px;
            color: white;
            font-weight: 700;
            font-size: 0.8rem;
            min-width: 92px;
        }
        .metric-card {
            background: rgba(255,255,255,0.85);
            border: 1px solid rgba(20, 72, 133, 0.08);
            border-radius: 16px;
            padding: 0.9rem 0.8rem;
            min-height: 110px;
            box-shadow: 0 10px 20px rgba(20, 72, 133, 0.06);
            transition: transform 0.2s ease;
        }
        .metric-card:hover { transform: translateY(-3px); }
        .metric-label {
            font-size: 0.72rem;
            color: var(--muted);
            text-transform: uppercase;
            letter-spacing: 0.04em;
            font-weight: 700;
        }
        .metric-value {
            margin-top: 0.45rem;
            font-size: 1.7rem;
            font-weight: 900;
            color: var(--medical-deep);
            font-family: "Consolas", "Courier New", monospace;
        }
        .metric-subtext {
            margin-top: 0.2rem;
            font-size: 0.74rem;
            color: var(--muted);
        }
        .stButton > button {
            border-radius: 12px;
            font-weight: 700;
            transition: all 0.2s ease;
        }
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(15,111,255,0.16);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# 渲染患者信息卡片
def render_patient_card(patient_name="未命名患者", age="未录入", sex="未指定"):
    st.markdown(
        f"""
        <div class="glass-panel" style="min-height:0;">
            <h3>👤 患者信息</h3>
            <div>姓名：{patient_name}</div>
            <div>年龄：{age}</div>
            <div>性别：{sex}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# 渲染上传卡片
def render_upload_card():
    st.markdown(
        """
        <div class="glass-panel" style="min-height:0;">
            <h3>⤴️ 采集数据</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )
    uploaded = st.file_uploader("选择 ECG 文件", type=["csv", "txt", "dat"], label_visibility="collapsed")
    demo = st.button("加载示例数据", use_container_width=True)
    return uploaded, demo


# 渲染风险等级徽章
def render_risk_badge(level="低危", value=None):
    level_map = {
        "低危": ("#2ecc71", "低危"),
        "中危": ("#f39c12", "中危"),
        "高危": ("#e74c3c", "高危"),
        "未知": ("#6c757d", "未知"),
    }
    color, text = level_map.get(level, level_map["未知"])
    if value is not None:
        text = f"{text} · {value:.2f}"
    st.markdown(f'<span class="risk-badge" style="background:linear-gradient(135deg,{color},#183b63);">{text}</span>', unsafe_allow_html=True)


# 渲染指标卡片
def render_metric_card(label="指标", value="--", unit=""):
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-subtext">{unit}</div></div>',
        unsafe_allow_html=True,
    )


# 渲染分节标题
def render_section_title(text="分节标题"):
    st.markdown(f'<div class="section-title">{text}</div>', unsafe_allow_html=True)
