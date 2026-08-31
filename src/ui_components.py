"""Reusable UI styling helpers for the medical dashboard.

This module centralizes the CSS and small presentation components used by
multiple pages.
"""

import streamlit as st


# 注入全局样式，包含医疗蓝主题、毛玻璃容器和通用组件样式
def inject_global_css():
    """注入全局 CSS，提升界面视觉层级和医疗风格。"""
    st.markdown(
        """
        <style>
        :root {
            --medical-blue: #0f6fff;
            --medical-deep: #0a2f6b;
            --medical-soft: #edf5ff;
            --glass-bg: rgba(255,255,255,0.7);
            --glass-border: rgba(255,255,255,0.9);
            --text-strong: #183b63;
            --text-muted: #6b7d99;
            --success: #2ecc71;
            --warning: #f39c12;
            --danger: #e74c3c;
            --shadow-soft: 0 12px 28px rgba(23, 77, 155, 0.12);
        }
        html, body, [data-testid="stAppViewContainer"] {
            background: linear-gradient(135deg, #edf6ff 0%, #eaf3ff 48%, #f6fbff 100%);
            color: var(--text-strong);
            font-family: "Microsoft YaHei", "Segoe UI", sans-serif;
        }
        [data-testid="stAppViewContainer"] > .main { background: transparent; }
        [data-testid="stSidebar"] { background: linear-gradient(180deg, #0d63d6 0%, #0a4aad 100%); }
        [data-testid="stSidebar"] * { color: white; }
        .stApp [data-testid="stHeader"] { background: rgba(255,255,255,0.0); }
        .stApp footer { visibility: hidden; }
        .stApp .css-1d391kg { display: none; }
        .glass-panel {
            background: var(--glass-bg);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border: 1px solid var(--glass-border);
            border-radius: 20px;
            box-shadow: var(--shadow-soft);
            padding: 1.1rem 1.2rem;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        .glass-panel:hover {
            transform: translateY(-5px);
            box-shadow: 0 16px 32px rgba(24, 59, 99, 0.18);
        }
        .section-title {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            font-size: 1.05rem;
            font-weight: 800;
            color: var(--medical-deep);
            margin: 0 0 0.8rem 0;
            padding-left: 0.7rem;
            border-left: 4px solid var(--medical-blue);
        }
        .risk-badge {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-width: 94px;
            padding: 0.42rem 0.85rem;
            border-radius: 999px;
            font-weight: 700;
            font-size: 0.82rem;
            color: white;
            box-shadow: 0 10px 20px rgba(23, 77, 155, 0.15);
        }
        .metric-card {
            background: rgba(255,255,255,0.84);
            border: 1px solid rgba(25, 64, 125, 0.08);
            border-radius: 18px;
            min-height: 110px;
            box-shadow: 0 12px 24px rgba(24, 59, 99, 0.08);
            transition: transform 0.2s ease;
            padding: 1rem 0.9rem;
        }
        .metric-card:hover { transform: translateY(-4px); }
        .metric-label {
            color: var(--text-muted);
            font-size: 0.72rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }
        .metric-value {
            margin-top: 0.5rem;
            font-size: 1.7rem;
            font-weight: 900;
            color: var(--medical-deep);
            font-family: "Consolas", "Courier New", monospace;
        }
        .metric-subtext { margin-top: 0.30rem; color: var(--text-muted); font-size: 0.74rem; }
        .stButton > button { border-radius: 12px; font-weight: 700; transition: all 0.2s ease; }
        .stButton > button:hover { transform: translateY(-2px); box-shadow: 0 10px 20px rgba(15,111,255,0.18); }
        </style>
        """,
        unsafe_allow_html=True,
    )


# 渲染患者信息卡片，展示姓名、年龄和性别
def render_patient_card(patient_name="未命名患者", age="未录入", sex="未指定"):
    """渲染患者基本信息卡片。"""
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown("### 👤 患者信息")
    st.write(f"姓名：{patient_name}")
    st.write(f"年龄：{age}")
    st.write(f"性别：{sex}")
    st.markdown("</div>", unsafe_allow_html=True)


# 渲染上传卡片，统一管理 ECG 文件和示例数据入口
def render_upload_card():
    """渲染上传卡片。"""
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown("### ⤴️ 采集数据")
    uploaded = st.file_uploader("选择 ECG 文件", type=["csv", "txt", "dat"], label_visibility="collapsed")
    demo = st.button("加载示例数据", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
    return uploaded, demo


# 渲染风险标签，按低中高危使用统一颜色和文案
def render_risk_badge(level="低危", value=None):
    """按风险等级渲染徽章组件。"""
    level_map = {"低危": ("#2ecc71", "低危"), "中危": ("#f39c12", "中危"), "高危": ("#e74c3c", "高危"), "未知": ("#6c757d", "未知")}
    color, text = level_map.get(level, level_map["未知"])
    if value is not None:
        text = f"{text} · {value:.2f}"
    st.markdown(f'<span class="risk-badge" style="background:linear-gradient(135deg,{color},#183b63);">{text}</span>', unsafe_allow_html=True)


# 渲染指标卡片，展示关键数值并保持页面视觉一致
def render_metric_card(label="指标", value="--", unit=""):
    """渲染单个指标卡片。"""
    st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-subtext">{unit}</div></div>', unsafe_allow_html=True)


# 渲染分节标题，规范页面结构和视觉层级
def render_section_title(title="分节标题"):
    """渲染分节标题。"""
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
