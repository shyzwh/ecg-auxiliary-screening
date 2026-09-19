"""Reusable UI helpers for the medical ECG dashboard."""

import streamlit as st


def _icon_html(icon):
    """过滤 Material 图标占位符，保留普通文本图标。"""
    if str(icon).startswith(":material/") and str(icon).endswith(":"):
        return ""
    return str(icon or "")


# 注入全局样式，包含毛玻璃卡片、医疗蓝主题、悬停效果
# 统一主题和视觉组件，保证多个页面风格一致

def inject_global_css(theme="医疗蓝"):
    THEMES = {
        "医疗蓝": {"primary": "#1A5CFF", "bg": "#F5F7FA"},
        "清新绿": {"primary": "#10B981", "bg": "#F0FDF4"},
        "暖阳橙": {"primary": "#F59E0B", "bg": "#FFFBEB"},
    }
    t = THEMES.get(theme, THEMES["医疗蓝"])
    st.markdown(
        f"""
        <style>
        /* 定义医疗蓝色系与通用主题颜色 */
        :root {{
            --primary-color: {t['primary']};
            --bg-color: {t['bg']};
            --medical-blue: var(--primary-color);
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
        }}
        html, body, [data-testid="stAppViewContainer"] {{
            background: linear-gradient(135deg, var(--bg-color) 0%, #ffffff 48%, var(--bg-color) 100%);
            color: var(--text);
            font-family: "Microsoft YaHei", "PingFang SC", sans-serif;
        }}
        [data-testid="stAppViewContainer"] > .main {{ background: transparent; }}
        [data-testid="stSidebar"] {{ background: linear-gradient(180deg, var(--primary-color) 0%, #0a4aad 100%); }}
        [data-testid="stSidebar"] * {{ color: white; }}
        /* 定义毛玻璃卡片样式 */
        .glass-panel {{
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
        }}
        .glass-panel:hover {{
            transform: translateY(-2px);
            box-shadow: 0 18px 35px rgba(15, 90, 170, 0.16);
        }}
        /* 定义分节标题样式 */
        .section-title {{
            color: var(--medical-deep);
            font-size: 1.08rem;
            font-weight: 800;
            padding-left: 0.7rem;
            border-left: 4px solid var(--medical-blue);
            margin-bottom: 0.8rem;
        }}
        .risk-badge {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            padding: 0.38rem 0.8rem;
            border-radius: 999px;
            color: white;
            font-weight: 700;
            font-size: 0.8rem;
            min-width: 92px;
        }}
        .metric-card {{
            background: rgba(255,255,255,0.85);
            border: 1px solid rgba(20, 72, 133, 0.08);
            border-radius: 16px;
            padding: 0.9rem 0.8rem;
            min-height: 110px;
            box-shadow: 0 10px 20px rgba(20, 72, 133, 0.06);
            transition: transform 0.2s ease;
        }}
        .metric-card:hover {{ transform: translateY(-3px); }}
        .metric-label {{
            font-size: 0.72rem;
            color: var(--muted);
            text-transform: uppercase;
            letter-spacing: 0.04em;
            font-weight: 700;
        }}
        .metric-value {{
            margin-top: 0.45rem;
            font-size: 1.7rem;
            font-weight: 900;
            color: var(--medical-deep);
            font-family: "Consolas", "Courier New", monospace;
        }}
        .metric-subtext {{
            margin-top: 0.2rem;
            font-size: 0.74rem;
            color: var(--muted);
        }}
        .ecg-section-title {{
            display: flex;
            align-items: center;
            gap: 0.45rem;
            color: var(--medical-deep);
            font-size: 1.08rem;
            font-weight: 800;
            padding: 0.2rem 0 0.2rem 0.7rem;
            border-left: 4px solid var(--medical-blue);
            margin: 1rem 0 0.8rem;
        }}
        .material-symbols-rounded {{
            font-family: "Material Symbols Rounded";
            font-size: 1.1em;
            vertical-align: -0.12em;
            margin-right: 0.25rem;
        }}
        .ecg-info-card,
        .ecg-stat-card,
        .ecg-feature-card,
        .ecg-warning-card {{
            border-radius: 16px;
            padding: 1rem 1.1rem;
            margin: 0.35rem 0 0.9rem;
            min-height: 0;
        }}
        .ecg-info-card {{
            background: rgba(255, 255, 255, 0.9);
            border: 1px solid rgba(20, 72, 133, 0.1);
            box-shadow: 0 10px 24px rgba(20, 72, 133, 0.08);
        }}
        .ecg-info-card.ecg-green-card {{
            background: #e8f5e9;
            border-color: #a5d6a7;
            color: #1b5e20;
        }}
        .ecg-info-card.ecg-blue-card {{
            background: #edf5ff;
            border-color: #b7d5ff;
        }}
        .ecg-card-title {{
            color: var(--medical-deep);
            font-size: 1rem;
            font-weight: 800;
            margin-bottom: 0.45rem;
        }}
        .ecg-green-card .ecg-card-title {{ color: #1b5e20; }}
        .ecg-card-content {{
            color: var(--text);
            line-height: 1.7;
            white-space: pre-wrap;
        }}
        .ecg-green-card .ecg-card-content {{ color: #1b5e20; }}
        .ecg-stat-card {{
            background: rgba(255, 255, 255, 0.92);
            border: 1px solid rgba(20, 72, 133, 0.1);
            box-shadow: 0 10px 24px rgba(20, 72, 133, 0.08);
            text-align: center;
        }}
        .ecg-stat-value {{
            color: var(--medical-deep);
            font-size: 1.7rem;
            font-weight: 900;
            line-height: 1.2;
        }}
        .ecg-stat-label {{
            color: var(--muted);
            font-size: 0.78rem;
            font-weight: 700;
            margin-top: 0.35rem;
        }}
        .ecg-stat-unit {{
            color: var(--muted);
            font-size: 0.75rem;
            margin-left: 0.2rem;
        }}
        .ecg-feature-card {{
            background: rgba(255, 255, 255, 0.88);
            border: 1px solid rgba(20, 72, 133, 0.1);
            box-shadow: 0 8px 20px rgba(20, 72, 133, 0.06);
            min-height: 105px;
        }}
        .ecg-feature-icon {{ font-size: 1.2rem; margin-right: 0.35rem; }}
        .ecg-feature-title {{ color: var(--medical-deep); font-weight: 800; }}
        .ecg-feature-desc {{ color: var(--muted); line-height: 1.6; margin-top: 0.45rem; }}
        .ecg-warning-card {{
            background: #fff8e1;
            border: 1px solid #f2c66d;
            box-shadow: 0 8px 18px rgba(173, 119, 20, 0.08);
            color: #7a4f01;
            line-height: 1.7;
        }}
        .stButton > button {{
            border-radius: 12px;
            font-weight: 700;
            transition: all 0.2s ease;
        }}
        .stButton > button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(15,111,255,0.16);
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# 渲染患者信息卡片
# 用于在分析页展示已录入的患者基本信息

def render_patient_card(patient_name="未命名患者", age="未录入", sex="未指定"):
    """渲染患者基础信息卡片。"""
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
# 展示文件选择框与示例数据入口，方便前台快速试用

def render_upload_card():
    """渲染 ECG 文件上传区和示例数据按钮。"""
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
# 通过统一颜色和文字给出当前风险状态

def render_risk_badge(level="低危", value=None):
    """按风险等级渲染颜色徽章。"""
    level_map = {
        "低危": ("#52c41a", "低危"),
        "中危": ("#faad14", "中危"),
        "高危": ("#ff4d4f", "高危"),
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
def render_section_title(title="分节标题", icon=""):
    st.markdown(f'<div class="ecg-section-title">{_icon_html(icon)}{title}</div>', unsafe_allow_html=True)


def render_info_card(title, content, icon="", variant=""):
    class_name = f"ecg-info-card {variant}".strip()
    st.markdown(
        f'<div class="{class_name}"><div class="ecg-card-title">{_icon_html(icon)}{title}</div><div class="ecg-card-content">{content}</div></div>',
        unsafe_allow_html=True,
    )


def render_stat_card(label, value, unit="", icon=""):
    st.markdown(
        f'<div class="ecg-stat-card"><div class="ecg-stat-value">{_icon_html(icon)}{value}<span class="ecg-stat-unit">{unit}</span></div><div class="ecg-stat-label">{label}</div></div>',
        unsafe_allow_html=True,
    )


def render_feature_card(title, desc, icon=""):
    st.markdown(
        f'<div class="ecg-feature-card"><div class="ecg-feature-title"><span class="ecg-feature-icon">{_icon_html(icon)}</span>{title}</div><div class="ecg-feature-desc">{desc}</div></div>',
        unsafe_allow_html=True,
    )


def render_warning_card(text):
    st.markdown(f'<div class="ecg-warning-card">{text}</div>', unsafe_allow_html=True)
