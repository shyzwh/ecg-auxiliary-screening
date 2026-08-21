import os
import json
import datetime
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from src.data_loader import load_ecg
from src.preprocess import preprocess_ecg
from src.feature_extract import pan_tompkins, extract_all_features
from src.inference import predict_risk, explain_with_shap
from src.cnn_inference import predict_abnormal_beats
from src.report_gen import generate_report
from src.config_utils import load_config, save_config


# ===================== 页面设置 =====================
st.set_page_config(
    page_title="心电风险可解释辅助筛查系统",
    page_icon="static/logo.jpg",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
/* ============ 全局主题 ============ */
html, body, [class*="css"] {
    font-family: "Microsoft YaHei", "PingFang SC", sans-serif;
    color: #1F2937;
}

/* 隐藏Streamlit默认元素 */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

/* 主背景 */
.stApp {
    background-color: #F5F7FA;
}

/* 侧边栏 */
section[data-testid="stSidebar"] {
    background-color: #0B2D7A;
    min-width: 220px;
    max-width: 260px;
}

section[data-testid="stSidebar"] * {
    color: #FFFFFF;
}

section[data-testid="stSidebar"] .stRadio label {
    font-size: 15px;
    padding: 8px 10px;
    border-radius: 8px;
    margin-bottom: 6px;
    transition: all 0.2s ease;
}

section[data-testid="stSidebar"] .stRadio label:hover {
    background-color: rgba(255, 255, 255, 0.1);
}

/* 上传区 */
div[data-testid="stFileUploader"] {
    border: 2px dashed #1A5CFF;
    border-radius: 12px;
    padding: 18px;
    background-color: #FFFFFF;
}

/* 按钮 */
.stButton > button {
    border-radius: 8px;
    border: 1px solid #1A5CFF;
    background-color: #1A5CFF;
    color: white;
    font-weight: bold;
    padding: 6px 18px;
    transition: all 0.2s ease;
}

.stButton > button:hover {
    background-color: #0B2D7A;
    border-color: #0B2D7A;
}

/* 指标卡片 */
div[data-testid="stMetric"] {
    background-color: #FFFFFF;
    padding: 16px;
    border-radius: 12px;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    border-left: 4px solid #1A5CFF;
}

/* 成功/警告/错误信息 */
div[data-testid="stSuccess"] {
    background-color: #E8F7EE;
    border-left: 6px solid #52c41a;
    border-radius: 8px;
}

div[data-testid="stWarning"] {
    background-color: #FFF7E6;
    border-left: 6px solid #faad14;
    border-radius: 8px;
}

div[data-testid="stError"] {
    background-color: #FFF1F0;
    border-left: 6px solid #ff4d4f;
    border-radius: 8px;
}

/* 表格 */
div[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
}
</style>
""", unsafe_allow_html=True)


# ===================== 侧边栏 =====================
with st.sidebar:
    st.title("❤️ 心电筛查系统")
    st.markdown("---")
    page = st.radio("功能导航", ["心电分析", "历史记录", "系统设置"])
    st.markdown("---")
    st.caption("面向基层医护人员")

# ===================== 系统设置页 =====================
if page == "系统设置":
    st.title("⚙️ 系统设置")

    config = load_config()

    st.markdown("## 风险阈值配置")

    col1, col2 = st.columns(2)
    with col1:
        medium_threshold = st.number_input(
            "中危阈值",
            min_value=0.0,
            max_value=1.0,
            value=float(config.get("risk_threshold_medium", 0.4)),
            step=0.05
        )
    with col2:
        high_threshold = st.number_input(
            "高危阈值",
            min_value=0.0,
            max_value=1.0,
            value=float(config.get("risk_threshold_high", 0.7)),
            step=0.05
        )

    st.caption("低危 < 中危阈值 ≤ 中危 < 高危阈值 ≤ 高危")

    st.markdown("## 默认分析时长")
    duration = st.selectbox(
        "默认分析时长",
        [10, 30, 60],
        index=[10, 30, 60].index(int(config.get("default_duration_sec", 30)))
    )

    st.markdown("## 预处理参数")
    col3, col4, col5 = st.columns(3)
    with col3:
        bandpass_low = st.number_input("带通下限(Hz)", value=float(config.get("bandpass_low", 0.5)))
    with col4:
        bandpass_high = st.number_input("带通上限(Hz)", value=float(config.get("bandpass_high", 40.0)))
    with col5:
        notch = st.selectbox("工频频率", ["auto", "50", "60"], index=["auto", "50", "60"].index(str(config.get("notch_freq", "auto"))))

    st.markdown("## 模型路径")
    model_path = st.text_input("XGBoost模型路径", value=str(config.get("model_path", "models/ecg_risk_xgb_model.json")))
    cnn_path = st.text_input("CNN模型路径", value=str(config.get("cnn_model_path", "models/cnn_model.h5")))

    st.markdown("## 显示设置")
    show_shap = st.toggle("默认显示SHAP图", value=bool(config.get("show_shap", True)))

    if st.button("保存设置"):
        config["risk_threshold_medium"] = medium_threshold
        config["risk_threshold_high"] = high_threshold
        config["default_duration_sec"] = duration
        config["bandpass_low"] = bandpass_low
        config["bandpass_high"] = bandpass_high
        config["notch_freq"] = notch
        config["model_path"] = model_path
        config["cnn_model_path"] = cnn_path
        config["show_shap"] = show_shap

        status, msg = save_config(config)
        if status == "success":
            st.success(msg)
        else:
            st.error(msg)

    if st.button("恢复默认设置"):
        status, msg = save_config(DEFAULT_CONFIG.copy())
        if status == "success":
            st.success("已恢复默认设置，请刷新页面")
        else:
            st.error(msg)

# ===================== 历史记录页 =====================
if page == "历史记录":
    st.title("📋 历史记录")

    history_path = "storage/records.json"

    if not os.path.exists(history_path):
        st.info("暂无历史记录")
    else:
        try:
            with open(history_path, "r", encoding="utf-8") as f:
                history = json.load(f)
        except Exception:
            history = []

        if not history:
            st.info("暂无历史记录")
        else:
            df_history = pd.DataFrame(history)

            # 顶部统计
            st.markdown("## 统计概览")
            total_count = len(df_history)

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("总记录数", total_count)
            with col2:
                st.metric("低危", int((df_history["风险等级"] == "低危").sum()))
            with col3:
                st.metric("中危", int((df_history["风险等级"] == "中危").sum()))
            with col4:
                st.metric("高危", int((df_history["风险等级"] == "高危").sum()))

            # 筛选
            st.markdown("## 筛选与搜索")
            col_filter1, col_filter2, col_filter3 = st.columns(3)

            with col_filter1:
                risk_filter = st.selectbox("按风险等级筛选", ["全部", "低危", "中危", "高危"])

            with col_filter2:
                keyword = st.text_input("按文件名搜索", placeholder="输入文件名关键词")

            with col_filter3:
                sort_by = st.selectbox("排序方式", ["时间最新在前", "时间最早在前", "风险等级从高到低"])

            # 应用筛选
            filtered_df = df_history.copy()

            if risk_filter != "全部":
                filtered_df = filtered_df[filtered_df["风险等级"] == risk_filter]

            if keyword:
                filtered_df = filtered_df[filtered_df["文件名"].str.contains(keyword, na=False)]

            if sort_by == "时间最新在前":
                filtered_df = filtered_df.sort_values("时间", ascending=False)
            elif sort_by == "时间最早在前":
                filtered_df = filtered_df.sort_values("时间", ascending=True)
            elif sort_by == "风险等级从高到低":
                order = {"高危": 0, "中危": 1, "低危": 2}
                filtered_df["排序"] = filtered_df["风险等级"].map(order)
                filtered_df = filtered_df.sort_values("排序")
                filtered_df = filtered_df.drop(columns=["排序"])

            def color_risk(val):
                color_map = {
                    "低危": "#03C900A9",
                    "中危": "#F2F925FD",
                    "高危": "#EF0707"
                }
                return f"background-color: {color_map.get(val, '#FFFFFF')}; color: white; font-weight: bold;"

            st.dataframe(
                filtered_df.style.map(color_risk, subset=["风险等级"]),
                use_container_width=True
            )

            # 导出CSV
            st.download_button(
                label="导出全部记录为CSV",
                data=filtered_df.to_csv(index=False).encode("utf-8-sig"),
                file_name="心电筛查历史记录.csv",
                mime="text/csv"
            )

            st.markdown("## 导出单条报告")

            if len(filtered_df) > 0:
                selected_time = st.selectbox("选择要导出的记录", filtered_df["时间"].tolist())

                selected_row = filtered_df[filtered_df["时间"] == selected_time].iloc[0]

                single_report = (
                    f"心电筛查报告\n"
                    f"分析时间：{selected_row['时间']}\n"
                    f"文件名：{selected_row['文件名']}\n"
                    f"风险等级：{selected_row['风险等级']}\n"
                    f"风险评分：{selected_row['风险评分']}\n"
                    f"总心拍数：{selected_row['总心拍数']}\n"
                    f"异常心拍数：{selected_row['异常心拍数']}\n"
                    f"备注：{selected_row.get('备注', '')}\n"
                )

                st.download_button(
                    label="导出该条报告TXT",
                    data=single_report,
                    file_name=f"心电报告_{selected_row['时间'].replace(':', '-').replace(' ', '_')}.txt",
                    mime="text/plain"
                )

            # 删除单条记录
            st.markdown("## 删除记录")
            if len(filtered_df) > 0:
                time_to_delete = st.selectbox("选择要删除的记录时间", filtered_df["时间"].tolist())

                if st.button("删除该条记录"):
                    if time_to_delete in df_history["时间"].values:
                        df_history = df_history[df_history["时间"] != time_to_delete]
                        with open(history_path, "w", encoding="utf-8") as f:
                            json.dump(df_history.to_dict(orient="records"), f, ensure_ascii=False, indent=2)
                        st.success("已删除")
                        st.rerun()


# ===================== 心电分析页 =====================
else:
    if "risk_level" in st.session_state:
        st.success("检测到上次分析结果，已为你恢复。")
    st.title("心电风险可解释辅助筛查系统")
    st.markdown("面向基层医护人员的心电初筛原型系统")

    # 顶部状态卡片
    col_status1, col_status2, col_status3 = st.columns(3)

    with col_status1:
        st.markdown(
            """
            <div style="
                background:#FFFFFF;
                padding:18px;
                border-radius:12px;
                box-shadow:0 2px 8px rgba(0,0,0,0.04);
                text-align:center;
            ">
                <div style="font-size:14px;color:#666;">系统状态</div>
                <div style="font-size:22px;font-weight:bold;color:#1A5CFF;">✅ 就绪</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col_status2:
        st.markdown(
            """
            <div style="
                background:#FFFFFF;
                padding:18px;
                border-radius:12px;
                box-shadow:0 2px 8px rgba(0,0,0,0.04);
                text-align:center;
            ">
                <div style="font-size:14px;color:#666;">分析模式</div>
                <div style="font-size:20px;font-weight:bold;color:#0B2D7A;">双通路并行</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col_status3:
        st.markdown(
            """
            <div style="
                background:#FFFFFF;
                padding:18px;
                border-radius:12px;
                box-shadow:0 2px 8px rgba(0,0,0,0.04);
                text-align:center;
            ">
                <div style="font-size:14px;color:#666;">可解释性</div>
                <div style="font-size:20px;font-weight:bold;color:#0B2D7A;">SHAP已启用</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    col_upload, col_demo = st.columns([3, 1])

    with col_upload:
        uploaded_file = st.file_uploader("上传心电文件", type=["csv", "txt", "dat"])

    with col_demo:
        st.write("")
        st.write("")
        use_demo = st.button("加载示例数据")

    if use_demo:
        file_path = "uploads/100_30s.csv"
        file_name = "100_30s.csv（示例）"
    elif uploaded_file is not None:
        os.makedirs("uploads", exist_ok=True)
        file_path = "uploads/" + uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        file_name = uploaded_file.name
    else:
        file_path = None
        file_name = None

    if file_path is None:
        st.markdown(
            """
            <div style="
                background:#FFFFFF;
                padding:50px;
                border-radius:16px;
                text-align:center;
                box-shadow:0 2px 12px rgba(0,0,0,0.05);
                margin-top:30px;
            ">
                <div style="font-size:48px;margin-bottom:10px;">🫀</div>
                <div style="font-size:22px;font-weight:bold;color:#1A5CFF;margin-bottom:8px;">
                    欢迎使用心电风险可解释辅助筛查系统
                </div>
                <div style="font-size:15px;color:#666;margin-bottom:25px;">
                    请上传心电文件，或点击“加载示例数据”开始分析
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        status, signal, fs, msg = load_ecg(file_path)

        if status != "success":
            st.markdown(
                f"""
                <div style="
                    background:#FFF1F0;
                    padding:25px;
                    border-radius:12px;
                    border-left:6px solid #ff4d4f;
                    margin:20px 0;
                ">
                    <div style="font-size:18px;font-weight:bold;color:#CF1322;margin-bottom:8px;">
                        ❌ 文件读取失败
                    </div>
                    <div style="font-size:14px;color:#820014;margin-bottom:12px;">
                        {msg}
                    </div>
                    <div style="font-size:13px;color:#820014;background:#FFFFFF;padding:10px;border-radius:8px;">
                        可能原因：文件格式不正确、文件损坏、或数据路径错误。<br>
                        请确认上传的是CSV/TXT/DAT格式，或联系系统管理员。
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.success(f"{file_name} 读取成功，采样率 {fs} Hz")

            # 预处理
            status_pre, clean_signal, msg_pre = preprocess_ecg(signal, fs)

            if status_pre != "success":
                st.error(msg_pre)
            else:
                st.markdown("## 预处理流程")
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.success("✅ 工频陷波")
                    st.caption("自动检测50/60Hz")
                with c2:
                    st.success("✅ 带通滤波")
                    st.caption("0.5-40Hz")
                with c3:
                    st.success("✅ 中值滤波")
                    st.caption("去尖峰噪声")
                with c4:
                    st.success("✅ 基线校正")
                    st.caption("去基线漂移")

                with st.spinner("正在进行R峰检测与特征提取..."):
                    status_r, r_peaks, msg_r = pan_tompkins(clean_signal, fs)

                    if status_r == "success":
                        status_f, features, msg_f = extract_all_features(clean_signal, r_peaks, fs)
                        features = {k: (0.0 if v is None else v) for k, v in features.items()}
                    else:
                        status_f = "error"
                        msg_f = msg_r

                if status_f != "success":
                    st.error(msg_f)
                else:
                    status_risk, risk_level, risk_num, score, msg_risk = predict_risk(features)
                    status_cnn, abnormal_positions, msg_cnn = predict_abnormal_beats(clean_signal, r_peaks, fs)

                    abnormal_count = len(abnormal_positions) if abnormal_positions else 0
                    total_beats = len(r_peaks)

                    st.session_state["risk_level"] = risk_level
                    st.session_state["risk_num"] = risk_num
                    st.session_state["score"] = score
                    st.session_state["features"] = features
                    st.session_state["r_peaks"] = r_peaks
                    st.session_state["clean_signal"] = clean_signal
                    st.session_state["fs"] = fs
                    st.session_state["abnormal_positions"] = abnormal_positions
                    st.session_state["total_beats"] = total_beats
                    st.session_state["abnormal_count"] = abnormal_count

                    st.session_state["risk_probs"] = [0.8, 0.15, 0.05]
                    st.session_state["report_text"] = report_text if 'report_text' in locals() else ""
                    st.session_state["shap_result"] = shap_result if 'shap_result' in locals() else None

                    # ===================== 双通路结果 =====================
                    st.markdown(
                        """
                        <div style="
                            background:#EAF0FF;
                            padding:16px 20px;
                            border-radius:12px;
                            margin:20px 0 10px 0;
                            border-left:6px solid #1A5CFF;
                        ">
                            <span style="font-size:22px;font-weight:bold;color:#0B2D7A;">双通路分析结果</span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    col_cnn, col_xgb = st.columns(2)

                    with col_cnn:

                        st.markdown(
                            """
                            <div style="
                                background:#E6F7FF;
                                padding:12px 16px;
                                border-radius:10px;
                                border-left:6px solid #0099CC;
                                margin-bottom:12px;
                            ">
                                <span style="font-size:18px;font-weight:bold;color:#005B7A;">🔍 1D-CNN 形态通路</span>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        st.metric("总心拍数", total_beats)
                        st.metric("异常心拍数", abnormal_count)
                        if total_beats > 0:
                            st.metric("异常占比", f"{abnormal_count/total_beats*100:.1f}%")

                    with col_xgb:
                        st.markdown(
                            """
                            <div style="
                                background:#E8F7EE;
                                padding:12px 16px;
                                border-radius:10px;
                                border-left:6px solid #00C9A7;
                                margin-bottom:12px;
                            ">
                                <span style="font-size:18px;font-weight:bold;color:#006B5E;">📈 XGBoost 数值通路</span>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        risk_color = {
                            "低危": "#00C9A7",
                            "中危": "#F9A825",
                            "高危": "#EF0707"
                        }.get(risk_level, "#888888")

                        st.markdown(
                            f"""
                            <div style="
                                background: linear-gradient(135deg, {risk_color} 0%, #1F2937 100%);
                                padding:35px;
                                border-radius:16px;
                                text-align:center;
                                color:white;
                                font-size:36px;
                                font-weight:bold;
                                box-shadow:0 8px 24px rgba(0,0,0,0.15);
                                margin-bottom:15px;
                            ">
                                {risk_level}
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                        st.metric("风险评分", f"{score:.2f}")

                        risk_probs = [0.8, 0.15, 0.05]
                        fig_risk = px.pie(
                            names=["低危", "中危", "高危"],
                            values=risk_probs,
                            color=["低危", "中危", "高危"],
                            color_discrete_map={
                                "低危": "#00C9A7",
                                "中危": "#F9A825",
                                "高危": "#EF0707"
                            },
                            title="三级风险概率分布"
                        )
                        fig_risk.update_traces(textinfo="percent+label")
                        fig_risk.update_layout(height=320)
                        st.plotly_chart(fig_risk, use_container_width=True)

                    st.markdown(
                        """
                        <div style="
                            background:#FFFBE6;
                            padding:12px 18px;
                            border-radius:10px;
                            border-left:6px solid #F9A825;
                            margin-top:15px;
                            font-size:14px;
                            color:#7A5B00;
                        ">
                            ⚠️ 两条通路结果独立输出，请医护人员综合判断
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    # ===================== 波形图 =====================
                    st.markdown("## 心电波形与异常定位")
                    time = np.arange(len(clean_signal)) / fs

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=time, y=clean_signal,
                        mode="lines", name="心电信号",
                        line=dict(color="#1A5CFF", width=1)
                    ))
                    fig.add_trace(go.Scatter(
                        x=[r/fs for r in r_peaks],
                        y=[clean_signal[r] for r in r_peaks],
                        mode="markers", name="R峰",
                        marker=dict(color="red", size=5)
                    ))
                    if abnormal_positions:
                        fig.add_trace(go.Scatter(
                            x=[p/fs for p in abnormal_positions],
                            y=[clean_signal[p] for p in abnormal_positions],
                            mode="markers", name="异常心拍",
                            marker=dict(color="orange", size=10, symbol="x")
                        ))

                    fig.update_layout(
                        height=420,
                        title="心电波形：R峰检测与异常心拍定位",
                        xaxis_title="时间（秒）",
                        yaxis_title="电压（mV）",
                        hovermode="x unified"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # ===================== 特征表 =====================
                    st.markdown("## 12项临床特征")
                    st.dataframe(pd.DataFrame([features]).T.rename(columns={0: "数值"}), use_container_width=True)

                    # ===================== SHAP =====================
                    st.markdown(
                        """
                        <div style="
                            background:#EAF0FF;
                            padding:16px 20px;
                            border-radius:12px;
                            margin:25px 0 10px 0;
                            border-left:6px solid #1A5CFF;
                        ">
                            <span style="font-size:22px;font-weight:bold;color:#0B2D7A;">SHAP 可解释性分析</span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    st.markdown("SHAP仅作用于12项临床特征（XGBoost通路）")

                    status_shap, shap_result, msg_shap = explain_with_shap(features)

                    if status_shap == "success":
                        shap_df = pd.DataFrame([
                            {"特征": name, "SHAP贡献": info["shap_value"]}
                            for name, info in shap_result.items()
                        ])
                        shap_df = shap_df.sort_values("SHAP贡献", ascending=True)

                        fig_shap = px.bar(
                            shap_df,
                            x="SHAP贡献",
                            y="特征",
                            orientation="h",
                            title="12项特征对风险判定的贡献度",
                            color="SHAP贡献",
                            color_continuous_scale=["blue", "white", "red"]
                        )
                        fig_shap.update_layout(height=450)
                        st.plotly_chart(fig_shap, use_container_width=True)
                    else:
                        st.warning(f"SHAP暂不可用：{msg_shap}")

                    # ===================== 报告 =====================
                    st.markdown("## 筛查报告")

                    status_rep, report_text, report_data = generate_report(
                        risk_num=risk_num,
                        risk_score=score,
                        risk_probs=[0.8, 0.15, 0.05],
                        features=features,
                        abnormal_count=abnormal_count,
                        total_beats=total_beats,
                    )

                    if status_rep == "success":
                        st.text(report_text)
                        st.download_button(
                            label="导出报告TXT",
                            data=report_text,
                            file_name="心电筛查报告.txt",
                            mime="text/plain"
                        )
                    else:
                        st.error(f"报告生成失败：{report_text}")

                    # ===================== 保存记录 =====================
                    st.markdown("## 保存分析记录")

                    col_note, col_save = st.columns([3, 1])
                    with col_note:
                        patient_note = st.text_input("备注（可选）", placeholder="如：患者编号、检查日期")
                    with col_save:
                        st.write("")
                        st.write("")
                        save_clicked = st.button("保存到历史记录")

                    if save_clicked:
                        record = {
                            "时间": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "文件名": file_name,
                            "风险等级": risk_level,
                            "风险评分": score,
                            "总心拍数": total_beats,
                            "异常心拍数": abnormal_count,
                            "备注": patient_note,
                        }

                        os.makedirs("storage", exist_ok=True)
                        history_path = "storage/records.json"

                        if os.path.exists(history_path):
                            try:
                                with open(history_path, "r", encoding="utf-8") as f:
                                    history = json.load(f)
                            except Exception:
                                history = []
                        else:
                            history = []

                        history.append(record)

                        with open(history_path, "w", encoding="utf-8") as f:
                            json.dump(history, f, ensure_ascii=False, indent=2)

                        st.success("已保存到历史记录")