<<<<<<< HEAD
﻿from pathlib import Path
import json
=======
import json
import os
from io import BytesIO
from xml.sax.saxutils import escape

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from .logger import logger
from .config_utils import DEFAULT_CONFIG
>>>>>>> 5bdb321dece5bc57619bf68e89bf0244b0465ab4

from src.logger import logger

DEFAULT_THRESHOLDS = {
    "HR": {"low": 50, "high": 100, "mild_low": 45, "mild_high": 120, "unit": "次/分"},
    "PR": {"low": 120, "high": 200, "mild_low": 100, "mild_high": 240, "unit": "ms"},
    "QRS": {"high": 120, "mild_high": 150, "unit": "ms"},
    "QT": {"unit": "ms"},
    "QTc": {"high": 450, "mild_high": 500, "unit": "ms"},
    "ST_shift": {"abs_normal": 0.1, "abs_mild": 0.2, "unit": "mV"},
    "P_amp": {"low": 0.05, "high": 0.25, "mild_low": 0.02, "unit": "mV"},
    "T_amp": {"low": 0.1, "high": 0.5, "mild_low": 0.05, "unit": "mV"},
    "RR_mean": {"low": 600, "high": 1200, "mild_low": 400, "mild_high": 1500, "unit": "ms"},
    "RR_std": {"high": 100, "mild_high": 200, "unit": "ms"},
    "SDNN": {"low": 50, "mild_low": 30, "unit": "ms"},
    "RMSSD": {"low": 20, "high": 80, "mild_low": 10, "mild_high": 120, "unit": "ms"},
}

FEATURE_ORDER = [
    "HR", "PR", "QRS", "QT", "QTc", "ST_shift",
    "P_amp", "T_amp", "RR_mean", "RR_std", "SDNN", "RMSSD"
]


def _register_pdf_font():
    font_candidates = [
        os.path.join(os.environ.get("WINDIR", "C:/Windows"), "Fonts", "Deng.ttf"),
        os.path.join(os.environ.get("WINDIR", "C:/Windows"), "Fonts", "NotoSansSC-VF.ttf"),
    ]
    for font_path in font_candidates:
        if os.path.exists(font_path):
            try:
                pdfmetrics.registerFont(TTFont("ECGChinese", font_path))
                return "ECGChinese"
            except Exception:
                continue
    return "Helvetica"


def _pdf_plot_images(result, font_name):
    matplotlib.rcParams["font.sans-serif"] = ["DengXian", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    images = []
    signal = np.asarray(result.get("signal", []), dtype=float)
    fs = float(result.get("fs", 1) or 1)
    if signal.size:
        figure, axis = plt.subplots(figsize=(10, 3.1))
        axis.plot(np.arange(signal.size) / fs, signal, color="#1677ff", linewidth=0.7)
        peaks = np.asarray(result.get("r_peaks", []), dtype=int)
        abnormal = np.asarray(result.get("abnormal_positions", []), dtype=int)
        if peaks.size:
            peaks = peaks[(peaks >= 0) & (peaks < signal.size)]
            axis.scatter(peaks / fs, signal[peaks], color="#ff4d4f", s=12, label="R峰")
        if abnormal.size:
            abnormal = abnormal[(abnormal >= 0) & (abnormal < signal.size)]
            axis.scatter(abnormal / fs, signal[abnormal], color="#fa8c16", marker="x", s=28, label="异常心拍")
        axis.set_title("单导联 ECG 波形")
        axis.set_xlabel("时间（秒）")
        axis.set_ylabel("电压")
        axis.grid(alpha=0.25)
        axis.legend(loc="upper right")
        figure.tight_layout()
        buffer = BytesIO()
        figure.savefig(buffer, format="png", dpi=150)
        plt.close(figure)
        buffer.seek(0)
        images.append(("waveform", buffer))

    visualization = result.get("shap_visualizations") or {}
    global_data = visualization.get("global")
    if global_data:
        shap_values = np.asarray(global_data.get("shap_values", []), dtype=float)
        if shap_values.ndim == 2 and shap_values.shape[1] == len(FEATURE_ORDER):
            importance = np.abs(shap_values).mean(axis=0)
            order = np.argsort(importance)
            figure, axis = plt.subplots(figsize=(10, 3.6))
            axis.barh(np.array(FEATURE_ORDER)[order], importance[order], color="#13c2c2")
            axis.set_title(f"全局 SHAP 特征重要性（目标类别：{visualization.get('target_label', '未知')}）")
            axis.set_xlabel("平均绝对 SHAP 值")
            figure.tight_layout()
            buffer = BytesIO()
            figure.savefig(buffer, format="png", dpi=150)
            plt.close(figure)
            buffer.seek(0)
            images.append(("shap_global", buffer))
    return images


def generate_pdf_report(result):
    """生成包含波形、SHAP和结构化建议的中文PDF报告。"""
    try:
        font_name = _register_pdf_font()
        styles = getSampleStyleSheet()
        body = ParagraphStyle("ECGBody", parent=styles["BodyText"], fontName=font_name, fontSize=9, leading=14, spaceAfter=4)
        heading = ParagraphStyle("ECGHeading", parent=styles["Heading2"], fontName=font_name, fontSize=13, leading=18, textColor=colors.HexColor("#0b2d7a"), spaceBefore=8, spaceAfter=6)
        title = ParagraphStyle("ECGTitle", parent=heading, alignment=TA_CENTER, fontSize=18, leading=24)
        story = [Paragraph("心电风险辅助筛查报告", title), Spacer(1, 5 * mm)]
        patient = result.get("patient_info", {})
        story.append(Paragraph("患者信息", heading))
        story.append(Paragraph("姓名：{}　年龄：{}　性别：{}".format(escape(str(patient.get("name", "未填写"))), escape(str(patient.get("age", "未填写"))), escape(str(patient.get("sex", "未指定")))), body))
        story.append(Paragraph("文件：{}　时间：{}".format(escape(str(result.get("file_name", ""))), escape(str(result.get("created_at", "")))), body))
        story.append(Paragraph("风险等级：{}　复核状态：{}".format(escape(str(result.get("risk_level", "未知"))), escape(str(result.get("review_status", "未生成")))), body))
        story.append(Paragraph("双通路结果", heading))
        path_data = [["通路", "结果", "置信度"], ["1D-CNN", "异常心拍 {} 个".format(len(result.get("abnormal_positions", []))), "{:.2%}".format(float(result.get("cnn_confidence", 0)))], ["XGBoost", str(result.get("risk_level", "未知")), "{:.2%}".format(float(result.get("xgb_confidence", result.get("score", 0))))]]
        table = Table(path_data, colWidths=[35 * mm, 75 * mm, 45 * mm])
        table.setStyle(TableStyle([("FONTNAME", (0, 0), (-1, -1), font_name), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e6f0ff")), ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#b7c9e2")), ("FONTSIZE", (0, 0), (-1, -1), 9), ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
        story.append(table)
        story.append(Paragraph("波形与异常心拍", heading))
        for name, image_buffer in _pdf_plot_images(result, font_name):
            if name == "waveform":
                story.append(Image(image_buffer, width=175 * mm, height=54 * mm))
        story.append(Paragraph("12项特征", heading))
        feature_rows = [["特征", "数值", "状态"]]
        judgements = (result.get("report_data") or {}).get("feature_judgements", {})
        for name, value in result.get("features", {}).items():
            feature_rows.append([str(name), str(value), str(judgements.get(name, {}).get("status", "未检测"))])
        feature_table = Table(feature_rows, colWidths=[45 * mm, 55 * mm, 45 * mm])
        feature_table.setStyle(TableStyle([("FONTNAME", (0, 0), (-1, -1), font_name), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e6f0ff")), ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#b7c9e2")), ("FONTSIZE", (0, 0), (-1, -1), 8)]))
        story.append(feature_table)
        story.append(PageBreak())
        story.append(Paragraph("SHAP可解释性分析", heading))
        visualization = result.get("shap_visualizations") or {}
        story.append(Paragraph("解释目标类别：{}（类别 {}）".format(escape(str(visualization.get("target_label", "未知"))), escape(str(visualization.get("target_class", "未知")))), body))
        for name, image_buffer in _pdf_plot_images(result, font_name):
            if name == "shap_global":
                story.append(Image(image_buffer, width=175 * mm, height=63 * mm))
        story.append(Paragraph("单样本贡献", heading))
        shap_rows = [["特征", "SHAP贡献"]]
        for name, info in (result.get("shap_result") or {}).items():
            shap_rows.append([str(name), "{:.6f}".format(float(info.get("shap_value", 0)))])
        shap_table = Table(shap_rows, colWidths=[75 * mm, 55 * mm])
        shap_table.setStyle(TableStyle([("FONTNAME", (0, 0), (-1, -1), font_name), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e6f0ff")), ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#b7c9e2")), ("FONTSIZE", (0, 0), (-1, -1), 8)]))
        story.append(shap_table)
        suggestion = (result.get("report_data") or {}).get("suggestion", {})
        story.append(Paragraph("结构化健康建议", heading))
        for label, key in [("医生专业建议", "medical_advice"), ("通俗解释", "plain_language"), ("医生沟通话术", "doctor_communication"), ("复查建议", "follow_up"), ("生活方式", "lifestyle")]:
            story.append(Paragraph("{}：{}".format(label, escape(str(suggestion.get(key, "请咨询医生")))), body))
        story.append(Paragraph("医生审核提示与免责声明", heading))
        story.append(Paragraph("本报告仅供参考，请咨询医生。AI结果不能替代执业医师判断，请由专业医护人员结合临床信息审核。", body))
        output = BytesIO()
        SimpleDocTemplate(output, pagesize=A4, rightMargin=15 * mm, leftMargin=15 * mm, topMargin=15 * mm, bottomMargin=15 * mm).build(story)
        return "success", output.getvalue(), "PDF报告生成成功"
    except Exception as error:
        logger.error(f"PDF报告生成失败：{error}")
        return "error", None, f"PDF报告生成失败：{error}"


def load_suggestion_rules(suggestions_path="config/suggestions.json"):
    # 读取现有建议配置文件并保留原有6条基础规则，新增动态特征描述扩展字段。
    try:
        file_path = Path(suggestions_path)
        if not file_path.exists():
            return {"rules": [], "dynamic_feature_descriptions": {}}
        with open(file_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return {
            "rules": data.get("rules", []),
            "dynamic_feature_descriptions": data.get("dynamic_feature_descriptions", {}),
        }
    except Exception as exc:
        logger.warning(f"建议规则加载失败：{exc}")
        return {"rules": [], "dynamic_feature_descriptions": {}}


def _format_value_for_description(name, value):
    # 将真实特征值格式化为自然语言中的数值，便于生成针对性建议语句。
    if value is None:
        return "未知"
    if name in {"ST_shift", "P_amp", "T_amp"}:
        return f"{float(value):.3f}"
    return f"{float(value):.2f}"


def build_dynamic_feature_description(feature_name, value, feature_rules=None):
    # 根据真实异常特征值生成针对性描述，如 ST_shift=-0.15 时输出临床提示。
    if feature_rules is None:
        feature_rules = {}
    template = feature_rules.get(feature_name) or feature_rules.get("default", "检测到{feature}异常，建议结合临床症状复核。")
    if feature_name == "ST_shift":
        direction = "抬高" if float(value) > 0 else "压低"
        risk = "心肌缺血风险" if float(value) < -0.1 else "复极异常风险"
        return template.format(feature="ST段", direction=direction, value=_format_value_for_description(feature_name, value), risk=risk)
    if feature_name == "QTc":
        return template.format(feature="QTc", value=_format_value_for_description(feature_name, value))
    if feature_name == "HR":
        direction = "偏快" if float(value) > 100 else "偏慢"
        return template.format(feature="心率", direction=direction, value=_format_value_for_description(feature_name, value))
    if feature_name == "QRS":
        return template.format(feature="QRS时限", value=_format_value_for_description(feature_name, value))
    if feature_name == "RR_std":
        return template.format(feature="RR间期波动", value=_format_value_for_description(feature_name, value))
    return template.format(feature=feature_name, value=_format_value_for_description(feature_name, value))


def judge_feature(name, value, thresholds, sex=""):
    # 复用现有规则阈值，保留原有6条建议逻辑并输出标准化异常判断。
    if value is None:
        return "未检测", "未检测"
    t = thresholds.get(name, {})
    unit = t.get("unit", "")
    if name == "QTc":
<<<<<<< HEAD
        normal_high = 440 if sex == "男" else 460 if sex == "女" else 450
=======
        if sex == "男":
            normal_high = thresholds.get("qtc_threshold_male", DEFAULT_CONFIG["qtc_threshold_male"])
        elif sex == "女":
            normal_high = thresholds.get("qtc_threshold_female", DEFAULT_CONFIG["qtc_threshold_female"])
        else:
            normal_high = thresholds.get("qtc_threshold_default", DEFAULT_CONFIG["qtc_threshold_default"])

        mild_high = thresholds.get("qtc_threshold_mild_high", DEFAULT_CONFIG["qtc_threshold_mild_high"])

>>>>>>> 5bdb321dece5bc57619bf68e89bf0244b0465ab4
        if value < normal_high:
            return "正常", f"{value}{unit}"
        if value < 500:
            return "轻度异常", f"延长 {value}{unit}"
        return "显著异常", f"显著延长 {value}{unit}"
    if name == "ST_shift":
        abs_val = abs(value)
        direction = "抬高" if value > 0 else "压低"
        if abs_val < t.get("abs_normal", 0.1):
            return "正常", f"{value}{unit}"
        if abs_val < t.get("abs_mild", 0.2):
            return "轻度异常", f"{direction} {value}{unit}"
        return "显著异常", f"显著{direction} {value}{unit}"
    if name == "QRS":
        if value < t.get("high", 120):
            return "正常", f"{value}{unit}"
        if value < t.get("mild_high", 150):
            return "轻度异常", f"增宽 {value}{unit}"
        return "显著异常", f"显著增宽 {value}{unit}"
    low = t.get("low")
    high = t.get("high")
    mild_low = t.get("mild_low")
    mild_high = t.get("mild_high")
    if mild_low is not None and value < mild_low:
        return "显著异常", f"显著降低 {value}{unit}"
    if low is not None and value < low:
        return "轻度异常", f"偏低 {value}{unit}"
    if mild_high is not None and value > mild_high:
        return "显著异常", f"显著升高 {value}{unit}"
    if high is not None and value > high:
        return "轻度异常", f"偏高 {value}{unit}"
    return "正常", f"{value}{unit}"


def build_suggestion(risk_num, abn_level, key_abnormals, features=None, cnn_status="normal", abnormal_count=0, total_beats=0, sex="未指定"):
    """综合形态通路和数值通路生成病因分析与生活建议。"""
    features = features or {}
    st_shift = float(features.get("ST_shift", 0) or 0)
    qtc = float(features.get("QTc", 0) or 0)
    heart_rate = float(features.get("HR", 0) or 0)
    st_abs = abs(st_shift)
    st_direction = "抬高" if st_shift > 0 else "压低"

    morphology = f"形态通路检测到{abnormal_count}个异常心拍，占总心拍{(abnormal_count / max(total_beats, 1)):.1%}，异常水平为{abn_level}。"
    numeric = "；".join(f"{name}：{features.get(name)}（{judge_feature(name, features.get(name), DEFAULT_THRESHOLDS, sex=sex)[1]}）" for name in key_abnormals)
    numeric = numeric or "数值通路未发现超过当前参考阈值的特征异常。"
    if cnn_status == "normal" and risk_num == 0:
        cause = f"{morphology}{numeric}两条通路均未提示明显高风险，当前结果更支持稳定状态。"
        advice = "保持规律作息和中等强度有氧运动，每次约30分钟；无特殊症状可按常规健康管理复查，若出现胸闷胸痛立即拨打120。"
    elif cnn_status == "normal" and risk_num == 1:
        cause = f"{morphology}{numeric}提示形态通路暂未发现异常，但数值通路存在风险特征，需排查早期缺血、传导或复极改变。"
        advice = "休息并避免剧烈运动，减少咖啡因摄入，1周内复查心电图；如出现胸闷、胸痛或气促立即拨打120。"
    elif cnn_status == "normal" and risk_num == 2:
        cause = f"{morphology}{numeric}提示虽未见异常心拍，但数值通路存在严重异常，可能对应急性缺血或显著复极风险。"
        advice = "立即停止活动并静卧，禁止自行驾车，立即到心内科急诊；出现胸痛、冷汗或呼吸困难立即拨打120。"
    elif cnn_status == "abnormal" and risk_num == 0:
        cause = f"{morphology}{numeric}提示波形异常与整体数值低危并存，可能为偶发节律变化，仍需观察其持续性。"
        advice = "当天避免剧烈运动并保证充分休息，1周内预约动态心电图；若出现持续心悸、胸闷胸痛立即拨打120。"
    elif cnn_status == "abnormal" and risk_num == 1:
        cause = f"{morphology}{numeric}提示双通路异常相互印证，可能存在节律不稳并伴缺血、传导或复极风险。"
        advice = "停止高强度运动并休息，24小时内尽快到心内科就诊，按医嘱复查；出现胸闷胸痛、晕厥立即拨打120。"
    elif cnn_status == "abnormal" and risk_num == 2:
        cause = f"{morphology}{numeric}提示双通路均为高风险信号，需警惕严重心律失常、急性缺血或复极异常。"
        advice = "立即停止活动、保持静卧并呼叫急救，马上到心内科急诊；不要等待复查，胸痛、晕厥或气促时立即拨打120。"
    else:
        cause = f"{morphology}{numeric}当前通路结果需要结合临床症状进一步判断。"
        advice = "暂缓剧烈运动并休息，尽快咨询专业医护人员；出现胸闷胸痛立即拨打120。"

    return f"病因分析：{cause}\n生活建议：{advice}"


<<<<<<< HEAD
def generate_report(risk_num, risk_score, risk_probs, features, abnormal_count, total_beats, sex="未指定", cnn_status="normal"):
    """生成说明性报告文本，供 app.py 在分析页面直接展示。"""
=======
def _risk_key(risk_num):
    return {0: "low", 1: "medium", 2: "high"}.get(risk_num, "low")


def _severity_key(abn_level):
    return {"无": "none", "偶发": "mild", "频发": "moderate", "显著": "significant"}.get(abn_level, "none")


def load_suggestion_rules(path=None):
    path = path or DEFAULT_CONFIG["suggestions_path"]
    try:
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)
        return data.get("rules", []), data.get("notice", "仅供参考，请咨询医生")
    except (OSError, json.JSONDecodeError):
        return [], "仅供参考，请咨询医生"


def match_suggestion(cnn_status, risk_num, abn_level, key_abnormals, suggestions_path=None):
    rules, notice = load_suggestion_rules(suggestions_path)
    cnn_key = "abnormal" if cnn_status in ("success", "abnormal") and key_abnormals else "normal"
    risk_key = _risk_key(risk_num)
    severity_key = _severity_key(abn_level)
    features = key_abnormals or ["*"]

    def matches(rule, field, value):
        return rule.get(field, "*") in ("*", value)

    candidates = []
    for rule in rules:
        if not matches(rule, "cnn", cnn_key) or not matches(rule, "risk", risk_key):
            continue
        if not matches(rule, "severity", severity_key):
            continue
        feature = rule.get("feature", "*")
        if feature != "*" and feature not in features:
            continue
        combination_specificity = sum(rule.get(field, "*") != "*" for field in ("cnn", "risk"))
        severity_specificity = int(rule.get("severity", "*") != "*")
        feature_specificity = int(feature in key_abnormals)
        candidates.append((combination_specificity, severity_specificity, feature_specificity, rule))

    if not candidates:
        return {
            "medical_advice": "请由医生结合波形、特征指标、症状和病史复核筛查结果。",
            "plain_language": "这是一份辅助筛查结果，建议把报告交给医生一起查看和解释。",
            "doctor_communication": "这份报告用于辅助筛查，建议结合您的症状和既往资料进行专业复核。",
            "follow_up": "请根据医生建议安排复查，复查时间和项目以医生评估为准。",
            "lifestyle": "保持规律作息、适度活动和均衡饮食，避免熬夜及过量咖啡因摄入。",
            "notice": notice,
            "matched_rule": "builtin_fallback"
        }
    _, _, _, selected = max(candidates, key=lambda item: (item[0], item[1], item[2]))
    suggestion = {field: selected.get(field, "") for field in ("medical_advice", "plain_language", "doctor_communication", "follow_up", "lifestyle")}
    suggestion["notice"] = selected.get("notice", notice)
    suggestion["matched_rule"] = selected.get("id", "configured_rule")
    return suggestion


def build_suggestion(risk_num, abn_level, key_abnormals):
    risk_map = {0: "低危", 1: "中危", 2: "高危"}

    if risk_num == 0:
        if abn_level == "无" and not key_abnormals:
            return "心电特征未见明显异常，建议保持健康作息，定期复查。"
        if abn_level == "无" and key_abnormals:
            if "QTc" in key_abnormals:
                return "总风险低，但QTc延长需关注，建议避免使用可能延长QT间期的药物，并复查心电图。"
            if "HR" in key_abnormals:
                return "总风险低，但存在心率异常，建议监测心率变化，保持规律作息。"
            return "总风险低，但部分指标异常，建议关注并定期复查。"
        if abn_level == "偶发":
            return "总体风险低，偶发异常心拍，建议减少熬夜与咖啡因摄入，定期复查。"
        if abn_level in ["频发", "显著"]:
            return "风险分级为低危，但异常心拍较多，建议进一步进行动态心电图检查。"
        return "总风险低，建议保持健康生活方式，定期复查。"

    if risk_num == 1:
        if abn_level == "无" and not key_abnormals:
            return "中危，建议近期前往心内科进一步评估。"
        if key_abnormals:
            if "ST_shift" in key_abnormals:
                return "中危且存在ST段改变，需警惕心肌缺血可能，建议尽快就医。"
            if "QRS" in key_abnormals:
                return "中危且QRS增宽，提示可能存在室内传导阻滞，建议近期就医。"
            if "HR" in key_abnormals:
                return "中危且心率异常，建议近期就医并监测心率。"
        if abn_level in ["频发", "显著"]:
            return "中危且异常心拍较多，建议尽快就医，避免剧烈运动。"
        return "中危，建议近期就医，并减少熬夜与高强度压力。"

    if risk_num == 2:
        if not key_abnormals and abn_level == "无":
            return "高危，建议立即就医，并避免剧烈运动。"
        if "HR" in key_abnormals:
            return "高危且心率异常，建议立即就医，避免情绪激动与剧烈活动。"
        if "ST_shift" in key_abnormals:
            return "高危且ST段显著改变，高度警惕急性心肌缺血，建议立即急诊。"
        if "QTc" in key_abnormals:
            return "高危且QTc显著延长，有发生恶性心律失常风险，建议立即就医。"
        if abn_level in ["频发", "显著"]:
            return "高危且异常心拍显著，提示心脏电活动不稳定，建议立即就医。"
        return "高危，建议立即就医，并保持安静休息。"

    return "建议进一步评估。"


def generate_report(risk_num, risk_score, risk_probs, features,
                    abnormal_count=0, total_beats=0, abnormal_types=None,
                    config=None, sex="", cnn_status="normal", review_status="", cnn_confidence=None, xgb_confidence=None):
    """生成双通路联合报告"""
>>>>>>> 5bdb321dece5bc57619bf68e89bf0244b0465ab4
    try:
        risk_level_map = {0: "低危", 1: "中危", 2: "高危"}
        risk_level = risk_level_map.get(int(risk_num), "未知")

        rules = load_suggestion_rules()
        feature_rules = rules.get("dynamic_feature_descriptions", {})
        detail_rows = []
        key_abnormals = []

        for name, value in features.items():
            if value is None:
                continue
            severity, display_text = judge_feature(name, value, DEFAULT_THRESHOLDS, sex=sex)
            status = {
                "正常": "正常",
                "轻度异常": "轻度异常",
                "显著异常": "显著异常",
                "未检测": "未检测",
            }.get(severity, severity)
            detail_rows.append(
                {
                    "feature": name,
                    "value": value,
                    "severity": status,
                    "display_text": display_text,
                }
            )
            if "异常" in severity:
                key_abnormals.append(name)

        if abnormal_count is not None and total_beats:
            ratio = abnormal_count / max(total_beats, 1)
            if ratio >= 0.2:
                abn_level = "显著"
            elif ratio >= 0.1:
                abn_level = "频发"
            elif ratio >= 0.02:
                abn_level = "偶发"
            else:
                abn_level = "无"
        else:
            abn_level = "无"

<<<<<<< HEAD
        suggestion = build_suggestion(int(risk_num), abn_level, key_abnormals, features, cnn_status=cnn_status, abnormal_count=abnormal_count, total_beats=total_beats, sex=sex)

        dynamic_descriptions = []
        for row in detail_rows:
            if "异常" not in row["severity"]:
                continue
            desc = build_dynamic_feature_description(row["feature"], row["value"], feature_rules)
            dynamic_descriptions.append(f"- {row['feature']}：{desc}")

        lines = []
        lines.append("心电筛查报告")
        lines.append("=" * 40)
        lines.append(f"风险分级：{risk_level}")
        lines.append(f"风险评分：{float(risk_score):.3f}")
        lines.append(f"低危概率：{float(risk_probs[0]):.2%} | 中危概率：{float(risk_probs[1]):.2%} | 高危概率：{float(risk_probs[2]):.2%}")
        lines.append(f"总心拍数：{int(total_beats)} | 异常心拍数：{int(abnormal_count)}")
        lines.append(f"异常水平：{abn_level}")
        lines.append("")
        lines.append("关键特征判读：")
        for row in detail_rows:
            lines.append(f"- {row['feature']}：{row['display_text']} ({row['severity']})")
        lines.append("")
        lines.append("特征增强描述：")
        if dynamic_descriptions:
            lines.extend(dynamic_descriptions)
        else:
            lines.append("- 未发现需特别提醒的异常特征。")
        lines.append("")
        lines.append("综合建议：")
        lines.append(suggestion)
=======
        suggestion = match_suggestion(cnn_status, risk_num, abn_level, key_abnormals, (config or {}).get("suggestions_path"))

        lines = []
        lines.append("心电风险筛查报告")
        lines.append(f"【风险结论】{risk_text}，异常程度：{abn_level}")
        lines.append(f"【风险评分】{risk_score:.2f}")
        if cnn_confidence is not None:
            lines.append(f"【CNN置信度】{cnn_confidence:.2f}")
        if xgb_confidence is not None:
            lines.append(f"【XGBoost置信度】{xgb_confidence:.2f}")
        if review_status:
            lines.append(f"【复核状态】{review_status}")
        lines.append(f"【风险概率】{risk_probs}")
        lines.append(f"【心拍分析】总心拍 {total_beats}，异常心拍 {abnormal_count}")
        if abnormal_types:
            type_str = "，".join([f"{k} {v}个" for k, v in abnormal_types.items()])
            lines.append(f"【异常类型】{type_str}")
        lines.append("【特征指标】")
        for name, info in feature_judgements.items():
            lines.append(f"{name}: {info['desc']}（{info['status']}）")
        if key_abnormals:
            lines.append("【重点异常】")
            for name in key_abnormals:
                lines.append(f"{name}: {feature_judgements[name]['desc']}")
        lines.append("【医生专业建议】" + suggestion["medical_advice"])
        lines.append("【通俗解释】" + suggestion["plain_language"])
        lines.append("【医生沟通话术】" + suggestion["doctor_communication"])
        lines.append("【复查建议】" + suggestion["follow_up"])
        lines.append("【生活方式】" + suggestion["lifestyle"])
        lines.append("【提示】" + suggestion["notice"])
        lines.append("【免责声明】本报告仅供辅助筛查参考，不能替代执业医师诊断。")
>>>>>>> 5bdb321dece5bc57619bf68e89bf0244b0465ab4

        report_text = "\n".join(lines)
        report_data = {
<<<<<<< HEAD
            "risk_num": int(risk_num),
            "risk_level": risk_level,
            "risk_score": float(risk_score),
            "risk_probs": [float(p) for p in risk_probs],
            "features": {name: float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else value for name, value in features.items()},
            "abnormal_count": int(abnormal_count),
            "total_beats": int(total_beats),
=======
            "risk_level": risk_text,
            "risk_num": risk_num,
            "risk_score": risk_score,
            "risk_probs": risk_probs,
            "cnn_status": cnn_status,
            "cnn_confidence": cnn_confidence,
            "xgb_confidence": xgb_confidence,
            "review_status": review_status,
>>>>>>> 5bdb321dece5bc57619bf68e89bf0244b0465ab4
            "abnormal_level": abn_level,
            "key_abnormals": key_abnormals,
            "detail_rows": detail_rows,
            "suggestion": suggestion,
            "sex": sex,
        }

        return "success", report_text, report_data
    except Exception as exc:
        logger.error(f"generate_report 失败：{exc}")
        return "error", f"报告生成失败：{exc}", {"risk_num": risk_num, "risk_score": risk_score, "risk_probs": list(risk_probs or [0.0, 0.0, 0.0]), "features": features, "abnormal_count": abnormal_count, "total_beats": total_beats, "sex": sex}
