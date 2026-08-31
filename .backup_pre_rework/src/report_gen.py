from .logger import logger


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


def judge_feature(name, value, thresholds, sex=""):
    """判断单个特征是否异常，返回(状态, 描述)"""
    if value is None:
        return "未检测", "未检测"
    if value is None:
        value = 0

    t = thresholds.get(name, {})
    unit = t.get("unit", "")

    if name == "QTc":
        if sex == "男":
            normal_high = 440
        elif sex == "女":
            normal_high = 460
        else:
            normal_high = 450

        mild_high = 500

        if value < normal_high:
            return "正常", f"{value}{unit}"
        elif value < mild_high:
            return "轻度异常", f"延长 {value}{unit}"
        else:
            return "显著异常", f"显著延长 {value}{unit}"

    if name == "ST_shift":
        abs_val = abs(value)
        direction = "抬高" if value > 0 else "压低"
        if abs_val < t.get("abs_normal", 0.1):
            return "正常", f"{value}{unit}"
        elif abs_val < t.get("abs_mild", 0.2):
            return "轻度异常", f"{direction} {value}{unit}"
        else:
            return "显著异常", f"显著{direction} {value}{unit}"

    if name == "QRS":
        if value < t.get("high", 120):
            return "正常", f"{value}{unit}"
        elif value < t.get("mild_high", 150):
            return "轻度异常", f"增宽 {value}{unit}"
        else:
            return "显著异常", f"显著增宽 {value}{unit}"

    if name == "RR_std":
        if value < t.get("high", 100):
            return "正常", f"{value}{unit}"
        elif value < t.get("mild_high", 200):
            return "轻度异常", f"{value}{unit}"
        else:
            return "显著异常", f"{value}{unit}"

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


def classify_abnormality(abnormal_count, total_beats):
    if total_beats <= 0 or abnormal_count <= 0:
        return "无"

    ratio = abnormal_count / total_beats

    if abnormal_count > 20 or ratio > 0.10:
        return "显著"
    elif abnormal_count >= 6 or ratio > 0.03:
        return "频发"
    elif abnormal_count >= 1:
        return "偶发"

    return "无"


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
                    config=None, sex=""):
    """生成双通路联合报告"""
    try:
        thresholds = config if config else DEFAULT_THRESHOLDS

        risk_map = {0: "低危", 1: "中危", 2: "高危"}
        risk_text = risk_map.get(risk_num, "未知")

        abn_level = classify_abnormality(abnormal_count, total_beats)

        feature_judgements = {}
        key_abnormals = []
        for name, val in features.items():
            status, desc = judge_feature(name, val, thresholds, sex)
            feature_judgements[name] = {"value": val, "status": status, "desc": desc}
            if status == "显著异常":
                key_abnormals.append(name)

        if abnormal_types is None:
            abnormal_types = {}

        suggestion = build_suggestion(risk_num, abn_level, key_abnormals)

        lines = []
        lines.append("心电风险筛查报告")
        lines.append(f"【风险结论】{risk_text}，异常程度：{abn_level}")
        lines.append(f"【风险评分】{risk_score:.2f}")
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
        lines.append(f"【综合建议】{suggestion}")
        lines.append("【免责声明】本报告仅供辅助筛查参考，不能替代执业医师诊断。")

        report_text = "\n".join(lines)

        report_data = {
            "risk_level": risk_text,
            "risk_num": risk_num,
            "risk_score": risk_score,
            "risk_probs": risk_probs,
            "abnormal_level": abn_level,
            "abnormal_count": abnormal_count,
            "total_beats": total_beats,
            "abnormal_types": abnormal_types,
            "feature_judgements": feature_judgements,
            "key_abnormals": key_abnormals,
            "suggestion": suggestion,
        }

        logger.info("双通路联合报告生成成功")
        return "success", report_text, report_data
    except Exception as e:
        logger.error(f"报告生成失败：{e}")
        return "error", f"报告生成失败：{e}", None

print("这是新的report_gen文件")