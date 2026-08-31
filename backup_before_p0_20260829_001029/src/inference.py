# 接收12项特征，优先用XGBoost模型推理。如果模型不存在，就用规则兜底，保证系统永远能出结果。

import os
import json
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
# 相对引入
from .logger import logger


FEATURE_ORDER = [
    "HR", "PR", "QRS", "QT", "QTc", "ST_shift",
    "P_amp", "T_amp", "RR_mean", "RR_std", "SDNN", "RMSSD"
]


def load_risk_model(model_path):
    """加载XGBoost模型和标准化器"""
    try:
        if not os.path.exists(model_path):
            return "error", None, None, "模型文件不存在"

        model = xgb.XGBClassifier()
        model.load_model(model_path)

        scaler_path = model_path.replace("ecg_risk_xgb_model.json", "ecg_scaler.pkl")
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
        else:
            scaler = None

        logger.info("XGBoost模型加载成功")
        return "success", model, scaler, "模型加载成功"
    except Exception as e:
        logger.error(f"模型加载失败：{e}")
        return "error", None, None, f"模型加载失败：{e}"


def rule_based_inference(features):
    """规则兜底推理，不依赖模型，根据临床阈值判断风险"""
    try:
        hr = features.get("HR", 0)
        qrs = features.get("QRS", 0)
        qtc = features.get("QTc", 0)
        rmssd = features.get("RMSSD", 0)
        sdnn = features.get("SDNN", 0)

        score = 0.0

        # 心率异常
        if hr > 100 or hr < 50:
            score += 0.25
        elif hr > 90 or hr < 55:
            score += 0.1

        # QRS过宽
        if qrs > 120:
            score += 0.25
        elif qrs > 100:
            score += 0.1

        # QTc延长
        if qtc > 500:
            score += 0.25
        elif qtc > 450:
            score += 0.1

        # HRV降低
        if rmssd < 20:
            score += 0.15
        if sdnn < 30:
            score += 0.1

        # 防止数据都是低危
        if score >= 0.35:
            risk_level = "高危"
            risk_num = 2
        elif score >= 0.15:
            risk_level = "中危"
            risk_num = 1
        else:
            risk_level = "低危"
            risk_num = 0

        logger.info(f"规则推理完成：risk={risk_level}, score={score:.2f}")
        return "success", risk_level, risk_num, score, "规则推理完成"
    except Exception as e:
        logger.error(f"规则推理失败：{e}")
        return "error", None, None, None, f"规则推理失败：{e}"


def predict_risk(features, model_path="models/ecg_risk_xgb_model.json"):
    """
    总推理入口。
    优先用XGBoost模型，模型不存在时用规则兜底。
    返回：status, risk_level, risk_num, score, msg
    """
    # 先尝试加载模型
    status, model, scaler, _ = load_risk_model(model_path)

    if status == "success":
        try:
            # 按顺序取出12项特征
            arr = np.array([[features[name] for name in FEATURE_ORDER]])
            if scaler is not None:
                arr = scaler.transform(arr)

            pred = model.predict(arr)[0]
            prob = model.predict_proba(arr)[0]
            risk_num = int(pred)
            risk_level = {0: "低危", 1: "中危", 2: "高危"}[risk_num]
            score = float(np.max(prob))

            logger.info(f"模型推理完成：risk={risk_level}, score={score:.4f}")
            return "success", risk_level, risk_num, score, "模型推理完成"
        except Exception as e:
            logger.error(f"模型推理失败，转规则兜底：{e}")

    # 模型不可用，走规则
    return rule_based_inference(features)

# SHAP
def explain_with_shap(features, model_path="models/ecg_risk_xgb_model.json"):
    """用SHAP解释XGBoost对当前样本的决策"""
    try:
        import shap

        status, model, scaler, _ = load_risk_model(model_path)
        if status != "success":
            return "error", None, "模型不可用，无法进行SHAP解释"

        # 构造输入
        vals = [float(features[name]) for name in FEATURE_ORDER]
        arr = np.array([vals], dtype=float)

        # 用原始模型预测
        pred = int(model.predict(arr)[0])

        # 使用TreeExplainer
        explainer = shap.TreeExplainer(model)

        # 对单样本计算SHAP
        shap_values = explainer(arr, check_additivity=False)

        # 取出预测类别对应结果
        shap_arr = np.array(shap_values.values)

        shap_sample = shap_arr[0, :, pred]

        shap_dict = {}
        for i, name in enumerate(FEATURE_ORDER):
            shap_dict[name] = {
                "value": float(vals[i]),
                "shap_value": float(shap_sample[i])
            }

        logger.info("SHAP解释计算完成")
        return "success", shap_dict, "SHAP解释计算完成"
    except Exception as e:
        logger.error(f"SHAP解释失败：{e}")
        return "error", None, f"SHAP解释失败：{e}"