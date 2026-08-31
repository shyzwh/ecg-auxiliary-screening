# 接收12项特征，优先用XGBoost模型推理。如果模型不存在，就用规则兜底。

import json
import os

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from .config_utils import DEFAULT_CONFIG
from .logger import logger


FEATURE_ORDER = [
    "HR", "PR", "QRS", "QT", "QTc", "ST_shift",
    "P_amp", "T_amp", "RR_mean", "RR_std", "SDNN", "RMSSD"
]


def load_risk_model(model_path):
    # 加载XGBoost模型和标准化器
    try:
        model_path = str(model_path).replace("\\", "/")
        if not os.path.exists(model_path):
            return "error", None, None, "模型文件不存在"

        model = xgb.XGBClassifier()
        model.load_model(model_path)

        scaler_path = model_path.replace("ecg_risk_xgb_model.json", "ecg_scaler.pkl")
        scaler_path = scaler_path.replace("\\", "/")
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
        else:
            scaler = None

        logger.info("XGBoost模型加载成功")
        return "success", model, scaler, "模型加载成功"
    except Exception as e:
        logger.error(f"模型加载失败：{e}")
        return "error", None, None, f"模型加载失败：{e}"


def rule_based_inference(features, config=None, sex=""):
    """规则兜底推理，不依赖模型，根据临床阈值判断风险"""
    try:
        config = config or {}
        hr = features.get("HR", 0)
        qrs = features.get("QRS", 0)
        qtc = features.get("QTc", 0)
        rmssd = features.get("RMSSD", 0)
        sdnn = features.get("SDNN", 0)
        qtc_thresholds = {
            "男": config.get("qtc_threshold_male", DEFAULT_CONFIG["qtc_threshold_male"]),
            "女": config.get("qtc_threshold_female", DEFAULT_CONFIG["qtc_threshold_female"]),
        }
        qtc_normal_high = float(qtc_thresholds.get(sex, config.get("qtc_threshold_default", DEFAULT_CONFIG["qtc_threshold_default"])))
        qtc_mild_high = float(config.get("qtc_threshold_mild_high", DEFAULT_CONFIG["qtc_threshold_mild_high"]))

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
        if qtc > qtc_mild_high:
            score += 0.25
        elif qtc > qtc_normal_high:
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


def predict_risk(features, model_path="models/ecg_risk_xgb_model.json", config=None, sex=""):
    # 输入12项特征，输出三级风险
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
    return rule_based_inference(features, config, sex)

# SHAP
def explain_with_shap(features, model_path="models/ecg_risk_xgb_model.json"):
    # 使用TreeExplainer生成特征贡献
    try:
        import shap

        model_path = str(model_path).replace("\\", "/")
        status, model, scaler, _ = load_risk_model(model_path)
        if status != "success":
            return "error", None, "模型不可用，无法进行SHAP解释"

        # 构造输入
        vals = [float(features[name]) for name in FEATURE_ORDER]
        arr = np.array([vals], dtype=float)
        if scaler is not None:
            arr = scaler.transform(arr)

        # 使用与风险推理一致的标准化输入
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


def explain_shap_visualizations(features, model_path="models/ecg_risk_xgb_model.json", background_path="training_data.csv"):
    """生成全局、交互和单样本SHAP绘图所需的可序列化数据。"""
    try:
        import shap

        status, model, scaler, _ = load_risk_model(model_path)
        if status != "success":
            return "error", None, "模型不可用，无法生成SHAP图表"

        values = np.array([[float(features[name]) for name in FEATURE_ORDER]], dtype=float)
        sample = scaler.transform(values) if scaler is not None else values
        prediction = int(model.predict(sample)[0])
        explainer = shap.TreeExplainer(model)
        sample_explanation = explainer(sample, check_additivity=False)
        sample_values = np.asarray(sample_explanation.values)
        if sample_values.ndim == 3:
            sample_contribution = sample_values[0, :, prediction]
            base_values = np.asarray(sample_explanation.base_values)
            base_value = float(base_values[0, prediction] if base_values.ndim == 2 else base_values[prediction])
        else:
            sample_contribution = sample_values[0]
            base_value = float(np.asarray(sample_explanation.base_values).reshape(-1)[0])

        result = {
            "target_class": prediction,
            "target_label": {0: "低危", 1: "中危", 2: "高危"}.get(prediction, str(prediction)),
            "base_value": base_value,
            "standardized_values": {name: float(sample[0, index]) for index, name in enumerate(FEATURE_ORDER)},
            "sample_contributions": {name: float(sample_contribution[index]) for index, name in enumerate(FEATURE_ORDER)},
            "global": None,
            "interaction": None,
            "decision": {
                "features": FEATURE_ORDER,
                "contributions": [float(value) for value in sample_contribution],
            },
        }

        if not os.path.exists(background_path):
            return "partial", result, "训练数据不存在，全局和交互图暂不可用"

        frame = pd.read_csv(background_path)
        if not set(FEATURE_ORDER).issubset(frame.columns):
            return "partial", result, "训练数据缺少完整特征列，全局和交互图暂不可用"
        background_values = frame[FEATURE_ORDER].apply(pd.to_numeric, errors="coerce").dropna().to_numpy(dtype=float)
        if len(background_values) < 2:
            return "partial", result, "训练数据样本不足，全局和交互图暂不可用"
        background = scaler.transform(background_values) if scaler is not None else background_values
        background_explanation = explainer(background, check_additivity=False)
        background_shap = np.asarray(background_explanation.values)
        if background_shap.ndim == 3:
            background_shap = background_shap[:, :, prediction]
        result["global"] = {
            "values": background_values.tolist(),
            "shap_values": background_shap.tolist(),
            "sample_count": int(len(background_values)),
        }
        qrs_index = FEATURE_ORDER.index("QRS")
        st_index = FEATURE_ORDER.index("ST_shift")
        result["interaction"] = {
            "feature": "QRS",
            "interaction_feature": "ST_shift",
            "x": background[:, qrs_index].tolist(),
            "y": background_shap[:, qrs_index].tolist(),
            "color": background[:, st_index].tolist(),
        }
        return "success", result, "SHAP可视化数据生成完成"
    except Exception as e:
        logger.error(f"SHAP可视化数据生成失败：{e}")
        return "error", None, f"SHAP可视化数据生成失败：{e}"