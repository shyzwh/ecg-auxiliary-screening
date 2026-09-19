import datetime
import os

import joblib
import numpy as np
import xgboost as xgb

from src.cnn_inference import predict_abnormal_beats
from src.data_loader import load_ecg
from src.feature_extract import extract_all_features, pan_tompkins
from src.inference import FEATURE_ORDER, explain_with_shap, predict_risk
from src.preprocess import preprocess_ecg
from src.report_gen import generate_report


def risk_probabilities(features, config, risk_num, score):
    """
    计算各风险类别概率，并在模型不可用时给出兜底值。

    参数:
        features: 12 项特征
        config: 配置字典
        risk_num: 当前风险等级编号
        score: 当前风险分值

    返回:
        [低危, 中危, 高危] 概率列表
    """
    try:
        model = xgb.XGBClassifier()
        model_path = str(config["model_path"]).replace("\\", "/")
        model.load_model(model_path)
        arr = np.array([[features[name] for name in FEATURE_ORDER]], dtype=float)
        scaler_path = str(config.get("scaler_path", "models/ecg_scaler.pkl")).replace("\\", "/")
        if os.path.exists(scaler_path):
            arr = joblib.load(scaler_path).transform(arr)
        probs = model.predict_proba(arr)[0]
        return [float(probs[i]) for i in range(len(probs))]
    except Exception:
        fallback = [0.0, 0.0, 0.0]
        fallback[int(risk_num)] = max(float(score), 0.0)
        return fallback


def run_analysis(file_path, file_name, config, patient_info=None, progress_callback=None, symptoms=""):
    """
    执行完整分析流程：读取、预处理、R 峰检测、特征提取、模型推理和报告生成。

    参数:
        file_path: ECG 文件路径
        file_name: 文件名
        config: 配置项
        patient_info: 患者信息
        progress_callback: 进度回调函数
        symptoms: 症状描述

    返回:
        (result, message)
    """
    def update(index):
        # 进度条按固定阶段推进，便于前台展示处理状态
        if progress_callback is not None:
            progress_callback(index)

    patient_info = patient_info or {}
    sex = patient_info.get("sex", config.get("default_sex", "未指定"))

    update(1)
    status, signal, fs, message = load_ecg(file_path)
    if status != "success":
        return None, message

    update(2)
    status, clean_signal, message = preprocess_ecg(signal, fs)
    if status != "success":
        return None, message

    update(3)
    status, r_peaks, message = pan_tompkins(clean_signal, fs)
    if status != "success":
        return None, message

    status, features, message = extract_all_features(clean_signal, r_peaks, fs)
    if status != "success":
        return None, message
    features = {key: 0.0 if value is None else value for key, value in features.items()}

    update(4)
    cnn_status, abnormal_positions, cnn_confidence, cnn_message = predict_abnormal_beats(
        clean_signal, r_peaks, fs, config["cnn_model_path"]
    )
    abnormal_positions = abnormal_positions or []

    update(5)
    xgb_status, risk_level, risk_num, score, risk_message = predict_risk(
        features, config["model_path"], config=config, sex=sex
    )
    if xgb_status != "success":
        return None, risk_message
    risk_probs = risk_probabilities(features, config, risk_num, score)

    update(6)
    shap_status, shap_result, shap_message = explain_with_shap(features, config["model_path"])

    update(7)
    report_status, report_text, report_data = generate_report(
        risk_num=risk_num,
        risk_score=score,
        risk_probs=risk_probs,
        features=features,
        abnormal_count=len(abnormal_positions),
        total_beats=len(r_peaks),
        sex=sex,
        cnn_status=cnn_status,
    )

    return {
        "file_name": file_name,
        "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "patient_info": patient_info,
        "symptoms": str(symptoms or "").strip(),
        "signal": clean_signal,
        "fs": fs,
        "r_peaks": r_peaks,
        "features": features,
        "abnormal_positions": abnormal_positions,
        "cnn_status": cnn_status,
        "cnn_confidence": cnn_confidence,
        "cnn_message": cnn_message,
        "risk_level": risk_level,
        "risk_num": risk_num,
        "score": score,
        "risk_probs": risk_probs,
        "shap_status": shap_status,
        "shap_result": shap_result,
        "shap_message": shap_message,
        "report_status": report_status,
        "report_text": report_text,
        "report_data": report_data,
    }, "分析完成"
