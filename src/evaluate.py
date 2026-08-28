"""Evaluate the saved ECG models without modifying model artifacts."""

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow import keras

from .config_utils import DEFAULT_CONFIG, resolve_path
from .inference import FEATURE_ORDER


RISK_LABELS = ["low", "medium", "high"]


def _metric(value):
    return None if value is None else float(value)


def _recall_or_none(labels, predictions, label):
    support = int(np.sum(labels == label))
    if support == 0:
        return None
    return _metric(recall_score(labels, predictions, labels=[label], average=None, zero_division=0)[0])


def evaluate_xgboost(root):
    data_path = root / "training_data.csv"
    model_path = root / DEFAULT_CONFIG["model_path"]
    scaler_path = root / DEFAULT_CONFIG["scaler_path"]
    result = {
        "data_source": str(data_path),
        "label_rule": "training_data.csv 的 risk_label；脚本不重新生成标签",
        "split": "分层 80/20 留出，random_state=42；不是独立医生标注测试集",
        "status": "未能验证",
    }
    if not data_path.exists() or not model_path.exists():
        result["message"] = "training_data.csv 或 XGBoost 模型不存在"
        return result

    frame = pd.read_csv(data_path)
    if not set(FEATURE_ORDER + ["risk_label"]).issubset(frame.columns):
        result["message"] = "数据缺少 12 项特征列或 risk_label"
        return result
    frame = frame[FEATURE_ORDER + ["risk_label"]].apply(pd.to_numeric, errors="coerce").dropna()
    if frame["risk_label"].nunique() < 2:
        result["message"] = "风险标签类别不足"
        return result

    model = xgb.XGBClassifier()
    model.load_model(model_path)
    scaler = joblib.load(scaler_path) if scaler_path.exists() else None
    features = frame[FEATURE_ORDER].to_numpy(dtype=float)
    labels = frame["risk_label"].astype(int).to_numpy()
    train_x, test_x, train_y, test_y = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )
    test_x_model = scaler.transform(test_x) if scaler is not None else test_x
    start = time.perf_counter()
    predictions = model.predict(test_x_model).astype(int)
    probabilities = model.predict_proba(test_x_model)
    inference_seconds = time.perf_counter() - start
    matrix = confusion_matrix(test_y, predictions, labels=[0, 1, 2])
    recalls = [_recall_or_none(test_y, predictions, index) for index in range(3)]
    specificities = []
    for index in range(3):
        true_negative = matrix.sum() - matrix[index, :].sum() - matrix[:, index].sum() + matrix[index, index]
        false_positive = matrix[:, index].sum() - matrix[index, index]
        specificities.append(_metric(true_negative / (true_negative + false_positive)) if true_negative + false_positive else None)

    result.update({
        "status": "verified_on_available_holdout",
        "sample_count": int(len(frame)),
        "test_sample_count": int(len(test_y)),
        "classes": {str(index): RISK_LABELS[index] for index in range(3)},
        "accuracy": _metric(accuracy_score(test_y, predictions)),
        "macro_f1": _metric(f1_score(test_y, predictions, average="macro", zero_division=0)),
        "sensitivity_by_class": {RISK_LABELS[index]: recalls[index] for index in range(3)},
        "specificity_by_class": {RISK_LABELS[index]: specificities[index] for index in range(3)},
        "high_risk_sensitivity": recalls[2],
        "confusion_matrix": matrix.tolist(),
        "model_inference_seconds_test_batch": inference_seconds,
        "model_inference_ms_per_sample": inference_seconds / len(test_y) * 1000,
        "model_parameters": {
            "feature_count": len(FEATURE_ORDER),
            "boosted_tree_count": len(model.get_booster().get_dump()),
            "objective": str(model.get_params().get("objective")),
            "exact_parameter_count": "不适用于 XGBoost 树模型；已输出树数量和模型参数配置",
            "configured_parameters": {key: value for key, value in model.get_params().items() if value is not None and key not in {"callbacks", "feature_types"}},
        },
    })
    return result


def evaluate_cnn(root):
    data_path = root / DEFAULT_CONFIG["cnn_beats_real_path"]
    labels_path = root / DEFAULT_CONFIG["cnn_labels_real_path"]
    model_path = root / DEFAULT_CONFIG["cnn_model_path"]
    result = {
        "data_source": str(data_path),
        "label_source": str(labels_path),
        "label_rule": "cnn_labels_real.npy 的 0/1 二分类标签；未提供心律失常细分类标签",
        "split": "分层 80/20 留出，random_state=42；不是按患者或记录独立划分",
        "arrhythmia_type_sensitivity": "未能验证：标签没有心律失常类型",
        "status": "未能验证",
    }
    if not data_path.exists() or not labels_path.exists() or not model_path.exists():
        result["message"] = "CNN 数据、标签或模型不存在"
        return result

    beats = np.load(data_path).astype(np.float32)
    labels = np.load(labels_path).astype(int).reshape(-1)
    if len(beats) != len(labels) or len(np.unique(labels)) < 2:
        result["message"] = "CNN 数据和标签长度/类别不符合评估要求"
        return result
    train_x, test_x, train_y, test_y = train_test_split(
        beats, labels, test_size=0.2, random_state=42, stratify=labels
    )
    model = keras.models.load_model(model_path)
    test_model_x = test_x[..., np.newaxis]
    start = time.perf_counter()
    probabilities = model.predict(test_model_x, verbose=0).reshape(-1)
    inference_seconds = time.perf_counter() - start
    predictions = (probabilities > 0.5).astype(int)
    matrix = confusion_matrix(test_y, predictions, labels=[0, 1])
    abnormal_recall = recall_score(test_y, predictions, pos_label=1, zero_division=0)
    result.update({
        "status": "verified_on_available_holdout",
        "sample_count": int(len(labels)),
        "test_sample_count": int(len(test_y)),
        "class_distribution": {str(label): int(count) for label, count in zip(*np.unique(labels, return_counts=True))},
        "accuracy": _metric(accuracy_score(test_y, predictions)),
        "macro_f1": _metric(f1_score(test_y, predictions, average="macro", zero_division=0)),
        "abnormal_beat_detection_rate": _metric(abnormal_recall),
        "sensitivity_by_class": {"normal": _metric(recall_score(test_y, predictions, pos_label=0, zero_division=0)), "abnormal": _metric(abnormal_recall)},
        "confusion_matrix": matrix.tolist(),
        "model_inference_seconds_test_batch": inference_seconds,
        "model_inference_ms_per_sample": inference_seconds / len(test_y) * 1000,
        "model_parameters": {"trainable_and_non_trainable_count": int(model.count_params()), "input_shape": [None, 252, 1], "output_shape": [None, 1]},
    })
    return result


def build_report(xgb_result, cnn_result):
    lines = [
        "ECG模型统一评估报告",
        f"生成时间：{datetime.now(timezone.utc).isoformat()}",
        "",
        "一、评估范围与真实性说明",
        "本报告只评估当前目录中的预训练模型和现有数据，不修改模型，不补造标签，不将规划指标当作实测结果。",
        "XGBoost 使用 training_data.csv 中已有的 risk_label；CNN 使用 cnn_labels_real.npy 中已有的 0/1 标签。",
        "训练标签依赖规则或已有数据标签，未提供医生独立标注，因此结果不能证明临床性能。",
        "申报书要求的 7:2:1 划分、患者级独立测试、真实临床验证和各类型心律失常敏感性：未能验证。",
        "",
        "二、XGBoost 评估",
        f"状态：{xgb_result.get('status')}",
        f"准确率：{xgb_result.get('accuracy', '未能验证')}",
        f"宏平均 F1：{xgb_result.get('macro_f1', '未能验证')}",
        f"各类别敏感性：{xgb_result.get('sensitivity_by_class', '未能验证')}",
        f"各类别特异性：{xgb_result.get('specificity_by_class', '未能验证')}",
        f"高危样本敏感性：{xgb_result.get('high_risk_sensitivity', '未能验证')}",
        f"混淆矩阵（标签顺序 low/medium/high）：{xgb_result.get('confusion_matrix', '未能验证')}",
        f"模型参数量：{xgb_result.get('model_parameters', {}).get('exact_parameter_count', '未能验证')}",
        f"模型推理耗时：{xgb_result.get('model_inference_seconds_test_batch', '未能验证')} 秒/测试批次",
        "",
        "三、CNN 评估",
        f"状态：{cnn_result.get('status')}",
        f"心拍二分类准确率：{cnn_result.get('accuracy', '未能验证')}",
        f"异常心拍检出率：{cnn_result.get('abnormal_beat_detection_rate', '未能验证')}",
        f"各类型心律失常敏感性：{cnn_result.get('arrhythmia_type_sensitivity', '未能验证')}",
        f"各类别敏感性：{cnn_result.get('sensitivity_by_class', '未能验证')}",
        f"混淆矩阵（标签顺序 normal/abnormal）：{cnn_result.get('confusion_matrix', '未能验证')}",
        f"模型参数量：{cnn_result.get('model_parameters', {}).get('trainable_and_non_trainable_count', '未能验证')}",
        f"模型推理耗时：{cnn_result.get('model_inference_seconds_test_batch', '未能验证')} 秒/测试批次",
        "",
        "四、端到端耗时与限制",
        "真实端到端分析耗时：未能验证。现有数据没有与原始 ECG、预处理、R 峰检测、特征提取和双通路输出一一对应的独立测试样本；上方仅报告模型批量推理耗时。",
        "如需证明申报书的准确率、敏感性、漏诊率、误诊率和 30 秒指标，需要医生标注的独立测试集及完整端到端基准脚本。",
    ]
    return "\n".join(lines)


def main():
    root = Path(__file__).resolve().parent.parent
    results_dir = root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    xgb_result = evaluate_xgboost(root)
    cnn_result = evaluate_cnn(root)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_limitations": [
            "training_data.csv 的 risk_label 不是医生独立标注的证据",
            "CNN 标签只有 normal/abnormal，不能计算各类型心律失常敏感性",
            "当前没有 7:2:1 划分产生的固定独立测试集",
            "真实端到端耗时未能验证，仅提供模型批量推理耗时",
        ],
        "xgboost": xgb_result,
        "cnn": cnn_result,
    }
    json_path = results_dir / "model_evaluation.json"
    text_path = results_dir / "model_evaluation.txt"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    text_path.write_text(build_report(xgb_result, cnn_result), encoding="utf-8")
    print(f"JSON report: {json_path}")
    print(f"Text report: {text_path}")
    print(f"XGBoost status: {xgb_result.get('status')}")
    print(f"CNN status: {cnn_result.get('status')}")


if __name__ == "__main__":
    main()
