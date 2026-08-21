import os
import json
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from .data_loader import load_ecg
from .preprocess import preprocess_ecg
from .feature_extract import pan_tompkins, extract_all_features
from .inference import rule_based_inference


FEATURE_ORDER = [
    "HR", "PR", "QRS", "QT", "QTc", "ST_shift",
    "P_amp", "T_amp", "RR_mean", "RR_std", "SDNN", "RMSSD"
]


def build_training_data(record_ids):
    """读取多条MIT-BIH记录，提取特征，并用规则打风险标签"""
    all_rows = []

    for rid in record_ids:
        file_path = f"D:/桌面/ECG-Auxiliary-Screening/data/mitbih/mit-bih-arrhythmia-database-1.0.0/{rid}.dat"

        status, signal, fs, msg = load_ecg(file_path)
        if status != "success":
            print(f"跳过{rid}：{msg}")
            continue

        status, clean_signal, msg = preprocess_ecg(signal, fs)
        if status != "success":
            print(f"跳过{rid}：{msg}")
            continue

        status, r_peaks, msg = pan_tompkins(clean_signal, fs)
        if status != "success":
            print(f"跳过{rid}：{msg}")
            continue

        status, features, msg = extract_all_features(clean_signal, r_peaks, fs)
        if status != "success":
            print(f"跳过{rid}：{msg}")
            continue

        # 规则打标签
        status_rule, risk_level, risk_num, score, _ = rule_based_inference(features)
        if status_rule != "success":
            continue

        row = {name: features[name] for name in FEATURE_ORDER}
        row["risk_label"] = risk_num
        all_rows.append(row)
        print(f"{rid} 处理完成：{risk_level}，score={score:.2f}")

    if not all_rows:
        raise ValueError("没有成功处理任何记录，请检查数据路径")

    df = pd.DataFrame(all_rows)
    return df


def main():
    # 选择几条MIT-BIH记录做训练演示
    record_ids = [
    "100", "101", "102", "103", "104", "105", "106", "107",
    "108", "109", "111", "112", "113", "114", "115", "116",
    "117", "118", "119", "121", "122", "123", "124", "200",
    "201", "202", "203", "205", "207", "208", "209", "210",
    "212", "213", "214", "215", "217", "219", "220", "221",
    "222", "223", "228", "230", "231", "232", "233", "234"
]

    print("开始构建训练数据...")
    df = build_training_data(record_ids)
    df.to_csv("training_data.csv", index=False)
    print(f"训练数据已保存，共{len(df)}条")

    X = df[FEATURE_ORDER]
    y = df["risk_label"]

    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    # 训练XGBoost
    model = xgb.XGBClassifier(
        objective="multi:softmax",
        num_class=3,
        random_state=42,
        eval_metric="mlogloss"
    )
    model.fit(X_train, y_train)

    # 评估
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")

    print("\n===== 模型评估 =====")
    print(f"准确率：{acc:.4f}")
    print(f"宏平均F1：{f1:.4f}")
    # 分类报告
    # print(classification_report(y_test, y_pred, target_names=["低危", "中危", "高危"]))
    print("风险等级分布：")
    print(pd.Series(y).value_counts())

    # 保存模型和标准化器
    os.makedirs("models", exist_ok=True)
    model.save_model("models/ecg_risk_xgb_model.json")
    joblib.dump(scaler, "models/ecg_scaler.pkl")

    print("\n模型已保存到 models/ecg_risk_xgb_model.json")
    print("标准化器已保存到 models/ecg_scaler.pkl")


if __name__ == "__main__":
    main()