print("测试文件开始运行")

from .inference import predict_risk, load_risk_model

# 测试1：是否能加载模型文件
print("===== 测试模型加载 =====")
status, model, scaler, msg = load_risk_model("models/ecg_risk_xgb_model.json")
print(f"加载状态：{status}")
print(f"提示信息：{msg}")

# 测试2：用12项特征做一次推理
print("\n===== 测试规则推理 =====")
features = {
    "HR": 75.48,
    "PR": 177.98,
    "QRS": 54.29,
    "QT": 278.02,
    "QTc": 307.12,
    "ST_shift": -0.083,
    "P_amp": 0.108,
    "T_amp": 0.038,
    "RR_mean": 794.93,
    "RR_std": 53.05,
    "SDNN": 53.05,
    "RMSSD": 70.58,
}

status, risk_level, risk_num, score, msg = predict_risk(features)
print(f"推理状态：{status}")
print(f"风险等级：{risk_level}")
print(f"风险分数：{score}")
print(f"提示信息：{msg}")