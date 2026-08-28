print("SHAP测试开始")
from .inference import explain_with_shap

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

status, shap_result, msg = explain_with_shap(features)
print(f"SHAP计算状态：{status}")
print(f"提示信息：{msg}")

if status == "success":
    print("\n各特征SHAP贡献：")
    for name, info in shap_result.items():
        print(f"{name}: 实际值={info['value']}, SHAP={info['shap_value']:.4f}")