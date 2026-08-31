# ===================== 导入依赖库 =====================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import shap
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                               confusion_matrix, classification_report)
# 设置中文显示，避免图表乱码
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams["axes.unicode_minus"] = False
# ===================== 1. 模拟论文12项心电标准化特征数据集 =====================
# 12项临床心电特征（与申请书完全对应）
feature_names = [
    "HR",        # 心率
    "PR",        # PR间期
    "QRS",       # QRS波时限
    "QT",        # QT间期
    "QTc",       # 校正QT间期
    "ST_shift",  # ST段偏移量
    "P_amp",     # P波振幅
    "T_amp",     # T波振幅
    "RR_mean",   # RR间期均值
    "RR_std",    # RR间期标准差
    "SDNN",      # HRV时域指标SDNN     "RMSSD"      # HRV时域指标RMSSD
]
np.random.seed(42)  # 固定随机种子，复现结果
n_sample = 3000      # 模拟3000份心电样本
# 生成12维标准化特征（模拟NeuroKit2提取后的归一化数据）
X_raw = np.random.randn(n_sample, len(feature_names))
# 构造三级风险标签：0=低危，1=中危，2=高危（分层分布）
y = np.random.choice([0,1,2], size=n_sample, p=[0.5, 0.35, 0.15])
# 转为DataFrame，贴合真实结构化临床数据
df = pd.DataFrame(X_raw, columns=feature_names)
df["risk_label"] = y
X = df[feature_names]
y = df["risk_label"]
# ===================== 2. 数据集划分+标准化 =====================
# 分层划分训练集80%、测试集20%，保证各类别比例均衡
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
# 标准化12项心电特征（论文预处理流程）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# 转回DataFrame方便SHAP绘图（保留特征名）
X_train_df = pd.DataFrame(X_train_scaled, columns=feature_names)
X_test_df = pd.DataFrame(X_test_scaled, columns=feature_names)
# ===================== 3. XGBoost多分类模型+网格搜索超参调优 =====================
# 初始化XGB三分类器，匹配论文三级风险分级
xgb_model = xgb.XGBClassifier(
    objective="multi:softmax",
    num_class=3,
    random_state=42,
    eval_metric="mlogloss"
)
# 网格搜索参数范围（论文使用GridSearchCV优化）
param_grid = {
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.1, 0.2],
    "n_estimators": [80, 120, 160],
    "subsample": [0.7, 0.8]
}
# 5折分层交叉验证，以宏平均F1为优化指标（论文核心评估指标）
cv_strat = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    scoring="f1_macro",
    cv=cv_strat,
    n_jobs=-1,
    verbose=1
)
# 网格搜索训练
print("===== 开始网格搜索超参数调优 =====")
grid_search.fit(X_train_scaled, y_train)
# 获取最优模型
best_xgb = grid_search.best_estimator_
print(f"\n最优超参数组合：{grid_search.best_params_}")
print(f"训练集5折交叉验证最优宏F1：{grid_search.best_score_:.4f}")
# ===================== 4. 模型测试集完整评估（论文指标：准确率、宏F1、敏感性、混淆矩阵） =====================
y_pred = best_xgb.predict(X_test_scaled)
# 核心评估指标
acc = accuracy_score(y_test, y_pred)
macro_prec = precision_score(y_test, y_pred, average="macro")
macro_recall = recall_score(y_test, y_pred, average="macro")
macro_f1 = f1_score(y_test, y_pred, average="macro")
# 高危类别(2)单独敏感性（论文重点：高危样本识别敏感性≥95%）
high_risk_recall = recall_score(y_test, y_pred, labels=[2], average=None)[0]
# 输出评估结果
print("\n===== 测试集模型性能评估 =====")
print(f"整体准确率 Accuracy: {acc:.4f}")
print(f"宏精确率 Macro-Precision: {macro_prec:.4f}")
print(f"宏召回率 Macro-Recall: {macro_recall:.4f}")
print(f"宏平均F1 Macro-F1: {macro_f1:.4f}")
print(f"高危(2类)样本识别敏感性: {high_risk_recall:.4f}")
print("\n分类报告：")
print(classification_report(y_test, y_pred, target_names=["低危(0)","中危(1)","高危(2)"]))
# 绘制混淆矩阵
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["低危","中危","高危"],
            yticklabels=["低危","中危","高危"])
plt.title("XGBoost心电风险分级混淆矩阵", fontsize=14)
plt.xlabel("模型预测风险等级")
plt.ylabel("真实风险等级")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300)
plt.close()
# ===================== 5. SHAP可解释分析（论文核心创新，TreeExplainer） =====================
print("\n===== 开始SHAP特征归因计算 =====")
# 初始化树模型解释器
explainer = shap.TreeExplainer(best_xgb)
# 计算测试集SHAP值（多分类返回三维数组：样本×特征×类别）
shap_values = explainer.shap_values(X_test_scaled)
# 选取高危类别(2)的SHAP值绘图，匹配论文SHAP散点图/决策力图
shap_high_risk = shap_values[2]
# 5.1 全局SHAP摘要散点图（对应论文图6）
plt.figure(figsize=(12,7))
shap.summary_plot(shap_high_risk, X_test_df, feature_names=feature_names, show=False)
plt.title("SHAP全局特征贡献散点图（高危风险分类）", fontsize=14)
plt.tight_layout()
plt.savefig("shap_summary.png", dpi=300)
plt.close()
# 5.2 全局特征重要性条形图
plt.figure(figsize=(10,6))
shap.summary_plot(shap_high_risk, X_test_df, plot_type="bar", feature_names=feature_names, show=False)
plt.title("SHAP特征重要性排序", fontsize=14) 
plt.tight_layout()
plt.savefig("shap_importance_bar.png", dpi=300)
plt.close()
# 5.3 单样本决策力图（对应论文图8，取第10个高危样本演示）
sample_idx = 10 
plt.figure(figsize=(10,6))
shap.decision_plot(
    explainer.expected_value[2],
    shap_high_risk[sample_idx],
    features=X_test_df.iloc[sample_idx],
    feature_names=feature_names,
    show=False
)
plt.title(f"单样本心电风险决策力图（样本{sample_idx}，真实标签：{y_test.iloc[sample_idx]}）", fontsize=14)
plt.tight_layout()
plt.savefig("shap_decision_plot.png", dpi=300)
plt.close()
# 5.4 特征交互依赖图（QRS与ST_shift交互，对应论文图7）
plt.figure(figsize=(10,6))
shap.dependence_plot(
   ind="QRS",
    shap_values=shap_high_risk,
    X=X_test_df,
    interaction_index="ST_shift",
    show=False
)
plt.title("QRS波宽与ST段偏移SHAP交互图", fontsize=14)
plt.tight_layout()
plt.savefig("shap_interaction_QRS_ST.png", dpi=300)
plt.close()
# ===================== 6. 模型&标准化器保存（系统集成模块使用） =====================
# 保存最优XGB模型
best_xgb.save_model("ecg_risk_xgb_model.json")
# 保存标准化器，推理阶段必须使用同一个
joblib.dump(scaler, "ecg_scaler.pkl")
print("\n模型已保存为 ecg_risk_xgb_model.json，标准化器已保存为 ecg_scaler.pkl，可用于Web系统推理")
# ===================== 【可选】推理测试：输入12维心电特征输出分级结果 =====================
def ecg_risk_predict(feature_list):
    model = xgb.XGBClassifier()
    model.load_model("ecg_risk_xgb_model.json")
    scaler_load = joblib.load("ecg_scaler.pkl")
    arr = np.array(feature_list).reshape(1, -1)
    arr_scaled = scaler_load.transform(arr)
    pred = model.predict(arr_scaled)[0]
    prob = model.predict_proba(arr_scaled)[0]
    label_dict = {0:"低危",1:"中危",2:"高危"}
    return label_dict[int(pred)], int(pred), prob
# 测试样例，12个特征严格按照顺序
test_input = [81, 0.17, 0.12, 0.39, 0.43, 0.07, 0.13, 0.24, 0.81, 0.14, 46, 30]
risk_text, risk_num, proba = ecg_risk_predict(test_input)
print("\n===== 单样本推理演示 =====")
print(f"输入12项心电特征：{test_input}")
print(f"预测风险等级：{risk_text}，数字标签：{risk_num}")
print(f"三类概率 → 低危:{proba[0]:.4f} 中危:{proba[1]:.4f} 高危:{proba[2]:.4f}")