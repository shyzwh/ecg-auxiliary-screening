# ECG-Auxiliary-Screening

基于深度学习异常定位与可解释 XGBoost 的心电风险辅助筛查系统。

## 项目简介

本项目是一个基于 Streamlit 的单导联 ECG 风险辅助筛查 Web 工具，面向科研、教学和初步筛查场景。系统对上传的心电数据进行预处理、R 峰检测、ECG/HRV 特征提取，并通过 1D-CNN 定位异常心拍、通过 XGBoost 进行整体风险分级，再使用 SHAP 提供特征贡献解释，最后生成结构化文本报告。

系统结果仅供辅助分析，不能替代执业医师的专业诊断。

## 核心功能

- 支持 CSV、TXT、DAT 格式 ECG 文件读取；底层同时支持 NPY 数据。
- 支持 MIT-BIH Arrhythmia Database 记录读取。
- 自动检测并抑制 50/60 Hz 工频干扰。
- 进行 0.5 至 40 Hz 带通滤波、中值滤波和基线校正。
- 使用简化 Pan-Tompkins 算法检测 R 峰。
- 提取 HR、PR、QRS、QT、QTc、ST_shift、P_amp、T_amp、RR_mean、RR_std、SDNN、RMSSD 共 12 项特征。
- 使用 1D-CNN 逐心拍检测并定位可能异常的心拍。
- 使用 XGBoost 输出低危、中危、高危风险等级和概率。
- 使用 SHAP 解释 XGBoost 特征贡献。
- 生成风险摘要、异常特征和辅助建议报告。
- 支持历史记录保存、查看和 CSV/TXT 导出。
- 提供模型路径、风险阈值、主题及显示选项配置。

CNN 和 XGBoost 是两条独立通路：CNN 负责异常心拍定位，XGBoost 负责整体风险分级，当前系统不自动融合两者结果。

## 技术架构

```text
Streamlit Web 界面
        |
        v
app.py 应用编排
        |
        +--> 数据读取与格式识别
        +--> ECG 预处理
        +--> R 峰检测与特征提取
        |       |
        |       +--> 1D-CNN：异常心拍定位
        |       +--> XGBoost：整体风险分级
        |                         |
        |                         +--> SHAP 解释
        |
        +--> 报告生成、历史记录和结果导出
```

核心模块位于 `src/`：

- `data_loader.py`：读取 ECG 文件。
- `preprocess.py`：信号滤波和基线校正。
- `feature_extract.py`：R 峰检测和特征提取。
- `beat_segmenter.py`：固定窗口心拍切分。
- `cnn_inference.py`：CNN 异常心拍推理。
- `inference.py`：XGBoost、规则兜底和 SHAP。
- `report_gen.py`：报告生成。
- `config_utils.py`：配置和路径管理。

## 项目结构

```text
ECG-Auxiliary-Screening/
├── .streamlit/config.toml
├── .vscode/settings.json
├── app.py
├── config.json
├── requirements.txt
├── cloudflared.exe
├── training_data.csv
├── cnn_beats.npy
├── cnn_labels.npy
├── cnn_beats_real.npy
├── cnn_labels_real.npy
├── data/
│   ├── mitbih/mit-bih-arrhythmia-database-1.0.0/
│   └── processed/
├── docs/开发日志.md
├── logs/
├── models/
│   ├── cnn_model.h5
│   ├── ecg_risk_xgb_model.json
│   └── ecg_scaler.pkl
├── reports/
├── results/
├── src/
│   ├── annotation_loader.py
│   ├── beat_segmenter.py
│   ├── build_cnn_dataset.py
│   ├── build_cnn_dataset_real.py
│   ├── cnn_inference.py
│   ├── config_utils.py
│   ├── data_loader.py
│   ├── export_csv.py
│   ├── feature_extract.py
│   ├── inference.py
│   ├── local_data.py
│   ├── logger.py
│   ├── main.py
│   ├── preprocess.py
│   ├── qrs_detect.py
│   ├── report_gen.py
│   ├── train.py
│   ├── train_cnn.py
│   └── test_*.py
├── static/
├── storage/records.json
└── uploads/
```

## 快速开始

### 环境要求

建议使用 Python 3.10 或更高版本。`requirements.txt` 未固定依赖版本，TensorFlow、NumPy、XGBoost 和 SHAP 之间可能存在版本兼容要求。

### Windows 安装

```powershell
cd "D:\桌面\王涵—AI 护心镜—基于深度学习异常定位与可解释XGBoost 的心电风险辅助筛查系统\ECG-Auxiliary-Screening"

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

也可以使用 Conda：

```powershell
conda create -n ecg python=3.10 -y
conda activate ecg
pip install -r requirements.txt
```

### 启动应用

必须在项目根目录执行：

```powershell
streamlit run app.py
```

启动后访问：

```text
http://localhost:8501
```

### 运行测试

```powershell
python -m src.test_full_pipeline
python -m src.test_inference
python -m src.test_shap
```

`src` 中的模块使用相对导入，因此应使用 `python -m src.xxx`，不要直接运行 `python src/xxx.py`。

## 使用说明

1. 启动 Streamlit 应用。
2. 打开“心电分析”页面。
3. 上传 CSV、TXT 或 DAT 文件，或加载示例数据。
4. 点击“开始分析”。
5. 系统执行数据读取、预处理、R 峰检测、特征提取、CNN 推理、XGBoost 分级、SHAP 分析和报告生成。
6. 查看 ECG 波形、R 峰、异常心拍、风险等级、风险概率、12 项特征和 SHAP 结果。
7. 保存分析记录，或下载 TXT 报告和 CSV 数据。
8. 在“历史记录”页面查看已保存结果，在“系统设置”页面修改配置。

MIT-BIH `.dat` 文件需要对应的 `.hea` 文件；单独上传 `.dat` 文件通常不足以读取完整记录。

## 技术栈

技术栈严格对应 `requirements.txt`：

| 依赖 | 用途 |
|---|---|
| `streamlit` | Web 应用界面 |
| `numpy` | 数值计算和数组处理 |
| `pandas` | 数据读取和表格处理 |
| `matplotlib` | 本地测试绘图 |
| `scipy` | 信号滤波和科学计算 |
| `wfdb` | MIT-BIH 数据读取 |
| `neurokit2` | ECG 相关工具依赖 |
| `xgboost` | 风险分级模型 |
| `shap` | 模型可解释性分析 |
| `tensorflow` | 1D-CNN 模型 |
| `plotly` | Web 交互式可视化 |
| `joblib` | 标准化器持久化 |
| `scikit-learn` | 数据划分、标准化和评估 |

## 数据集

项目主要使用 MIT-BIH Arrhythmia Database：

- 记录通常为 360 Hz 采样。
- 主要使用单导联 MLII 信号。
- 记录由 `.dat`、`.hea` 和 `.atr` 等文件组成。
- `build_cnn_dataset_real.py` 可使用 MIT-BIH 注释生成正常/异常二分类心拍数据。
- `training_data.csv` 用于 XGBoost 训练。

需要注意：XGBoost 训练脚本通过规则生成风险标签，并非独立临床专家标注；PTB-XL 仅属于后续扩展方向，当前项目目录未发现实际 PTB-XL 数据。

## 模型说明

### XGBoost

输入为 12 项固定顺序的 ECG/HRV 特征，输出 0、1、2 三类风险标签：

```text
0 = 低危
1 = 中危
2 = 高危
```

训练流程包括特征提取、规则标签生成、80/20 分层划分、`StandardScaler` 标准化和 XGBoost 多分类训练。模型保存为：

```text
models/ecg_risk_xgb_model.json
models/ecg_scaler.pkl
```

### 1D-CNN

CNN 输入为以 R 峰为中心切分的固定长度心拍，默认使用 R 峰前 0.25 秒和 R 峰后 0.45 秒。模型输出 0 到 1 的异常概率，概率大于 0.5 时判为异常。

模型结构包含多层 `Conv1D`、`MaxPooling1D`、`GlobalAveragePooling1D`、全连接层和 Sigmoid 输出层。模型保存为：

```text
models/cnn_model.h5
```

## 部署方式

### 本地部署

```powershell
streamlit run app.py
```

### 临时公网演示

项目包含 `cloudflared.exe`，可在 Streamlit 启动后执行：

```powershell
cloudflared.exe tunnel --url http://localhost:8501
```

该方式生成临时公网地址，不适合生产环境。当前项目没有 Docker、数据库或正式云部署配置。

## 已知问题

- 当前完整项目位于同名中文路径副本；仅包含 `app.py` 的目录不是完整可运行项目。
- MIT-BIH `.dat` 文件需要配套 `.hea` 文件。
- 上传控件不开放 `.npy`，但底层加载器支持 `.npy`。
- `preprocess_ecg()` 当前仍固定使用自动陷波、0.5 至 40 Hz 带通和 3 点中值滤波，配置文件中的对应参数尚未完全生效。
- XGBoost 风险阈值主要用于界面显示；模型分类由模型输出，规则兜底使用独立阈值。
- CNN 和 XGBoost 不自动融合。
- 历史记录保存文本、特征和统计信息，不保存完整波形及 SHAP 图。
- 大模型字段仅用于配置预留，当前分析流程不会调用大模型。
- 部分测试脚本仍按旧的函数返回值接口编写，可能需要调整。
- 依赖未固定版本，安装时可能出现兼容性问题。

## 开发注意事项

- 始终从项目根目录启动应用。
- `src` 内部使用相对导入，模块测试使用 `python -m src.xxx`。
- 修改特征后，必须同步修改 `FEATURE_ORDER`、训练脚本、模型和测试。
- 修改模型输入特征后，旧模型通常不能继续使用，需要重新训练。
- 模型文件、标准化器和特征顺序必须来自同一套训练流程。
- 不要将真实患者数据、API 密钥、密码或令牌提交到仓库。
- 日志中不应记录患者身份信息或完整健康数据。
- 公网演示前应增加访问控制和数据保护措施。

## 免责声明

本系统为心电辅助筛查工具，仅用于科研、教育和辅助分析，不替代执业医师的专业诊断。所有分析结果仅供参考，应由专业医护人员结合原始心电波形、临床症状、病史及其他检查结果综合判断。如出现胸痛、晕厥、呼吸困难、持续心悸等不适，请及时就医或寻求急诊帮助。
