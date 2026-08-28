# ECG-Auxiliary-Screening 开发文档

本文档面向项目开发者，介绍系统架构、核心模块、数据流、配置、模型训练、扩展开发和调试方法。

## 1. 项目架构

### 1.1 整体架构

```text
用户浏览器
    |
    v
Streamlit Web 界面
    |
    v
app.py 应用编排层
    |
    +--> 配置管理
    +--> 文件上传与数据加载
    +--> ECG 信号预处理
    +--> R 峰检测
    +--> ECG/HRV 特征提取
    |
    +--> 1D-CNN 心拍异常检测
    |
    +--> XGBoost 风险分级
    |        |
    |        v
    |      SHAP 可解释性分析
    |
    +--> 报告生成
    +--> 历史记录保存
    +--> 结果可视化与导出
```

### 1.2 各层职责

#### 用户界面层

由 `app.py` 实现，负责页面布局、导航、文件上传、分析触发、结果展示、历史记录、配置修改和文件下载。

#### 应用编排层

由 `app.py` 中的页面函数和 `run_analysis()` 负责组织底层模块调用、错误处理、会话状态保存和结果展示。

#### 数据访问层

由 `data_loader.py`、`annotation_loader.py` 和 `export_csv.py` 负责读取 CSV、TXT、DAT、NPY 文件，读取 MIT-BIH 注释，以及导出 CSV。

#### 信号处理层

由 `preprocess.py` 和 `feature_extract.py` 负责滤波、基线校正、R 峰检测和 ECG/HRV 特征计算。

#### 模型推理层

由 `cnn_inference.py` 和 `inference.py` 负责加载模型、构造模型输入、执行 CNN/XGBoost 推理及规则兜底。

#### 可解释性和报告层

由 `inference.py` 和 `report_gen.py` 负责 SHAP 分析、异常特征判断、风险摘要和报告生成。

#### 配置和基础设施层

由 `config_utils.py` 和 `logger.py` 负责配置加载、默认值补全、路径解析、配置保存和日志记录。

## 2. 模块详细说明

### 2.1 `src/data_loader.py`

负责读取 ECG 文件。

#### `load_ecg(file_path, fs_input=None)`

- 参数：`file_path` 输入路径；`fs_input` 可选采样率。
- 返回：`(status, signal, fs, message)`。
- 功能：按扩展名读取 `.dat`、`.csv`、`.txt` 和 `.npy`。

#### `_load_mitbih(file_path)`

- 参数：MIT-BIH `.dat` 文件路径。
- 返回：`(status, signal, fs, message)`。
- 功能：使用 `wfdb.rdrecord()` 读取第 0 导联，要求对应 `.hea` 文件。

#### `_load_csv_signal(file_path, fs_input=None)`

- 参数：文件路径和可选采样率。
- 返回：`(status, voltage, fs, message)`。
- 功能：自动识别分隔符、时间列和电压列；有时间列时计算采样率，无时间列时使用 `fs_input`。

#### `_load_npy_signal(file_path, fs_input=None)`

- 参数：NPY 路径和采样率。
- 返回：`(status, signal, fs, message)`。
- 功能：读取 NumPy 数组；必须提供采样率。

### 2.2 `src/preprocess.py`

负责 ECG 预处理。

#### `preprocess_ecg(ecg_signal, fs)`

- 参数：一维 ECG 信号和采样率。
- 返回：`(status, clean_signal, message)`。
- 功能：自动检测 50/60 Hz 工频，执行陷波、0.5 至 40 Hz 带通、3 点中值滤波和基线校正。

#### `_detect_notch_freq(ecg_signal, fs)`

- 参数：信号和采样率。
- 返回：`50`、`60` 或 `None`。
- 功能：通过 FFT 比较 50 Hz 和 60 Hz 附近能量。

#### `_notch_filter(ecg_signal, fs, freq)`

- 参数：信号、采样率和陷波频率。
- 返回：滤波后的信号。
- 功能：使用 `iirnotch()` 和 `filtfilt()` 进行零相位陷波。

#### `_bandpass_filter(ecg_signal, fs, low, high)`

- 参数：信号、采样率、低频截止值和高频截止值。
- 返回：滤波后的信号。
- 功能：使用二阶 Butterworth 带通滤波器。

#### `_baseline_correction(ecg_signal)`

- 参数：ECG 信号。
- 返回：`ecg_signal - np.mean(ecg_signal)`。
- 功能：去除平均基线偏移。

注意：`preprocess_ecg()` 当前使用固定的预处理参数，配置文件中的同名参数尚未完全接入。

### 2.3 `src/qrs_detect.py`

当前为空模块。实际 R 峰检测位于 `feature_extract.py` 的 `pan_tompkins()`。

### 2.4 `src/feature_extract.py`

负责 R 峰检测和 12 项特征提取。

#### `pan_tompkins(ecg_signal, fs)`

- 参数：预处理 ECG 信号和采样率。
- 返回：`(status, r_peaks, message)`。
- 功能：执行差分、平方、移动平均积分、阈值判断和不应期处理，返回 R 峰采样点索引。

#### `compute_hrv_features(r_peaks, fs)`

- 参数：R 峰位置和采样率。
- 返回：`(status, features, message)`。
- 功能：计算 `HR`、`RR_mean`、`RR_std`、`SDNN` 和 `RMSSD`。
- 单位：HR 为次/分钟，其余间期特征为毫秒。

#### `compute_wave_timing_features(ecg_signal, r_peaks, fs)`

- 参数：ECG 信号、R 峰位置和采样率。
- 返回：`(status, features, message)`。
- 功能：启发式计算 `PR`、`QRS`、`QT` 和 `QTc`。

#### `compute_wave_amp_features(ecg_signal, r_peaks, fs)`

- 参数：ECG 信号、R 峰位置和采样率。
- 返回：`(status, features, message)`。
- 功能：计算 `ST_shift`、`P_amp` 和 `T_amp`。

#### `extract_all_features(ecg_signal, r_peaks, fs)`

- 参数：ECG 信号、R 峰位置和采样率。
- 返回：`(status, features, message)`。
- 功能：合并所有特征，输出固定的 12 项特征：

```text
HR, PR, QRS, QT, QTc, ST_shift,
P_amp, T_amp, RR_mean, RR_std, SDNN, RMSSD
```

### 2.5 `src/beat_segmenter.py`

负责固定窗口心拍切分。

#### `segment_beats(ecg_signal, r_peaks, fs, before=0.25, after=0.45)`

- 参数：ECG 信号、R 峰位置、采样率、R 峰前秒数和 R 峰后秒数。
- 返回：`(status, beats, beat_positions, message)`。
- 功能：默认截取 R 峰前 0.25 秒和后 0.45 秒，边界越界的心拍会被跳过。

### 2.6 `src/annotation_loader.py`

负责读取 MIT-BIH `.atr` 注释。

#### `load_beat_annotations(record_path)`

- 参数：不含扩展名的记录基础路径。
- 返回：`(status, ann_positions, ann_symbols, message)`。
- 功能：调用 `wfdb.rdann(record_path, "atr")`。

#### `map_labels(symbols)`

- 参数：注释符号列表。
- 返回：整数标签列表。
- 功能：`N`、`L`、`R`、`e`、`j` 映射为 0，其余映射为 1。

### 2.7 `src/cnn_inference.py`

负责 CNN 异常心拍推理。

#### `predict_abnormal_beats(signal, r_peaks, fs, model_path="models/cnn_model.h5")`

- 参数：信号、R 峰位置、采样率和模型路径。
- 返回：`(status, abnormal_positions, message)`。
- 功能：切分心拍，调整为 `(samples, beat_length, 1)`，调用 Keras 模型，概率大于 0.5 时返回对应异常 R 峰位置。

### 2.8 `src/inference.py`

负责 XGBoost 推理、规则兜底和 SHAP 分析。

#### `FEATURE_ORDER`

固定特征顺序：

```python
[
    "HR", "PR", "QRS", "QT", "QTc", "ST_shift",
    "P_amp", "T_amp", "RR_mean", "RR_std", "SDNN", "RMSSD"
]
```

#### `load_risk_model(model_path)`

- 参数：XGBoost 模型路径。
- 返回：`(status, model, scaler, message)`。
- 功能：加载 XGBoost 模型和同目录的 `ecg_scaler.pkl`。

#### `rule_based_inference(features)`

- 参数：12 项特征字典。
- 返回：`(status, risk_level, risk_num, score, message)`。
- 功能：在模型不可用时根据 HR、QRS、QTc、RMSSD 和 SDNN 进行规则风险判断。
- 编码：`0` 低危，`1` 中危，`2` 高危。

#### `predict_risk(features, model_path=...)`

- 参数：特征字典和模型路径。
- 返回：风险等级、数字标签、评分、风险概率及状态信息。
- 功能：按固定顺序排列特征，使用标准化器后调用 XGBoost；失败时进入规则兜底。

#### `explain_with_shap(features, model_path=...)`

- 参数：特征字典和模型路径。
- 返回：`(status, shap_dict, message)`。
- 功能：使用 `shap.TreeExplainer` 返回各特征原始值及 SHAP 贡献。

### 2.9 `src/report_gen.py`

负责报告生成。

#### `judge_feature(name, value, thresholds, sex="")`

- 参数：特征名、特征值、阈值配置和性别。
- 返回：`(status, description)`。
- 功能：判断特征是否异常，支持 QTc 性别阈值。

#### `classify_abnormality(abnormal_count, total_beats)`

- 参数：异常心拍数和总心拍数。
- 返回：`无`、`偶发`、`频发` 或 `显著`。
- 功能：根据数量和比例划分异常程度。

#### `build_suggestion(risk_num, abn_level, key_abnormals)`

- 参数：风险标签、异常程度和重点异常列表。
- 返回：建议文本。
- 功能：生成辅助建议。

#### `generate_report(...)`

- 参数：风险标签、评分、概率、特征、异常心拍统计、异常类型、配置和性别。
- 返回：`(status, report_text, report_data)`。
- 功能：生成风险结论、特征摘要、异常说明、建议和免责声明。

### 2.10 `src/config_utils.py`

负责配置管理。

#### `resolve_path(path)`

- 参数：相对或绝对路径。
- 返回：解析后的路径。
- 功能：相对路径解析到项目根目录，支持 `~` 展开。

#### `resolve_config_paths(config)`

- 参数：配置字典。
- 返回：解析路径后的配置字典。
- 功能：统一解析模型、数据、上传、报告、日志和历史记录路径。

#### `load_config(config_path="config.json")`

- 参数：配置文件路径。
- 返回：配置字典。
- 功能：读取 JSON，缺失或损坏时使用默认配置补全。

#### `save_config(config, config_path="config.json")`

- 参数：配置字典和保存路径。
- 返回：`(status, message)`。
- 功能：保存配置，并尽量将项目内路径转换为相对路径。

### 2.11 `src/logger.py`

负责文件日志。

#### `Logger(log_dir="logs")`

- 参数：日志目录。
- 功能：创建目录并按日期生成日志文件。

#### `info(message)`、`warning(message)`、`error(message)`

- 参数：日志文本。
- 返回：无。
- 功能：记录不同级别的日志。

#### `_write(level, message)`

- 参数：日志级别和消息。
- 返回：无。
- 功能：按统一格式写入日志文件。

### 2.12 `src/export_csv.py`

#### `export_mitbih_to_csv(record_id, output_path=None, data_dir=None)`

- 参数：记录编号、可选输出路径和数据目录。
- 返回：无统一结构化返回值。
- 功能：将 MIT-BIH 记录前 30 秒导出为包含 `time` 和 `voltage` 两列的 CSV，默认输出到 `uploads/{record_id}_30s.csv`。

### 2.13 `src/local_data.py`

独立的数据可视化脚本，读取 MIT-BIH 记录 100、加载注释并使用 Matplotlib 绘制波形。该文件没有函数，执行或导入时会直接运行绘图逻辑。

### 2.14 `src/main.py`

独立的模拟 XGBoost/SHAP 实验脚本，不是 Streamlit 入口。负责模拟数据、训练、评估、绘图和模型保存。

### 2.15 `src/train.py`

负责 XGBoost 训练。

#### `build_training_data(record_ids, data_dir=None)`

- 参数：记录编号列表和数据目录。
- 返回：训练用 `DataFrame`。
- 功能：读取、预处理、检测 R 峰、提取 12 项特征，并通过规则生成 `risk_label`。

#### `main()`

- 功能：生成 `training_data.csv`，按 80/20 分层划分，标准化特征，训练三分类 XGBoost，并保存模型和标准化器。

### 2.16 `src/train_cnn.py`

#### `build_cnn_model(input_length)`

- 参数：单个心拍的采样点数。
- 返回：Keras 模型。
- 功能：构建包含多层 `Conv1D`、池化、全局平均池化、全连接层和 Sigmoid 输出的二分类模型。

#### `train_cnn(beats, labels, model_path="models/cnn_model.h5")`

- 参数：二维心拍数组、0/1 标签和保存路径。
- 返回：`(status, accuracy, message)`。
- 功能：将输入调整为三维张量，进行 80/20 分层训练，训练 20 个 epoch，每批 32 个样本，并保存 `.h5` 模型。

### 2.17 CNN 数据集构建模块

#### `src/build_cnn_dataset.py`

`build_cnn_dataset(record_ids, before=0.25, after=0.45, data_dir=None, beats_path=None, labels_path=None)` 读取 ECG、预处理、检测 R 峰并切分心拍，保存 `cnn_beats.npy` 和 `cnn_labels.npy`。当前实现会将心拍全部标记为正常。

#### `src/build_cnn_dataset_real.py`

`build_real_cnn_dataset(record_ids, data_dir=None, beats_path=None, labels_path=None)` 读取 MIT-BIH `.atr` 注释，根据最近注释生成正常/异常标签，并保存 `cnn_beats_real.npy` 和 `cnn_labels_real.npy`。

### 2.18 测试模块

- `test_full_pipeline.py`：测试数据读取、预处理、R 峰检测和特征提取。
- `test_inference.py`：测试模型加载和风险推理。
- `test_pan_tompkins.py`：测试模拟信号 R 峰检测。
- `test_real_ecg.py`：测试 MIT-BIH 真实 ECG 和 HRV。
- `test_shap.py`：测试 SHAP 解释。

当前部分测试脚本仍可能使用旧的函数返回值接口，调试时应以被测函数当前签名为准。

## 3. 数据流说明

```text
文件上传
  -> 保存到 uploads/
  -> load_ecg()
  -> 原始 ECG 信号和采样率
  -> preprocess_ecg()
  -> 清洗后的信号
  -> pan_tompkins()
  -> R 峰位置
  -> extract_all_features()
  -> 12 项 ECG/HRV 特征
  -> predict_risk()
  -> XGBoost 风险等级和概率
  -> explain_with_shap()
  -> XGBoost 特征贡献
  -> generate_report()
  -> 页面展示、历史记录和文件导出
```

CNN 分支从 R 峰位置并行展开：

```text
清洗后的信号 + R 峰
  -> segment_beats()
  -> 固定长度心拍
  -> CNN 模型
  -> 异常概率
  -> 异常心拍位置
```

完整流程由 `app.py` 的 `run_analysis()` 编排，结果保存在 `st.session_state["analysis_result"]` 中。CNN 和 XGBoost 结果并列展示，不执行自动融合。

## 4. 配置说明

配置文件为 `config.json`。相对路径由 `resolve_config_paths()` 解析到项目根目录。

| 配置项 | 默认值 | 含义和可选值 |
|---|---:|---|
| `risk_threshold_medium` | `0.4` | 中危阈值，通常为 0 到 1 的数值 |
| `risk_threshold_high` | `0.7` | 高危阈值，通常为 0 到 1 的数值，建议大于中危阈值 |
| `default_duration_sec` | `30` | 默认分析时长，单位秒；当前主流程未完全使用 |
| `model_path` | `models/ecg_risk_xgb_model.json` | XGBoost 模型路径 |
| `cnn_model_path` | `models/cnn_model.h5` | CNN 模型路径 |
| `scaler_path` | `models/ecg_scaler.pkl` | 特征标准化器路径 |
| `data_dir` | `data/mitbih/mit-bih-arrhythmia-database-1.0.0` | MIT-BIH 数据目录 |
| `cnn_beats_path` | `cnn_beats.npy` | 基础 CNN 心拍数据路径 |
| `cnn_labels_path` | `cnn_labels.npy` | 基础 CNN 标签路径 |
| `cnn_beats_real_path` | `cnn_beats_real.npy` | 真实注释 CNN 心拍数据路径 |
| `cnn_labels_real_path` | `cnn_labels_real.npy` | 真实注释 CNN 标签路径 |
| `storage_path` | `storage/records.json` | 历史记录 JSON 路径 |
| `upload_dir` | `uploads` | 上传文件目录 |
| `report_dir` | `reports` | 报告目录；当前主流程未直接写入 |
| `log_dir` | `logs` | 日志目录 |
| `notch_freq` | `auto` | `auto`、`50` 或 `60`；当前预处理仍自动检测 |
| `show_shap` | `true` | 是否显示 SHAP 图 |
| `show_r_peaks` | `true` | 是否显示 R 峰 |
| `bandpass_low` | `0.5` | 带通低频截止值，单位 Hz；当前预处理未完全读取 |
| `bandpass_high` | `40.0` | 带通高频截止值，单位 Hz；当前预处理未完全读取 |
| `medfilt_kernel` | `3` | 中值滤波核大小；当前预处理固定使用 3 |
| `theme` | `浅色` | `医疗蓝`、`浅色` 或 `深色` |
| `default_sex` | `未指定` | `未指定`、`男` 或 `女` |
| `qtc_threshold_male` | `440` | 男性 QTc 阈值，单位 ms |
| `qtc_threshold_female` | `460` | 女性 QTc 阈值，单位 ms |
| `qtc_threshold_default` | `450` | 未指定性别 QTc 阈值，单位 ms |
| `llm_enabled` | `false` | 是否启用大模型配置；当前流程不调用大模型 |
| `llm_provider` | `OpenAI 兼容接口` | `OpenAI 兼容接口` 或 `自定义服务` |
| `llm_api_key` | 空字符串 | API 密钥配置项；不得写入文档或提交真实值 |
| `llm_endpoint` | 空字符串 | 接口地址；不得写入文档中的真实内部地址 |
| `llm_model` | 空字符串 | 模型名称 |
| `_comment` | 配置说明文本 | 非业务配置字段 |

风险阈值、规则兜底阈值和报告特征阈值是不同层次的配置，不会自动同步。修改时应分别检查界面、`inference.py` 和 `report_gen.py`。

## 5. 模型说明

### 5.1 XGBoost

#### 输入

XGBoost 使用以下固定顺序的 12 项特征：

```text
HR, PR, QRS, QT, QTc, ST_shift,
P_amp, T_amp, RR_mean, RR_std, SDNN, RMSSD
```

输入过程为：

```text
特征字典
  -> FEATURE_ORDER 排序
  -> StandardScaler 标准化
  -> XGBoost 分类器
```

#### 输出

```text
0 = 低危
1 = 中危
2 = 高危
```

同时输出风险概率，最大类别概率作为风险评分。

#### 训练方式

`src/train.py` 的训练流程：

1. 读取 MIT-BIH 记录。
2. 预处理 ECG。
3. 检测 R 峰。
4. 提取 12 项特征。
5. 使用 `rule_based_inference()` 生成风险标签。
6. 按 80/20 进行分层划分。
7. 使用 `StandardScaler` 标准化。
8. 训练 XGBoost 三分类模型。
9. 计算 Accuracy 和 Macro-F1。
10. 保存模型和标准化器。

当前标签由规则生成，不是独立临床专家标注。

#### 保存格式

```text
models/ecg_risk_xgb_model.json
models/ecg_scaler.pkl
```

### 5.2 1D-CNN

#### 输入

以 R 峰为中心切分固定长度心拍，默认范围为：

```text
R 峰前 0.25 秒 + R 峰后 0.45 秒
```

模型输入形状：

```text
(samples, beat_length, 1)
```

#### 输出

输出单个心拍的 Sigmoid 异常概率：

```text
probability > 0.5 -> 异常
probability <= 0.5 -> 正常
```

#### 训练方式

模型结构包含三层 `Conv1D`、池化层、`GlobalAveragePooling1D`、全连接层和 Sigmoid 输出层。

训练配置：

```text
优化器：Adam
损失函数：binary_crossentropy
训练轮数：20
Batch Size：32
验证集比例：20%
评价指标：accuracy
```

#### 保存格式

```text
models/cnn_model.h5
```

## 6. 扩展开发指南

### 6.1 新增特征

以新增 `PR_variability` 为例：

1. 在 `src/feature_extract.py` 中实现特征计算函数。
2. 在 `extract_all_features()` 中合并该特征。
3. 在 `src/inference.py` 的 `FEATURE_ORDER` 中加入特征名。
4. 修改 `src/train.py`，重新生成训练数据。
5. 重新训练 XGBoost 模型和标准化器。
6. 如需报告判断，在 `report_gen.py` 增加阈值和展示逻辑。
7. 增加正常信号、噪声信号和边界条件测试。

新增模型特征后，旧 XGBoost 模型通常不能继续使用，必须重新训练。

### 6.2 新增页面

1. 在 `app.py` 中定义页面函数，例如：

```python
def diagnostics_page(config):
    ...
```

2. 将页面名称加入侧边栏导航。
3. 在 `main()` 的页面分发逻辑中调用该函数。
4. 尽量复用 `st.session_state["analysis_result"]`。
5. 页面函数只负责交互和展示，信号处理、模型推理等逻辑应放在 `src/` 中。

### 6.3 修改风险阈值

#### 界面阈值

修改 `config.json`：

```json
{
  "risk_threshold_medium": 0.4,
  "risk_threshold_high": 0.7
}
```

应满足：

```text
0 <= medium < high <= 1
```

#### 规则兜底阈值

修改 `src/inference.py` 的 `rule_based_inference()`。当前规则评分边界为：

```text
score >= 0.35 -> 高危
score >= 0.15 -> 中危
score <  0.15 -> 低危
```

#### 报告特征阈值

修改 `src/report_gen.py` 中的默认阈值和 `judge_feature()`。修改后应同步更新测试和文档。

## 7. 调试指南

### 7.1 应用无法启动

检查：

```text
src/
requirements.txt
models/
config.json
```

确认依赖已安装，并且从项目根目录执行：

```powershell
streamlit run app.py
```

### 7.2 找不到 `src`

使用：

```powershell
python -m src.test_full_pipeline
```

不要使用：

```powershell
python src/test_full_pipeline.py
```

### 7.3 MIT-BIH 读取失败

确认同一记录的以下文件同时存在：

```text
100.dat
100.hea
100.atr
```

同时检查 `data_dir`、记录编号和 `wfdb` 安装情况。

### 7.4 CSV/TXT 采样率错误

- 有时间列时，确认时间单位为秒且采样点有序。
- 无时间列时，必须提供 `fs_input`。
- 确认采样率大于 0。

### 7.5 R 峰过少

可能原因包括信号过短、噪声过大、振幅过低、采样率错误或当前简化算法参数不适合输入。建议依次绘制原始信号和预处理信号，检查采样率、积分窗口、阈值和不应期。

### 7.6 CNN 加载失败

确认以下文件存在：

```text
models/cnn_model.h5
```

并检查 TensorFlow、心拍长度以及输入形状 `(samples, beat_length, 1)` 是否匹配。

### 7.7 XGBoost 加载失败

确认以下文件同时存在且来自同一次训练：

```text
models/ecg_risk_xgb_model.json
models/ecg_scaler.pkl
```

同时检查特征数量、`FEATURE_ORDER` 和当前 XGBoost 版本。

### 7.8 SHAP 失败

检查 SHAP、NumPy 和 XGBoost 版本兼容性，并确认输入特征顺序正确。修改解释逻辑时，应确保 SHAP 输入空间与模型训练输入空间一致。

### 7.9 历史记录保存失败

检查：

```text
storage/records.json
```

确认目录存在、JSON 格式合法、当前用户有写权限，且没有其他进程同时写入文件。

### 7.10 配置不生效

确认配置项是否：

1. 被 `app.py` 读取。
2. 被传入实际业务函数。
3. 被默认配置覆盖。
4. 需要重启 Streamlit 才能生效。

当前已知未完全接入的配置包括 `notch_freq`、`bandpass_low`、`bandpass_high` 和 `medfilt_kernel`。

### 7.11 测试接口不匹配

当前 `pan_tompkins()` 返回：

```python
(status, r_peaks, message)
```

如果测试脚本将函数结果直接当作 R 峰列表使用，应先解包返回值，再继续处理。

## 8. 代码规范

### 8.1 相对导入

`src` 内部模块统一使用相对导入：

```python
from .logger import logger
from .feature_extract import pan_tompkins
```

运行 `src` 模块时使用：

```powershell
python -m src.module_name
```

### 8.2 命名规范

- 函数和变量使用 `snake_case`。
- 常量使用 `UPPER_SNAKE_CASE`。
- 类名使用 `PascalCase`。
- 布尔变量优先使用 `is_`、`has_`、`show_` 或 `enable_` 前缀。
- 模块文件使用小写下划线命名。

### 8.3 返回值规范

底层业务函数优先使用：

```python
(status, result, message)
```

多个结果可以使用：

```python
(status, result_a, result_b, message)
```

调用方必须检查 `status`，不要忽略错误状态。

### 8.4 错误处理

- 文件读取、滤波、模型加载和推理应捕获可预期异常。
- 记录详细日志。
- 向界面返回可理解的错误信息。
- 不要吞掉异常。
- 不要在日志或文档中记录 API 密钥、密码、令牌或其他敏感信息。

### 8.5 特征和模型一致性

修改特征时必须同步检查：

```text
feature_extract.py
inference.py
train.py
report_gen.py
app.py
测试脚本
```

训练和推理必须使用完全一致的特征名称、数量、顺序和预处理方式。

### 8.6 路径和配置

- 项目内部路径优先使用相对路径。
- 统一通过 `resolve_path()` 解析路径。
- 不要在业务代码中硬编码本机绝对路径。
- 配置变更后同步更新默认值、界面校验、实际业务调用和文档。

## 9. 常用开发命令

```powershell
pip install -r requirements.txt
streamlit run app.py
python -m src.test_full_pipeline
python -m src.test_inference
python -m src.test_shap
python -m src.train
```

临时公网演示：

```powershell
cloudflared.exe tunnel --url http://localhost:8501
```

## 10. 当前技术限制

- `src/qrs_detect.py` 当前为空，R 峰检测位于 `feature_extract.py`。
- CNN 与 XGBoost 没有自动融合。
- 基础 CNN 数据集构建脚本会将心拍全部标记为正常。
- XGBoost 训练标签由规则生成，不是独立临床标注。
- 部分预处理配置尚未真正传入预处理函数。
- SHAP 输入标准化流程需要进一步统一。
- 历史记录不保存完整波形和 SHAP 图。
- 大模型配置目前没有实际调用逻辑。
- 部分测试脚本与当前函数返回值接口不一致。
- 依赖没有固定版本，可能存在环境兼容性问题。

## 11. 数据安全和医疗用途说明

开发和部署时不要将真实患者数据、身份信息、API 密钥、密码或令牌提交到代码仓库，也不要在日志中记录完整 ECG 或其他敏感健康信息。公网部署前应增加访问控制、文件校验和数据保护措施。

本系统仅用于科研、教学和心电辅助筛查，不能替代执业医师的专业诊断。所有分析结果仅供参考，应由专业医护人员结合原始心电波形、临床症状、病史及其他检查结果综合判断。如出现胸痛、晕厥、呼吸困难、持续心悸等不适，请及时就医或寻求急诊帮助。
