# 部署指南

## 1. 项目状态

项目入口为 `app.py`，当前版本为 `v2.1.0`。应用使用 Streamlit 启动，默认监听本机 8501 端口。

当前架构为：

```text
app.py 页面层
    -> service/ 业务编排
    -> src/ 算法、模型、报告和 UI 组件
    -> database/ + storage/ecg.db 数据持久化
```

## 2. 环境要求

- Windows 或可运行 Python 的本地环境。
- Python 3.10+，当前项目使用过 Python 3.11 环境验证。
- 安装 `requirements.txt` 中的 Streamlit、NumPy、pandas、SciPy、WFDB、TensorFlow CPU、Keras、XGBoost、SHAP、Plotly、joblib、scikit-learn、ReportLab、requests 和 python-dotenv。
- TensorFlow、Keras、NumPy、XGBoost 和 SHAP 版本未在 requirements 中锁定，安装时可能有兼容性差异。

## 3. 本地安装

在项目根目录执行：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

若 PowerShell 禁止执行脚本，可先按本机策略激活虚拟环境，或直接使用虚拟环境中的 Python 和 Streamlit 可执行文件。

## 4. 启动应用

```powershell
streamlit run app.py
```

默认地址：

```text
http://localhost:8501
```

如果端口被占用：

```powershell
streamlit run app.py --server.port 8502
```

首次启动会加载 TensorFlow/CNN、XGBoost、SHAP 等依赖，启动时间可能较长。启动日志中的 TensorFlow 或 `use_container_width` 警告不等于应用启动失败，应以页面能否加载和是否出现 traceback 判断。

## 5. 必要文件

```text
app.py
config.json
requirements.txt
config/cases.json
config/suggestions.json
knowledge/ecg_diseases.json
models/cnn_model.h5
models/ecg_risk_xgb_model.json
training_data.csv
cnn_beats_real.npy
cnn_labels_real.npy
```

若使用标准化器，还应保留：

```text
models/ecg_scaler.pkl
```

MIT-BIH DAT 输入需要对应的 `.hea` 等文件。仓库中的 `uploads/`、`logs/`、`storage/waveforms/` 属于运行目录，不应当依赖其为空或拥有固定内容。

## 6. 配置说明

`config.json` 保存模型、数据、上传、报告和阈值等配置。`service/config_service.py` 会将相对路径按项目根目录归一化，并固定使用 `storage/ecg.db` 作为历史数据库。

系统设置页可修改：

- 中危和高危阈值。
- XGBoost、CNN 模型路径。
- 主题。
- 是否显示 SHAP。
- 自定义 GLM API 地址、模型和密钥。

## 7. GLM API 配置

GLM 是可选功能，不参与 ECG 风险等级的核心判定。密钥优先从 Streamlit Secrets 的 `ZHIPU_API_KEY` 读取，其次读取环境变量 `ZHIPU_API_KEY`。不要把密钥写入 Git 或普通页面文本。

当前模型 fallback 顺序：

```text
glm-4-flash-250414
glm-4-flash
glm-4.7-flash
```

429 或请求超时会尝试下一个模型，每个模型最多一次；全部失败时返回 `None`，页面保留离线报告或显示不可用提示。

## 8. 数据库与运行时目录

数据库文件：

```text
storage/ecg.db
```

`records` 表有 13 个字段：

```text
record_id, 时间, 文件名, 风险等级, 风险评分, 总心拍数,
异常心拍数, 备注, 报告, 特征, patient_name, patient_info, symptoms
```

数据库迁移脚本：

```powershell
python -m database.migrate_filename_column
```

该脚本用于把历史字段 `文 件名` 改为 `文件名`，已正确时会安全跳过。

## 9. 部署后检查

1. 打开 Streamlit 页面，确认六个页面都能导航。
2. 进入“心电分析”，确认患者信息和症状输入可用。
3. 使用 CSV/TXT/DAT 或示例数据进行一次分析。
4. 确认波形、R 峰、异常心拍、12 项特征、风险结果和报告显示。
5. 进入“历史记录”，确认 SQLite 记录可读、可删除和可导出。
6. 进入“病例教学”和“病情统计”，确认配置文件和数据库数据可读取。
7. 进入“系统设置”，确认模型路径、阈值和主题可读取。
8. 未配置 API 时，确认离线建议仍可使用；配置 API 时不要在日志中输出密钥。

## 10. 常见问题

### 模型文件不存在

检查 `models/` 是否包含模型文件，并确认 `config.json` 使用项目相对路径。

### DAT 文件无法读取

确认 `.dat` 与对应 `.hea` 文件位于数据目录，并检查 `data_dir` 配置。

### SHAP 图不可用

检查 XGBoost 模型、`training_data.csv` 和 12 项特征列。训练数据不足时，全局图或交互图可能只返回提示。

### GLM 调用失败

检查 `ZHIPU_API_KEY`、API 地址、网络和限流状态。失败时使用离线报告，不应把 GLM 失败当成 ECG 分析失败。

### Streamlit 页面很久没有加载

首次启动可能在加载 TensorFlow 和 CNN 模型。检查终端是否出现 traceback；如果只有依赖警告，等待页面完成初始化。

## 11. 部署边界

当前项目适合本地、科研和教学演示。SQLite 是本地文件存储，不提供多人并发、权限、备份和审计能力。未提供 Docker、正式云数据库或生产级身份认证配置。公开部署前必须增加访问控制、患者数据保护和可靠持久化。

## 12. 医学与数据免责声明

系统输出仅用于辅助分析，不能替代执业医师诊断或急救判断。当前未进行外部临床数据集验证，高危敏感性未能验证，心律失常细分类未能验证。不要上传未经授权的真实患者数据；出现胸痛、呼吸困难、晕厥或持续心悸时，应立即就医。