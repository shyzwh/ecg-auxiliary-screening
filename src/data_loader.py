# 统一入口：以后只调 load_ecg，它会自动判断文件类型。
# 完整错误处理：文件不存在、格式不支持、缺列，都不会崩。
# 日志记录：每次读取成功或失败，都会写进日志。
# 新增 .npy 支持：为以后扩展留了口子。

# 该模块负责读取 .dat / .csv / .txt / .npy 格式 ECG 数据，并统一返回 (status, signal, fs, msg)

import os
import numpy as np
import pandas as pd
import wfdb
# 相对引入
from src.logger import logger


def load_ecg(file_path, fs_input=None):
    """
    自动识别 ECG 文件类型并读取信号。

    参数:
        file_path: ECG 文件路径
        fs_input: 手动指定采样率，CSV/NPY 读取时可能使用

    返回:
        (status, signal, fs, msg)
    """
    # 先做文件存在性与扩展名校验，再分流到对应解析函数
    if not os.path.exists(file_path):
        return "error", None, None, "文件不存在，请检查路径"

    ext = os.path.splitext(file_path)[1].lower()

    try:
        if ext == ".dat":
            return _load_mitbih(file_path)
        elif ext in [".csv", ".txt"]:
            return _load_csv_signal(file_path, fs_input)
        elif ext == ".npy":
            return _load_npy_signal(file_path, fs_input)
        else:
            return "error", None, None, f"不支持的文件格式：{ext}"
    except Exception as e:
        logger.error(f"读取文件失败：{file_path}，错误：{e}")
        return "error", None, None, f"读取文件失败：{e}"


def _load_mitbih(file_path):
    """读取 MIT-BIH .dat 文件，需要同目录下存在对应 .hea 文件。"""
    # wfdb 会根据 .hea 文件解析通道和采样率，便于读取标准 MIT-BIH 数据
    record_name = os.path.splitext(os.fspath(file_path))[0]
    record = wfdb.rdrecord(record_name, channels=[0])
    signal = record.p_signal[:, 0]
    fs = record.fs
    logger.info(f"成功读取MIT-BIH记录：{record_name}，采样率{fs}Hz")
    return "success", signal, fs, "MIT-BIH读取成功"


def _load_csv_signal(file_path, fs_input=None):
    """读取 CSV/TXT 格式，自动识别时间列和电压列。"""
    # 兼容不同表头命名：时间可能是 time / sec / 秒，电压可能是 voltage / mv / 电压
    df = pd.read_csv(file_path, sep=None, engine="python")

    cols = [str(c).lower() for c in df.columns.tolist()]

    time_col = None
    voltage_col = None

    time_keywords = ["time", "t", "sec", "s", "时间", "秒"]
    voltage_keywords = ["voltage", "volt", "mv", "电压", "幅值", "amplitude"]

    for i, col in enumerate(cols):
        if any(k in col for k in time_keywords):
            time_col = df.columns[i]
            break

    for i, col in enumerate(cols):
        if any(k in col for k in voltage_keywords):
            voltage_col = df.columns[i]
            break

    if time_col is None and len(df.columns) >= 1:
        time_col = df.columns[0]
    if voltage_col is None and len(df.columns) >= 2:
        voltage_col = df.columns[1]

    if voltage_col is None:
        return "error", None, None, "未找到电压列"

    voltage = df[voltage_col].to_numpy(dtype=float)

    if time_col is not None:
        time = df[time_col].to_numpy(dtype=float)
        if len(time) >= 2:
            dt = time[1] - time[0]
            fs = 1.0 / dt
        else:
            fs = fs_input if fs_input else 360.0
    else:
        if fs_input is None:
            return "error", None, None, "未检测到时间列，请手动输入采样率"
        fs = fs_input

    logger.info(f"成功读取CSV：{file_path}，采样率{fs:.2f}Hz，点数{len(voltage)}")
    return "success", voltage, fs, "CSV读取成功"


def _load_npy_signal(file_path, fs_input=None):
    """读取.npy 格式信号，需在调用前提供采样率。"""
    # NPY 数据通常只有纯信号数组，没有时间轴，因此必须由上层显式传入 fs
    if fs_input is None:
        return "error", None, None, "读取.npy需要手动提供采样率"

    signal = np.load(file_path)
    fs = fs_input
    logger.info(f"成功读取NPY：{file_path}，采样率{fs}Hz，点数{len(signal)}")
    return "success", signal, fs, "NPY读取成功"