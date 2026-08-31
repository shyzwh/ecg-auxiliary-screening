# 对读取到的心电信号做清洗。包括带通滤波、工频陷波、中值滤波、基线校正。

import numpy as np
from scipy import signal
# 相对引入
from src.logger import logger


def _sanitize_signal(ecg_signal):
    arr = np.asarray(ecg_signal, dtype=float).reshape(-1)
    if arr.size == 0:
        return arr, False
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if not np.isfinite(arr).all():
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if arr.size < 10 or np.max(np.abs(arr)) < 1e-8:
        return arr, False
    return arr, True


def preprocess_ecg(ecg_signal, fs):
    """
    对心电信号进行标准化预处理。
    返回：clean_signal
    """
    if ecg_signal is None:
        return "error", None, "信号为空，无法预处理"

    signal_array, valid = _sanitize_signal(ecg_signal)
    if not valid:
        return "error", None, "信号噪声过大或心跳过少，建议更换数据"

    try:
        notch_freq = _detect_notch_freq(signal_array, fs)
        if notch_freq is not None:
            signal_array = _notch_filter(signal_array, fs, notch_freq)

        signal_array = _bandpass_filter(signal_array, fs, 0.5, 40.0)
        signal_array = signal.medfilt(signal_array, kernel_size=3)
        signal_array = _baseline_correction(signal_array)
        signal_array = np.nan_to_num(signal_array, nan=0.0, posinf=0.0, neginf=0.0)

        if signal_array.size < 10:
            return "error", None, "信号太短，无法预处理"
        if np.max(np.abs(signal_array)) < 1e-8:
            return "error", None, "信号质量过差，建议更换数据"

        return "success", signal_array, "预处理完成"
    except Exception as e:
        logger.error(f"预处理失败：{e}")
        return "error", None, f"预处理失败：{e}"


def _detect_notch_freq(ecg_signal, fs):
    """自动判断50Hz还是60Hz工频干扰"""
    if len(ecg_signal) < fs:
        return None

    freqs = np.fft.fftfreq(len(ecg_signal), 1 / fs)
    fft_vals = np.abs(np.fft.fft(ecg_signal))

    def energy_around(freq):
        mask = (freqs > freq - 2) & (freqs < freq + 2)
        return np.sum(fft_vals[mask])

    e50 = energy_around(50)
    e60 = energy_around(60)

    if max(e50, e60) == 0:
        return None

    return 50 if e50 > e60 else 60


def _notch_filter(ecg_signal, fs, freq):
    """陷波滤波器"""
    b, a = signal.iirnotch(freq, 30, fs)
    return signal.filtfilt(b, a, ecg_signal)


def _bandpass_filter(ecg_signal, fs, low, high):
    """带通滤波器"""
    nyq = fs / 2.0
    b, a = signal.butter(2, [low / nyq, high / nyq], btype="band")
    return signal.filtfilt(b, a, ecg_signal)


def _baseline_correction(ecg_signal):
    """基线校正，去除漂移"""
    return ecg_signal - np.mean(ecg_signal)