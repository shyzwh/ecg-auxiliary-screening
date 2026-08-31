# 对读取到的心电信号做清洗。包括带通滤波、工频陷波、中值滤波、基线校正。

import numpy as np
from scipy import signal
# 相对引入
from src.logger import logger


def preprocess_ecg(ecg_signal, fs):
    """
    对心电信号进行标准化预处理。
    返回：clean_signal
    """
    if len(ecg_signal) < 10:
        return "error", None, "信号太短，无法预处理"

    try:
        # 1. 自动工频陷波
        notch_freq = _detect_notch_freq(ecg_signal, fs)
        if notch_freq is not None:
            ecg_signal = _notch_filter(ecg_signal, fs, notch_freq)

        # 2. 带通滤波 0.5-40Hz
        ecg_signal = _bandpass_filter(ecg_signal, fs, 0.5, 40.0)

        # 3. 中值滤波去尖峰
        ecg_signal = signal.medfilt(ecg_signal, kernel_size=3)

        # 4. 基线校正
        ecg_signal = _baseline_correction(ecg_signal)

        return "success", ecg_signal, "预处理完成"
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