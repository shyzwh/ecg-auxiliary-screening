# 12项特征模块


# 所有函数都返回统一格式：status, data, msg。
# 对R峰过少、信号太短都做了判断。

import numpy as np
from src.logger import logger


def _sanitize_signal(signal):
    arr = np.asarray(signal, dtype=float).reshape(-1)
    if arr.size == 0:
        return arr, False
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if np.max(np.abs(arr)) < 1e-8:
        return arr, False
    return arr, True


def pan_tompkins(ecg_signal, fs):
    """简化版Pan-Tompkins算法，检测R波位置"""
    signal_array, valid = _sanitize_signal(ecg_signal)
    if not valid or len(signal_array) < 30:
        return "error", None, "信号噪声过大或心跳过少，建议更换数据"

    try:
        diff = np.diff(signal_array)
        squared = diff ** 2

        window_size = int(0.12 * fs)
        integral = np.zeros_like(squared)
        for i in range(len(squared)):
            start = max(0, i - window_size)
            integral[i] = np.mean(squared[start:i + 1])

        integral = np.nan_to_num(integral, nan=0.0, posinf=0.0, neginf=0.0)
        if np.max(integral) <= 0:
            return "error", None, "信号噪声过大或心跳过少，建议更换数据"

        peak_threshold = 0.3 * np.max(integral)
        r_peaks = []
        refractory = int(0.2 * fs)
        last_peak = -refractory

        for idx in range(1, len(integral) - 1):
            if integral[idx] > integral[idx - 1] and integral[idx] > integral[idx + 1]:
                if integral[idx] > peak_threshold:
                    if idx - last_peak > refractory:
                        r_peaks.append(idx)
                        last_peak = idx

        r_peaks = [p + 1 for p in r_peaks]

        if len(r_peaks) < 3:
            return "error", None, "信号噪声过大或心跳过少，建议更换数据"

        logger.info(f"R峰检测完成，共{len(r_peaks)}个")
        return "success", r_peaks, "R峰检测成功"
    except Exception as e:
        logger.error(f"R峰检测失败：{e}")
        return "error", None, f"R峰检测失败：{e}"


def compute_hrv_features(r_peaks, fs):
    """根据R峰位置计算5项基础HRV特征"""
    if len(r_peaks) < 3:
        return "error", None, "R峰数量不足，无法计算HRV特征"

    try:
        rr_intervals = []
        for i in range(1, len(r_peaks)):
            interval = (r_peaks[i] - r_peaks[i - 1]) / fs * 1000.0
            rr_intervals.append(interval)

        rr = np.asarray(rr_intervals, dtype=float)
        rr = np.nan_to_num(rr, nan=0.0, posinf=0.0, neginf=0.0)
        if rr.size == 0 or not np.isfinite(rr).all() or np.any(rr <= 0):
            return "error", None, "信号噪声过大或心跳过少，建议更换数据"

        rr_mean = float(np.mean(rr))
        rr_std = float(np.std(rr))
        hr = 60000.0 / rr_mean
        sdnn = rr_std

        diff = np.diff(rr)
        rmssd = float(np.sqrt(np.mean(diff ** 2)))

        features = {
            "HR": round(hr, 2),
            "RR_mean": round(rr_mean, 2),
            "RR_std": round(rr_std, 2),
            "SDNN": round(sdnn, 2),
            "RMSSD": round(rmssd, 2),
        }

        logger.info(f"HRV特征计算完成：{features}")
        return "success", features, "HRV特征计算成功"
    except Exception as e:
        logger.error(f"HRV特征计算失败：{e}")
        return "error", None, f"HRV特征计算失败：{e}"


# 计算PR、QRS、QT、QTc:找到R峰后，再找Q波和S波，从而确定QRS起点和终点，再推算PR、QT。

def compute_wave_timing_features(ecg_signal, r_peaks, fs):
    """计算PR间期、QRS时限、QT间期、QTc"""
    if len(r_peaks) < 3:
        return "error", None, "R峰数量不足，无法计算时间特征"

    try:
        ecg_signal = np.asarray(ecg_signal, dtype=float).reshape(-1)
        ecg_signal = np.nan_to_num(ecg_signal, nan=0.0, posinf=0.0, neginf=0.0)
        pr_list = []
        qrs_list = []
        qt_list = []
        qtc_list = []

        for r in r_peaks[1:-1]:
            q_start = max(0, r - int(0.08 * fs))
            q_idx = q_start + np.argmin(ecg_signal[q_start:r]) if r > q_start else q_start
            s_end = min(len(ecg_signal), r + int(0.08 * fs))
            s_idx = r + np.argmin(ecg_signal[r:s_end]) if s_end > r else r
            t_end = min(len(ecg_signal), r + int(0.4 * fs))
            t_idx = r + np.argmax(ecg_signal[r:t_end]) if t_end > r else r

            qrs = (s_idx - q_idx) / fs * 1000.0
            qrs_list.append(qrs)

            qt = (t_idx - q_idx) / fs * 1000.0
            qt_list.append(qt)
            qtc = qt / np.sqrt((r_peaks[1] - r_peaks[0]) / fs) if (r_peaks[1] - r_peaks[0]) > 0 else 0.0
            qtc_list.append(qtc)

            p_start = max(0, r - int(0.22 * fs))
            p_end = r - int(0.06 * fs)
            if p_end > p_start:
                p_idx = p_start + np.argmax(ecg_signal[p_start:p_end])
                pr = (r - p_idx) / fs * 1000.0
                pr_list.append(pr)

        if not qrs_list:
            return "error", None, "无法计算波形时间特征"

        wave_features = {
            "PR": round(float(np.mean(pr_list)), 2) if pr_list else None,
            "QRS": round(float(np.mean(qrs_list)), 2),
            "QT": round(float(np.mean(qt_list)), 2),
            "QTc": round(float(np.mean(qtc_list)), 2),
        }

        if not all(np.isfinite(value) for value in wave_features.values() if value is not None):
            return "error", None, "信号噪声过大或心跳过少，建议更换数据"

        logger.info(f"波形时间特征计算完成：{wave_features}")
        return "success", wave_features, "波形时间特征计算成功"
    except Exception as e:
        logger.error(f"波形时间特征计算失败：{e}")
        return "error", None, f"波形时间特征计算失败：{e}"


# 计算ST段偏移、P波振幅、T波振幅    

def compute_wave_amp_features(ecg_signal, r_peaks, fs):
    """计算ST段偏移量、P波振幅、T波振幅"""
    if len(r_peaks) < 3:
        return "error", None, "R峰数量不足，无法计算振幅特征"

    try:
        ecg_signal = np.asarray(ecg_signal, dtype=float).reshape(-1)
        ecg_signal = np.nan_to_num(ecg_signal, nan=0.0, posinf=0.0, neginf=0.0)
        st_list = []
        p_amp_list = []
        t_amp_list = []

        for r in r_peaks[1:-1]:
            st_start = r + int(0.06 * fs)
            st_end = r + int(0.12 * fs)
            if st_end <= len(ecg_signal):
                st_val = np.mean(ecg_signal[st_start:st_end])
                st_list.append(st_val)

            p_start = max(0, r - int(0.2 * fs))
            p_end = r - int(0.08 * fs)
            if p_end > p_start:
                p_amp = np.max(ecg_signal[p_start:p_end])
                p_amp_list.append(p_amp)

            t_start = r + int(0.08 * fs)
            t_end = min(len(ecg_signal), r + int(0.35 * fs))
            if t_end > t_start:
                t_amp = np.max(ecg_signal[t_start:t_end])
                t_amp_list.append(t_amp)

        if not st_list:
            return "error", None, "无法计算振幅特征"

        amp_features = {
            "ST_shift": round(float(np.mean(st_list)), 3),
            "P_amp": round(float(np.mean(p_amp_list)), 3) if p_amp_list else None,
            "T_amp": round(float(np.mean(t_amp_list)), 3) if t_amp_list else None,
        }

        if not all(np.isfinite(value) for value in amp_features.values() if value is not None):
            return "error", None, "信号噪声过大或心跳过少，建议更换数据"

        logger.info(f"波形振幅特征计算完成：{amp_features}")
        return "success", amp_features, "波形振幅特征计算成功"
    except Exception as e:
        logger.error(f"波形振幅特征计算失败：{e}")
        return "error", None, f"波形振幅特征计算失败：{e}"


# 总入口函数

def extract_all_features(ecg_signal, r_peaks, fs):
    """合并所有12项特征"""
    status1, hrv, _ = compute_hrv_features(r_peaks, fs)
    status2, timing, _ = compute_wave_timing_features(ecg_signal, r_peaks, fs)
    status3, amp, _ = compute_wave_amp_features(ecg_signal, r_peaks, fs)

    if status1 != "success" or status2 != "success" or status3 != "success":
        default_features = {
            "HR": 0.0, "RR_mean": 0.0, "RR_std": 0.0, "SDNN": 0.0, "RMSSD": 0.0,
            "PR": 0.0, "QRS": 0.0, "QT": 0.0, "QTc": 0.0,
            "ST_shift": 0.0, "P_amp": 0.0, "T_amp": 0.0,
        }
        return "error", default_features, "信号噪声过大或心跳过少，建议更换数据"

    all_features = {}
    all_features.update(hrv)
    all_features.update(timing)
    all_features.update(amp)

    for key, value in list(all_features.items()):
        if value is None or not np.isfinite(float(value)):
            all_features[key] = 0.0

    logger.info(f"12项特征提取完成：{all_features}")
    return "success", all_features, "全部特征提取成功"