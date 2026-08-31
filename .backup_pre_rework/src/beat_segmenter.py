# 根据R峰位置，把心电信号切成一个个固定长度的心拍。
import numpy as np
from .logger import logger


def segment_beats(ecg_signal, r_peaks, fs, before=0.25, after=0.45):
    """
    根据R峰位置切分心拍。
    默认以R峰前0.25秒、后0.45秒为一个心拍。
    """
    beat_list = []
    beat_positions = []

    pre = int(before * fs)
    post = int(after * fs)

    for r in r_peaks:
        start = r - pre
        end = r + post

        if start < 0 or end > len(ecg_signal):
            continue

        beat = ecg_signal[start:end]
        beat_list.append(beat)
        beat_positions.append(r)

    if not beat_list:
        return "error", None, None, "没有成功切分出任何心拍"

    beats = np.array(beat_list)
    logger.info(f"心拍切分完成，共{len(beat_list)}个")
    return "success", beats, beat_positions, "心拍切分完成"