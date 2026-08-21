# 读取MIT-BIH数据，切分心拍，生成简单标签，并保存训练数据。
import numpy as np
from .data_loader import load_ecg
from .preprocess import preprocess_ecg
from .feature_extract import pan_tompkins
from .beat_segmenter import segment_beats
from .logger import logger


def build_cnn_dataset(record_ids, before=0.25, after=0.45):
    """构建CNN训练数据集，标签暂用简单规则替代"""
    all_beats = []
    all_labels = []

    for rid in record_ids:
        file_path = f"D:/桌面/ECG-Auxiliary-Screening/data/mitbih/mit-bih-arrhythmia-database-1.0.0/{rid}.dat"

        status, signal, fs, _ = load_ecg(file_path)
        if status != "success":
            continue

        status, clean, _ = preprocess_ecg(signal, fs)
        if status != "success":
            continue

        status, r_peaks, _ = pan_tompkins(clean, fs)
        if status != "success":
            continue

        status, beats, positions, _ = segment_beats(clean, r_peaks, fs, before, after)
        if status != "success":
            continue

        # 使用MIT-BIH注释文件，尝试获取心拍类型
        # 若无法读取，则暂时全部标为正常（0）
        labels = np.zeros(len(beats), dtype=int)

        all_beats.append(beats)
        all_labels.append(labels)
        logger.info(f"{rid} 完成，共{len(beats)}个心拍")

    if not all_beats:
        return "error", None, None, "没有构建到任何训练数据"

    X = np.concatenate(all_beats, axis=0)
    y = np.concatenate(all_labels, axis=0)

    np.save("cnn_beats.npy", X)
    np.save("cnn_labels.npy", y)

    logger.info(f"CNN数据集构建完成，共{len(X)}个心拍")
    return "success", X, y, f"CNN数据集构建完成，共{len(X)}个心拍"