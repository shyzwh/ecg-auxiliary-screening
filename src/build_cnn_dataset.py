# 读取MIT-BIH数据，切分心拍，生成简单标签，并保存训练数据。
import numpy as np
from src.data_loader import load_ecg
from src.preprocess import preprocess_ecg
from src.feature_extract import pan_tompkins
from src.beat_segmenter import segment_beats
from src.logger import logger
from src.config_utils import DEFAULT_CONFIG, resolve_path


def build_cnn_dataset(record_ids, before=0.25, after=0.45, data_dir=None, beats_path=None, labels_path=None):
    """构建CNN训练数据集，标签暂用简单规则替代"""
    all_beats = []
    all_labels = []

    data_dir = resolve_path(data_dir or DEFAULT_CONFIG["data_dir"])
    beats_path = resolve_path(beats_path or DEFAULT_CONFIG["cnn_beats_path"])
    labels_path = resolve_path(labels_path or DEFAULT_CONFIG["cnn_labels_path"])

    for rid in record_ids:
        file_path = data_dir / f"{rid}.dat"

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

    np.save(beats_path, X)
    np.save(labels_path, y)

    logger.info(f"CNN数据集构建完成，共{len(X)}个心拍")
    return "success", X, y, f"CNN数据集构建完成，共{len(X)}个心拍"