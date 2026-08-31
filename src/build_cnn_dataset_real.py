# 读取MIT-BIH记录，用真实注释给每个心拍打上“正常=0，异常=1”的标签。
import numpy as np
from src.data_loader import load_ecg
from src.preprocess import preprocess_ecg
from src.feature_extract import pan_tompkins
from src.beat_segmenter import segment_beats
from src.annotation_loader import load_beat_annotations, map_labels
from src.logger import logger
from src.config_utils import DEFAULT_CONFIG, resolve_path


def build_real_cnn_dataset(record_ids, data_dir=None, beats_path=None, labels_path=None):
    """用真实注释构建CNN训练数据集"""
    all_beats = []
    all_labels = []

    data_dir = resolve_path(data_dir or DEFAULT_CONFIG["data_dir"])
    beats_path = resolve_path(beats_path or DEFAULT_CONFIG["cnn_beats_real_path"])
    labels_path = resolve_path(labels_path or DEFAULT_CONFIG["cnn_labels_real_path"])

    for rid in record_ids:
        base_path = data_dir / rid

        status, signal, fs, _ = load_ecg(f"{base_path}.dat")
        if status != "success":
            logger.warning(f"{rid} 数据读取失败，跳过")
            continue

        status, clean_signal, _ = preprocess_ecg(signal, fs)
        if status != "success":
            logger.warning(f"{rid} 预处理失败，跳过")
            continue

        status, r_peaks, _ = pan_tompkins(clean_signal, fs)
        if status != "success":
            logger.warning(f"{rid} R峰检测失败，跳过")
            continue

        status, beats, positions, _ = segment_beats(clean_signal, r_peaks, fs)
        if status != "success":
            logger.warning(f"{rid} 心拍切分失败，跳过")
            continue

        # 读取真实注释
        status, ann_positions, ann_symbols, _ = load_beat_annotations(base_path)
        if status != "success":
            logger.warning(f"{rid} 注释读取失败，跳过")
            continue

        # 把注释位置和心拍位置对齐
        labels = []
        for r in positions:
            # 找到离这个R峰最近的注释
            if len(ann_positions) > 0:
                nearest_idx = np.argmin(np.abs(np.array(ann_positions) - r))
                symbol = ann_symbols[nearest_idx]
                # 正常=0，异常=1
                if symbol in ["N", "L", "R", "e", "j"]:
                    labels.append(0)
                else:
                    labels.append(1)
            else:
                labels.append(0)

        all_beats.append(beats)
        all_labels.append(labels)
        logger.info(f"{rid} 完成，共{len(beats)}个心拍，异常{sum(labels)}个")

    if not all_beats:
        return "error", None, None, "没有成功构建任何CNN训练数据"

    X = np.concatenate(all_beats, axis=0)
    y = np.concatenate([np.array(x) for x in all_labels], axis=0)

    np.save(beats_path, X)
    np.save(labels_path, y)

    logger.info(f"真实CNN数据集构建完成，共{len(X)}个心拍，异常{sum(y)}个")
    return "success", X, y, f"真实CNN数据集完成，共{len(X)}个，异常{sum(y)}个"
