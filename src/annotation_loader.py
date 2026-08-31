# 读取 .atr 注释文件里的心拍类型。
import wfdb
import numpy as np
from src.logger import logger


def load_beat_annotations(record_path):
    """
    读取MIT-BIH .atr注释，返回R峰位置和心拍类型。
    返回：status, ann_positions, ann_symbols, msg
    """
    try:
        annotation = wfdb.rdann(record_path, "atr")
        return "success", annotation.sample, annotation.symbol, "注释读取成功"
    except Exception as e:
        logger.error(f"注释读取失败：{e}")
        return "error", None, None, f"注释读取失败：{e}"


def map_labels(symbols):
    """
    将MIT-BIH心拍类型转换为二分类标签。
    正常=0，异常=1。
    """
    normal_codes = {"N", "L", "R", "e", "j"}
    labels = []
    for s in symbols:
        if s in normal_codes:
            labels.append(0)
        else:
            labels.append(1)
    return labels