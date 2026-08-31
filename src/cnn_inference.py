# 用CNN对新数据做预测
import numpy as np
from tensorflow import keras
from src.beat_segmenter import segment_beats
from src.logger import logger


def predict_abnormal_beats(signal, r_peaks, fs, model_path="models/cnn_model.h5"):
    """用训练好的CNN模型，预测哪些心拍异常"""
    try:
        # 1. 切分心拍
        status, beats, positions, _ = segment_beats(signal, r_peaks, fs)
        if status != "success":
            return "error", None, "心拍切分失败"

        # 2. 加载模型
        model = keras.models.load_model(model_path)

        # 3. 整理输入格式
        X = np.array(beats)
        X = X[..., np.newaxis]

        # 4. 预测
        probs = model.predict(X)
        preds = (probs > 0.5).astype(int).flatten()

        # 5. 找出异常心拍位置
        abnormal_positions = [positions[i] for i in range(len(preds)) if preds[i] == 1]
        abnormal_count = len(abnormal_positions)

        logger.info(f"CNN推理完成，异常心拍数：{abnormal_count}")
        return "success", abnormal_positions, f"CNN推理完成，异常心拍数：{abnormal_count}"
    except Exception as e:
        logger.error(f"CNN推理失败：{e}")
        return "error", None, f"CNN推理失败：{e}"