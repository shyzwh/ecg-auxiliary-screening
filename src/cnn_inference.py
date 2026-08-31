# 用CNN对新数据做预测
import numpy as np
from tensorflow import keras

from src.beat_segmenter import segment_beats
from src.logger import logger


def predict_abnormal_beats(signal, r_peaks, fs, model_path="models/cnn_model.h5"):
    # CNN推理：输入信号和R峰，输出异常心拍位置
    try:
        model_path = str(model_path).replace("\\", "/")
        status, beats, positions, _ = segment_beats(signal, r_peaks, fs)
        if status != "success":
            return "error", None, "心拍切分失败"

        model = keras.models.load_model(model_path)

        # 3. 整理输入格式
        X = np.array(beats)
        X = X[..., np.newaxis]

        # 4. 预测
        probs = model.predict(X)
        preds = (probs > 0.5).astype(int).flatten()
        confidence = float(np.mean(np.maximum(probs.flatten(), 1 - probs.flatten())))

        # 5. 找出异常心拍位置
        abnormal_positions = [positions[i] for i in range(len(preds)) if preds[i] == 1]
        abnormal_count = len(abnormal_positions)

        logger.info(f"CNN推理完成，异常心拍数：{abnormal_count}")
        return "success", abnormal_positions, confidence, f"CNN推理完成，异常心拍数：{abnormal_count}"
    except Exception as e:
        logger.error(f"CNN推理失败：{e}")
        return "error", None, 0.0, f"CNN推理失败：{e}"