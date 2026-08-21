# 用切分好的心拍，训练一个正常/异常二分类CNN。
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from .logger import logger

if __name__ == "__main__":
    # 直接运行时，临时改成绝对导入
    import sys, os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from logger import logger
else:
    # 作为模块被导入时，用相对导入
    from .logger import logger

def build_cnn_model(input_length):
    """构建轻量级1D-CNN二分类模型"""
    model = keras.Sequential([
        keras.layers.Input(shape=(input_length, 1)),
        keras.layers.Conv1D(32, kernel_size=5, activation="relu", padding="same"),
        keras.layers.MaxPooling1D(pool_size=2),
        keras.layers.Conv1D(64, kernel_size=5, activation="relu", padding="same"),
        keras.layers.MaxPooling1D(pool_size=2),
        keras.layers.Conv1D(128, kernel_size=3, activation="relu", padding="same"),
        keras.layers.GlobalAveragePooling1D(),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(1, activation="sigmoid")
    ])
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model


def train_cnn(beats, labels, model_path="models/cnn_model.h5"):
    """训练CNN模型并保存"""
    try:
        if len(beats) < 20:
            return "error", None, "心拍数量太少，无法训练CNN"

        # 整理输入格式
        X = np.array(beats)
        X = X[..., np.newaxis]
        y = np.array(labels)

        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # 构建模型
        model = build_cnn_model(X.shape[1])

        # 训练
        history = model.fit(
            X_train, y_train,
            epochs=20,
            batch_size=32,
            validation_data=(X_test, y_test),
            verbose=1
        )

        # 保存模型
        model.save(model_path)

        # 输出测试集评估
        loss, acc = model.evaluate(X_test, y_test, verbose=0)
        logger.info(f"CNN训练完成，测试准确率：{acc:.4f}")
        return "success", acc, f"CNN训练完成，测试准确率：{acc:.4f}"
    except Exception as e:
        logger.error(f"CNN训练失败：{e}")
        return "error", None, f"CNN训练失败：{e}"