import numpy as np
import pandas as pd
from .data_loader import load_ecg
from .preprocess import preprocess_ecg
from .config_utils import DEFAULT_CONFIG, resolve_path


def export_mitbih_to_csv(record_id, output_path=None, data_dir=None):
    """把MIT-BIH记录导出为带表头CSV：time,voltage"""
    file_path = resolve_path(data_dir or DEFAULT_CONFIG["data_dir"]) / f"{record_id}.dat"

    status, signal, fs, msg = load_ecg(file_path)
    if status != "success":
        print(msg)
        return

    # 取前30秒
    n = int(fs * 30)
    signal_30s = signal[:n]
    time = np.arange(n) / fs

    df = pd.DataFrame({
        "time": time,
        "voltage": signal_30s
    })

    if output_path is None:
        output_path = resolve_path(DEFAULT_CONFIG["upload_dir"]) / f"{record_id}_30s.csv"

    output_path = resolve_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"已保存：{output_path}")
    print(f"采样率：{fs} Hz，点数：{len(df)}")

# 测试数据
if __name__ == "__main__":
    record_list = ["100", "106", "108", "109", "203"]
    for rid in record_list:
        export_mitbih_to_csv(rid)