import wfdb
import matplotlib.pyplot as plt
from feature_extract import pan_tompkins

# 1. 读取真实MIT-BIH数据
record_path = "D:/桌面/ECG-Auxiliary-Screening/data/mitbih/mit-bih-arrhythmia-database-1.0.0/100" # 绝对路径
record = wfdb.rdrecord(record_path, channels=[0])
signal = record.p_signal[:, 0]
fs = record.fs

# 2. 取前30秒进行分析
duration = 30
n_samples = int(fs * duration)
signal_30s = signal[:n_samples]

# 3. R峰检测
r_peaks = pan_tompkins(signal_30s, fs)

# 画图前解决中文空格
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 4. 画图
time = [i / fs for i in range(len(signal_30s))]
plt.figure(figsize=(14, 5))
plt.plot(time, signal_30s, color="black", linewidth=0.6)
plt.scatter([r / fs for r in r_peaks],
            [signal_30s[r] for r in r_peaks],
            c="red", s=30, marker="*", label="R-Peak")
plt.xlabel("时间（秒）")
plt.ylabel("电压（mV）")
plt.title("MIT-BIH 100 号记录：真实心电R峰检测")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# 5. 输出结果
print(f"采样率：{fs} Hz")
print(f"前30秒检测到R峰数量：{len(r_peaks)}")
print(f"估算心率：{len(r_peaks) / duration * 60:.1f} bpm")

from feature_extract import compute_hrv_features

hrv_features = compute_hrv_features(r_peaks, fs)
print("hrv特征：")
for key, value in hrv_features.items():
    if value is not None:
        print(f"{key}: {value}")
    else:
        print(f"{key}: 无法计算")