import numpy as np
import matplotlib.pyplot as plt
from feature_extract import pan_tompkins

fs = 360
t_total = 8
t = np.linspace(0, t_total, fs * t_total)

# 构造模拟心电
ecg_sim = np.zeros_like(t)
heart_rate = 75
beat_interval = fs * 60 / heart_rate

# 边界问题
for beat in np.arange(0, len(t), beat_interval):
    pos = int(beat)
    start = pos - 15
    end = pos + 15
    if start < 0 or end > len(ecg_sim):
        continue
    ecg_sim[start:end] += 1.2 * np.hanning(30)

# 加噪声
noise = 0.08 * np.random.randn(len(ecg_sim))
ecg_noisy = ecg_sim + noise

# 检测R峰
peaks = pan_tompkins(ecg_noisy, fs)

# 画图
plt.figure(figsize=(12, 4))
plt.plot(t, ecg_noisy, color="#1f77b4", linewidth=0.8)
plt.scatter(np.array(peaks) / fs, ecg_noisy[peaks], c="red", s=40, marker="*", label="R-Peak")
plt.xlabel("Time(s)")
plt.ylabel("ECG Amplitude")
plt.title("Pan-Tompkins R-wave detection")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

print(f"检测到R波数量：{len(peaks)}")
hr_calc = len(peaks) / t_total * 60
print(f"估算心率：{hr_calc:.1f} bpm")