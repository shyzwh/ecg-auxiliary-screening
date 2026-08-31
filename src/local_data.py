import wfdb
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from src.config_utils import DEFAULT_CONFIG, resolve_path

data_path = resolve_path(DEFAULT_CONFIG["data_dir"]) / "100"
record = wfdb.rdrecord(data_path, channels=[0])
annotation = wfdb.rdann(data_path, 'atr')

# 画出前10秒的心电波形
fs = record.fs
signal = record.p_signal[:, 0]
time = [i / fs for i in range(len(signal))]

plt.figure(figsize=(12, 4))
plt.plot(time[:fs*10], signal[:fs*10])
plt.xlabel('时间（秒）')
plt.ylabel('电压（mV）')
plt.title('MIT-BIH 100号记录 - 前10秒心电图')
plt.show()