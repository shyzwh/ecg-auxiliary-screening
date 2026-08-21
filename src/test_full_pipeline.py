# 测试完整链路——数据读取、预处理、R峰检测、12项特征提取。
# 运行后需要再输入 python -m src.test_full_pipeline 进行测试

# 相对引入
from .data_loader import load_ecg
from .preprocess import preprocess_ecg
from .feature_extract import pan_tompkins, extract_all_features

# 1. 读取真实MIT-BIH数据
file_path = "D:/桌面/ECG-Auxiliary-Screening/data/mitbih/mit-bih-arrhythmia-database-1.0.0/100.dat"
status, signal, fs, msg = load_ecg(file_path)
print(f"读取结果：{status}，{msg}")

if status == "success":
    # 2. 预处理
    status_pre, clean_signal, msg_pre = preprocess_ecg(signal, fs)
    print(f"预处理结果：{status_pre}，{msg_pre}")

    if status_pre == "success":
        # 3. R峰检测
        status_r, r_peaks, msg_r = pan_tompkins(clean_signal, fs)
        print(f"R峰检测结果：{status_r}，{msg_r}")

        if status_r == "success":
            # 4. 提取12项特征
            status_f, features, msg_f = extract_all_features(clean_signal, r_peaks, fs)
            print(f"特征提取结果：{status_f}，{msg_f}")

            if status_f == "success":
                print("\n===== 12项心电特征 =====")
                for key, value in features.items():
                    print(f"{key}: {value}")
            else:
                print(msg_f)
        else:
            print(msg_r)
    else:
        print(msg_pre)
else:
    print(msg)