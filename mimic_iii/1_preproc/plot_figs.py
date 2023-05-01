import numpy as np
import h5py
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms

from scipy.optimize import curve_fit
from scipy.stats import boxcox
from sklearn.model_selection import train_test_split
import os


# 画图用


class Peak:
    """
    寻波
    """

    def __init__(self, fs):
        self.fs = fs  # 采样率

    def ppg2ssf_memo(self, signal):
        # SSF 算法
        # 将PPG转换成SSF信号
        sig_len = len(signal)  # 信号总长度
        ssf = [0] * sig_len  # 这样创建比较快

        memo = self.helper(signal)  # 获取备忘录

        win_size = int(self.fs * 0.125)  # 窗口大小，系数也可以是0.128
        # 这里为 125 * 0.125 = 15
        # 滑动窗口
        ssf_tmp = sum(memo[:win_size])  # list使用自带sum计算较快
        ssf[win_size - 1] = ssf_tmp
        for j in range(win_size, sig_len):
            ssf_tmp += memo[j] - memo[j - win_size]
            ssf[j] = ssf_tmp

        return ssf

    def helper(self, signal):
        # 备忘录
        arr_sig = np.array(signal)
        memo = np.diff(arr_sig)
        memo[memo < 0] = 0  # 小于0的置0
        memo = np.insert(memo, 0, 0)  # 在最前面插入0
        return memo

    def find_onset(self, signal, thd_T):
        # ppg或abp波峰一定在ssf onset之间
        # 先找出ssf的onset，再寻波
        ssf_data = self.ppg2ssf_memo(signal)  # ppg转换成ssf
        length = len(ssf_data)
        radius = int(self.fs * 0.15)  # 搜索半径
        certain_diff = 0.01  # 窗口内最大值与最小值的差值，之前0.8太大，导致有些检测不到
        onset_idx = []  # 不将0设为起始点
        ref_period = int(self.fs * 0.2)  # 不应期:37个点
        step = int(self.fs * 0.05)

        # np.mean运行速度不如sum后除以长度
        thd = 3 * sum(ssf_data[: self.fs * thd_T]) / self.fs / thd_T * 0.6  # 初始阈值
        for i in range(0, length, step):
            if ssf_data[i] >= thd:
                # 边界处理
                win_start = i - radius if (i - radius > 0) else 0
                win_end = i + radius if (i + radius < length) else length
                # 开始搜索
                search_window = ssf_data[win_start: win_end]
                tmp_max = max(search_window)  # 搜索前后150ms(fs * 0.15)内的最值
                tmp_min = min(search_window)
                if tmp_max - tmp_min >= certain_diff:  # certain value
                    thd = tmp_max * 0.6  # 更新阈值
                    # 向前搜索onset
                    j = i
                    while ssf_data[j] > 0.01 * tmp_max:
                        j -= 1
                    # j -= 4  # onset position，是第一个超过最大值1%的索引
                    # 这里减4是为了去除低通滤波器的相移，总的其实是减5
                    # 下面的判断，一是为了防止onset_idx为空，二是不能有重复值
                    if onset_idx == [] or j != onset_idx[-1]:
                        onset_idx.append(j)
        # print("原始波峰个数：", len(onset_idx) - 1)

        # 使用不应期对onset校准
        onset_res = []
        for k in range(len(onset_idx) - 1):
            # 使用下述方法的结果是取相邻点中的第二个点
            if onset_idx[k + 1] - onset_idx[k] >= ref_period:
                onset_res.append(onset_idx[k])

        onset_res.append(onset_idx[-1])  # onset最后一个点
        onset_res.append(length - 1)  # 信号最后一个点
        return onset_res

    def detect_peaks_long(self, signal):
        # 长片段的信号处理
        # 将信号切分，分段处理
        interval = 2000  # 500太小了
        win_nums = len(signal) // interval  # 分段数

        peak_idx = []
        valley_idx = []
        onset = []
        for i in range(win_nums):
            start_intv = i * interval
            end_intv = (i + 1) * interval
            # 待处理的信号片段
            data = signal[start_intv: end_intv]
            try:
                onset = self.find_onset(data, thd_T=3)
            except IndexError:
                pass
                # print(f"找不到onset: {start_intv} -- {end_intv}")

            onset_length = len(onset)

            valley = [onset_loc + start_intv for onset_loc in onset]
            valley_idx.extend(valley[:-1])

            for j in range(onset_length - 1):
                start = onset[j]
                end = onset[j + 1]
                try:
                    peak = np.argmax(data[start: end]) + start + start_intv
                    peak_idx.append(peak)
                except ValueError:
                    # print(f"有问题的索引: {start} -- {end}")
                    peak_idx.append(start + start_intv)

        return peak_idx, valley_idx

    def detect_peaks_short(self, signal):
        # 针对几个周期切片的小片段
        onset = self.find_onset(signal, thd_T=3)
        onset_length = len(onset)

        # onset索引就是波谷索引
        valley_idx = onset[:-1]  # 最后一个是片段端点，一般不是波谷
        peak_idx = []
        for i in range(onset_length - 1):
            start = onset[i]
            end = onset[i + 1]
            try:
                peak = np.argmax(signal[start: end]) + start
                peak_idx.append(peak)
            except ValueError:
                # print(f"有问题的索引: {start} -- {end}")
                peak_idx.append(start)

        # 一次性输出波峰波谷
        return peak_idx, valley_idx

    def abp_mean(self, abp_seg):
        # 计算给定ABP片段内的SBP、DBP均值
        peaks, valleys = self.detect_peaks_short(abp_seg)
        sbp, dbp = 0, 0

        y_peaks = []  # 峰值
        for idx in peaks:
            y_peaks.append(abp_seg[idx])

        if y_peaks:
            sbp = sum(y_peaks) / len(y_peaks)  # np.mean有点慢

        y_valleys = []  # 谷值
        for idx in valleys:
            y_valleys.append(abp_seg[idx])  # 最值需要从原始片段中选取

        if valleys:
            dbp = sum(y_valleys) / len(y_valleys)

        return sbp, dbp


def see_wav():
    path = r"H:\p000333\3092245_f_recmb.npy"
    data = np.load(path, allow_pickle=True).item()
    wav_data = data['wav_data']
    # print(wav_data.shape)

    idx = 200

    ppg = wav_data[idx][:, 0]
    abp = wav_data[idx][:, 1]
    ecg = wav_data[idx][:, 2]

    plt.subplot(311)
    plt.plot(ppg)

    plt.subplot(312)
    plt.plot(ecg)

    plt.subplot(313)
    plt.plot(abp)

    plt.show()


def wav_ssf(font_size=30):
    path = r"D:\Projects\Py_projects\blood_pressure\Blood_Pressure\my_bp_final\data"
    bp = np.load(os.path.join(path, "abp.npy")).reshape(-1)
    # print(bp.shape)

    start = 99900
    end = start + 500
    abp = bp[start: end]

    peak = Peak(fs=125)

    ssf = peak.ppg2ssf_memo(abp)
    onset_idx = peak.find_onset(abp, thd_T=2)
    my_ppg_peaks, _ = peak.detect_peaks_short(abp)

    y_ssf_peaks = []
    y_ppg_peaks = []
    y_onset_ssf_peaks = []
    y_onset_ppg_peaks = []

    for e in my_ppg_peaks:
        y_ssf_peaks.append(ssf[e])
        y_ppg_peaks.append(abp[e])
    for a in onset_idx:
        y_onset_ssf_peaks.append(ssf[a])
        y_onset_ppg_peaks.append(abp[a])

    # print(len(onset_idx))

    # 画图
    # plt.figure(figsize=(12, 10))
    plt.subplot(211)
    plt.title("ABP signal", fontsize=font_size)
    plt.plot(abp, linewidth=3, label="abp data")
    plt.scatter(my_ppg_peaks, y_ppg_peaks, s=100, c='r', label="peak")
    plt.scatter(onset_idx, y_onset_ppg_peaks, s=100, c='g', label="valley")
    plt.ylabel("Amplitude", fontsize=font_size)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.legend(loc="center left", fontsize=font_size//2)

    plt.subplot(212)
    plt.title("ssf signal", fontsize=font_size)
    plt.plot(ssf, linewidth=3, label="ssf data")
    plt.scatter(onset_idx, y_onset_ssf_peaks, s=100, c='g', label="ssf onset")
    plt.xlabel("samples", fontsize=font_size)
    plt.ylabel("Amplitude", fontsize=font_size)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.legend(loc="center left", fontsize=font_size//2)

    plt.show()


def see_ssf(font_size=30):
    path = r"D:\Projects\Py_projects\blood_pressure\Blood_Pressure\my_bp_final\data"
    bp = np.load(os.path.join(path, "abp.npy")).reshape(-1)

    start = 99900
    end = start + 500
    abp = bp[start: end]

    peak = Peak(fs=125)
    ssf = peak.ppg2ssf_memo(abp)

    plt.title("ssf signal", fontsize=font_size)
    plt.plot(ssf, linewidth=3, label="ssf data")
    # plt.xlabel("samples", fontsize=font_size)
    plt.ylabel("Amplitude", fontsize=font_size)
    plt.xticks([])
    plt.yticks(fontsize=font_size)

    plt.show()


def plot_hist():
    root_path = r"D:\Projects\Py_projects\blood_pressure\Blood_Pressure\my_bp_torch\mimic_ii\dataset"
    # root_path = r"D:\Projects\Py_projects\blood_pressure\Blood_Pressure\my_bp_torch\mimic_iii\dataset"
    dset = h5py.File(os.path.join(root_path, "bp_dataset_ii.h5"))
    bp = dset['bp'][:, 1]
    # print(bp.shape)

    bp_boxcox, lmbda = boxcox(bp)

    plt.figure()
    plt.title("DBP distribution", fontsize=30)
    plt.hist(x=bp, bins=50, edgecolor='black')
    plt.xlabel("DBP(mmHg)", fontsize=30)
    plt.ylabel("Frequence", fontsize=30)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.show()

    plt.figure()
    plt.title("BoxCox DBP distribution", fontsize=30)
    plt.hist(x=bp_boxcox, bins=50, edgecolor='black')
    plt.xlabel("BoxCox DBP(mmHg)", fontsize=30)
    plt.ylabel("Frequence", fontsize=30)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.show()


if __name__ == "__main__":
    # plot_hist()
    # wav_ssf()
    # see_ssf()
    plot_hist()
