import os
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import signal
from scipy.stats import boxcox


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
        certain_diff = 0.2  # 窗口内最大值与最小值的差值，之前0.8太大，导致有些检测不到
        onset_idx = []      # 不将0设为起始点
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


class SplitData:
    """
    数据切片
    split_by_time: 按时间切片
    split_by_feature: 按特征切片
    """

    def __init__(self, fs):
        """
        fs: 采样率
        """
        self.fs = fs  # 采样率
        self.peak = Peak(fs=fs)  # 实例化peak类

    def split_by_time(self, wav, abp, sample_num, overlap):
        """
        Parameters
        ----------
        wav: 原始ppg信号
        abp: 原始abp信号
        sample_num: 截取的周期数
        overlap: 重叠周期
        """
        ppg, ecg = wav
        wav_data = []
        bp_data = []
        interval = sample_num * self.fs 
        roll_num = sample_num - overlap
        roll_step = roll_num * self.fs  # 滑动步长

        for i in range(0, len(ppg) - interval, roll_step):
            # 重叠2秒，所以步进时间是3秒（5-2）
            start = i
            end = i + interval
            tmp = [ppg[start: end], ecg[start: end]]

            try:
                sbp, dbp = self.peak.abp_mean(abp[start: end])
            except IndexError:
                sbp = np.max(abp[start: end])  # 用最大值代替
                dbp = np.min(abp[start: end])  # 用最小值代替

            if not quality_assess(tmp, sbp, dbp):
                continue

            wav_data.append(tmp)
            bp_data.append([sbp, dbp])

        return np.asarray(wav_data).astype(np.float32), np.asarray(bp_data).astype(np.float32)

    def split_by_feature(self, ppg_data, abp_data, feat_num, overlap):
        """
        Parameters
        ----------
        个人认为这种方法是有问题的，加入了很多fake data（重采样），因此放弃
        ppg_data: 原始ppg信号
        abp_data: 原始abp信号
        feat_num: 截取的周期数
        overlap: 重叠周期
        """
        if len(ppg_data) <= 4000:
            _, feat_idx = self.peak.detect_peaks_short(ppg_data)
        else:
            _, feat_idx = self.peak.detect_peaks_long(ppg_data)

        feat_length = len(feat_idx)
        roll_num = feat_num - overlap
        sec_length = feat_num * self.fs  # 切片最终长度，feat_num表示周期数
        wav_data = []
        bp_data = []

        for i in range(0, feat_length - feat_num, roll_num):
            start_index = feat_idx[i]
            end_index = feat_idx[i + feat_num]

            tmp_ppg = ppg_data[start_index: end_index]  # 切片
            try:
                res_ppg = signal.resample(tmp_ppg, sec_length)  # 重采样
            except ValueError:
                continue

            wav_data.append([res_ppg])

            try:
                sbp, dbp = self.peak.abp_mean(abp_data[start_index: end_index])
            except IndexError:
                # print(f"找不到血压值：{start_index} -- {end_index}")
                sbp = np.max(abp_data[start_index: end_index])  # 用最大值代替
                dbp = np.min(abp_data[start_index: end_index])  # 用最小值代替
            bp_data.append([sbp, dbp])

        return np.asarray(wav_data), np.asarray(bp_data)


def quality_assess(wav, sbp, dbp):
    """
    质量评估
    """
    wav = np.asarray(wav)
    PPG_THD = int(wav.shape[1] * 0.3)
    ECG_THD = int(wav.shape[1] * 0.3)

    # 超过范围的不要
    # 收缩压：60-200，舒张压：30-130
    if sbp < 60 or sbp > 200 or dbp < 30 or dbp > 130:
        return False

    wav_diff = np.diff(wav, axis=1)
    zero_nums = np.sum(wav_diff == 0, axis=1)
    nan_nums = np.sum(np.isnan(wav_diff))

    if nan_nums > 0 or zero_nums[0] > PPG_THD or zero_nums[1] > ECG_THD:
        return False

    return True


def create_dataset(root_path, sample_num=10, overlap=0):
    my_split = SplitData(fs=125)
    for i in range(1, 5):
        part = h5py.File(os.path.join(root_path, f"Part_{i}.mat"))
        part_data = part[f"Part_{i}"]

        part_wav_data = tuple()
        part_bp_data = tuple()

        # for j in range(3):  # for test
        for j in tqdm(range(len(part_data)), desc=f"processing Part_{i}"):
            ppg = part[part_data[j][0]][:, 0]  # 1000到几万不等, array格式
            abp = part[part_data[j][0]][:, 1]
            ecg = part[part_data[j][0]][:, 2]

            wav_intv, bp_intv = my_split.split_by_time(wav=[ppg, ecg],
                                                       abp=abp,
                                                       sample_num=sample_num,
                                                       overlap=overlap)
            if bp_intv.size == 0:
                continue

            part_wav_data += (wav_intv,)
            part_bp_data += (bp_intv,)

        part_wav_data = np.concatenate(part_wav_data, axis=0)
        part_bp_data = np.concatenate(part_bp_data, axis=0)

        np.save(os.path.join(root_path, f"Part_{i}_wav.npy"), part_wav_data)
        np.save(os.path.join(root_path, f"Part_{i}_bp.npy"), part_bp_data)

        print(f"Part_{i} wav shape: {part_wav_data.shape}")
        print(f"Part_{i} bp shape: {part_bp_data.shape}")


def create_h5():
    # mimic-ii preprocessed part data path
    path = r"I:\dataset\MIMIC_II"
    f = h5py.File(os.path.join(path, "bp_dataset_ii.h5"), "a")  # append mode
    for i in range(1, 5):
        wav_data = np.load(os.path.join(path, f"Part_{i}_wav.npy"))
        bp_data = np.load(os.path.join(path, f"Part_{i}_bp.npy"))

        if i == 1:
            f.create_dataset("wav", data=wav_data, compression=6, chunks=True, maxshape=(None, 2, 2000))
            f.create_dataset("bp", data=bp_data, compression=6, chunks=True, maxshape=(None, 2))
        else:
            f["wav"].resize(f["wav"].shape[0] + wav_data.shape[0], axis=0)
            f["wav"][-wav_data.shape[0]:] = wav_data

            f["bp"].resize(f["bp"].shape[0] + bp_data.shape[0], axis=0)
            f["bp"][-bp_data.shape[0]:] = bp_data

        print(f"add {i}th , chunk wav shape: {f['wav'].shape}")

    all_bp = f["bp"]
    print(all_bp.shape)
    bp_df = pd.DataFrame(all_bp)
    bp_df.to_csv(os.path.join(path, "bp.csv"))

    f.close()


def get_constant_para():
    """
    获取常数参数
    sbp_many: 3500
    sbp_low: 1000
    dbp_many: 4000
    dbp_low: 1000
    bias: 605
    """
    # root_path = r"I:\dataset\MIMIC_II\bp_torch_final"
    root_path = r"D:\Projects\Py_projects\blood_pressure\Blood_Pressure\bp_torch_final\mimic_ii\dataset"
    dset = h5py.File(os.path.join(root_path, "bp_train.h5"))
    bp = dset['bp'][:, 1]

    bp_boxcox, lmbda = boxcox(bp)
    
    # lambda is :  -1.450212080509659
    # after boxcox : 0.6871845722198486 -- 0.6887989640235901

    bp_mean = np.mean(bp, axis=0) 
    bp_std = np.std(bp, axis=0)    
    one_sigma = (bp_mean + bp_std).astype(np.int32)
    two_sigma = (bp_mean + 2 * bp_std).astype(np.int32)

    many_shot = np.sum((bp > one_sigma) & (bp < one_sigma + 1))
    low_shot = np.sum((bp > two_sigma) & (bp < two_sigma + 1))
    
    print(f"many shot: {many_shot}")
    print(f"low_shot: {low_shot}")
    print("lambda is : ", lmbda)
    print(f"after boxcox : {min(bp_boxcox)} -- {max(bp_boxcox)}")


if __name__ == "__main__":
    root_path = r"I:\dataset\MIMIC_II"
    
    # create_dataset(root_path, sample_num=5, overlap=0)
    # create_dataset(root_path, sample_num=8, overlap=0)
    # create_dataset(root_path, sample_num=10, overlap=0)
    # create_dataset(root_path, sample_num=10, overlap=5)
    # create_dataset(root_path, sample_num=12, overlap=0)
    # create_dataset(root_path, sample_num=15, overlap=0)
    create_h5()
    # get_constant_para()




