import heartpy  # 心电信号去噪
import pywt
import math
import pickle
import os
import fnmatch  # 文件名匹配
import wfdb
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import signal
from scipy.stats import boxcox
from sklearn.utils import shuffle


# MIMIC-III
# all_wav: (N, 2, 1250)
# all_bp: (N, 2)
# all_ga: (N, 2)  # gender: {0., 1.} age: [0., 1.]


class DataExtract:
    """
    提取原始数据
    """

    def __init__(self, origin_root_path, save_root_path):
        """
        :param origin_root_path: 存放原始数据的目录，数据为npy格式
        :param save_root_path: 存储路径
        """
        self.origin_root_path = origin_root_path
        self.save_root_path = save_root_path

    def read_hea(self, hea_path):
        # 读取病人hea文件信息
        # hea_path = 'E:/npy_data/p00/p000107/p000107-2121-11-30-20-03n'  # 不用带后缀
        hea_data = wfdb.rdheader(hea_path)

        into_id = hea_data.file_name[0][:7]  # 入院编号，e.g. 3805787
        into_date = str(hea_data.base_datetime).split(" ")[0]  # 入院时间, 2122-05-14
        tmp = [s.lstrip("0") for s in into_date.split("-")]  # 去除前导0
        into_date = '/'.join(tmp)  # 2122/5/14, 2122/9/7
        return into_date, into_id

    def extract(self, csv_path):
        """
        :param csv_path: 存储病人生理信息的文件（身高体重等）
        """
        all_sbj_info = pd.read_csv(csv_path)
        for cur, dirs, files in os.walk(self.origin_root_path):
            # cur: 当前遍历到的目录
            # dirs: 当前目录下的文件夹列表
            # files: 当前目录下的文件列表

            sbj_id = cur.split('\\')[-1]  # p000107
            # cur最先遍历根目录，比如这里设置的就是"dataset"，其他都是pxxxxx
            # 这段语句就是为了区分这两种情况
            if sbj_id[1].isalpha():
                continue

            hea_files = fnmatch.filter(files, "*.hea")  # 筛选当前目录下所有的hea文件
            ID = int(sbj_id[1:])  # 000107, 匹配用
            match_phy_data = all_sbj_info.loc[all_sbj_info["SUBJECT_ID"] == ID]  # 某个病人的所有住院记录

            # 病人全部数据
            sbj_data = {'gender': None, 'age': None, 'height': None,
                        'weight': None, 'wav_data': None, 'into_date': None}
            # wav_data shape: (3, N)
            # 0: ppg
            # 1: abp
            # 2: ecg

            print(f"processing {sbj_id} ...")
            # os.walk是迭代器，不能用tqdm查看进度
            for hf in hea_files:
                # 读取hea文件，获取入院时间与编号
                hea_into_date, into_id = self.read_hea(os.path.join(cur, hf[:-4]))

                # 根据入院编号获取 临床波形数据
                # wav_data = []
                wav_data = tuple()
                into_npy_files = fnmatch.filter(files, f"{into_id}*.npy")
                for nf in into_npy_files:
                    clinical_data = np.load(os.path.join(cur, nf))  # 波形数据称为 临床数据
                    wav_data += (clinical_data,)

                if len(wav_data) > 0:
                    wav_data = np.concatenate(wav_data, axis=1)  # 在波形长度的维度上拼接
                    sbj_data['wav_data'] = wav_data  # 存储波形数据
                else:
                    # 放在这里有点后处理的感觉，只是为了形式上更美观
                    # 如果没有相应的npy文件，那么wav_data一定是空，也就没有继续寻找的必要了
                    continue

                # 根据入院时间获取 生理数据（性别、年龄、身高、体重）
                if match_phy_data.empty:  # data.csv不包含该病人的生理信息
                    # 存放到另一个文件夹
                    no_record_path = os.path.join(self.save_root_path, 'no_record_data', sbj_id)
                    if not os.path.exists(no_record_path):
                        os.mkdir(no_record_path)

                    np.save(os.path.join(no_record_path, f"{into_id}.npy"), sbj_data)
                    # with open(os.path.join(no_record_path, f"{into_id}.pkl"), "wb") as f:
                    #     pickle.dump(sbj_data, f)
                    continue

                # 二次筛选
                match_data = match_phy_data.loc[match_phy_data['ADMI_TIME'].str.contains(f'{hea_into_date}')]
                # 就算存在同一天有多条记录的情况，记录的数据也是一样的，因此选取第一条即可
                if not match_data.empty:  # 有筛选结果
                    _, _, gender, age, weight, height, _ = match_data.iloc[0]

                    sbj_data['gender'] = gender
                    sbj_data['age'] = age
                    sbj_data['into_date'] = hea_into_date  # 入院时间
                    if weight:
                        sbj_data['weight'] = weight
                    if height:
                        sbj_data['height'] = height
                else:
                    sbj_data['gender'] = match_phy_data.iloc[0]['GENDER']
                    sbj_data['age'] = match_phy_data.iloc[0]['AGE']

                # 什么时候存储是个问题
                # 放到if内，就是同时具有临床数据和生理数据才进行存储
                # 放到if外，就是无论有没有生理数据都会进行存储
                save_path = os.path.join(self.save_root_path, 'record_data',
                                         sbj_id)  # H:\extract_data\record_data\p000107
                if not os.path.exists(save_path):
                    os.mkdir(save_path)
                try:
                    # 超过4GB会报溢出错误
                    np.save(os.path.join(save_path, f"{into_id}.npy"), sbj_data)
                except OverflowError:
                    # 文件记录哪个病人的哪次住院记录数据量过大
                    # 溢出后会有个空文件，需要删除
                    os.remove(os.path.join(save_path, f"{into_id}.npy"))
                    with open(os.path.join(self.save_root_path, 'large_file.txt'), 'a') as txt_file:
                        txt_file.write(f"{sbj_id} -- {into_id} exceed 4GB, save {into_id}.pkl")
                    # 过大的文件用pkl存储
                    with open(os.path.join(save_path, f"{into_id}.pkl"), "wb") as f:
                        # print(f"{sbj_id} -- {into_id} exceed 4GB, save {into_id}.pkl")
                        pickle.dump(sbj_data, f, protocol=4)


class PPGFilter:
    """
    PPG滤波
    """

    def __init__(self, ppg_raw):
        self.ppg_raw = ppg_raw  # ppg未滤波信号

    def sure_shrink(self, var, coeffs):
        """
        求SureShrink法阈值
        :param var:  噪声方差
        :param coeffs: 小波分解系数
        :return:
        """
        N = len(coeffs)
        sqr_coeffs = []
        for coeff in coeffs:
            sqr_coeffs.append(math.pow(coeff, 2))
        sqr_coeffs.sort()
        pos = 0
        r = 0
        for idx, sqr_coeff in enumerate(sqr_coeffs):
            new_r = (N - 2 * (idx + 1) + (N - (idx + 1)) * sqr_coeff + sum(sqr_coeffs[0:idx + 1])) / N
            if r == 0 or r > new_r:
                r = new_r
                pos = idx
        thre = math.sqrt(var) * math.sqrt(sqr_coeffs[pos])
        return thre

    def visu_shrink(self, var, coeffs):
        """
        求VisuShrink法阈值
        """
        N = len(coeffs)
        thre = math.sqrt(var) * math.sqrt(2 * math.log(N))
        return thre

    def heur_sure(self, var, coeffs):
        """
        求HeurSure法阈值
        """
        N = len(coeffs)
        s = 0
        for coeff in coeffs:
            s += math.pow(coeff, 2)
        theta = (s - N) / N
        miu = math.pow(math.log2(N), 3 / 2) / math.pow(N, 1 / 2)
        if theta < miu:
            return self.visu_shrink(var, coeffs)
        else:
            return min(self.visu_shrink(var, coeffs), self.sure_shrink(var, coeffs))

    def mini_max(self, var, coeffs):
        """
        求Minimax法阈值:
        """
        N = len(coeffs)
        if N > 32:
            return math.sqrt(var) * (0.3936 + 0.1829 * math.log2(N))
        else:
            return 0

    def get_var(self, cD):
        """
        :param cD: 小波细节系数
        :return: 噪声方差
        """
        coeffs = cD
        abs_coeffs = []
        for coeff in coeffs:
            abs_coeffs.append(math.fabs(coeff))
        abs_coeffs.sort()
        pos = math.ceil(len(abs_coeffs) / 2)
        var = abs_coeffs[pos] / 0.6745
        return var

    def denoise(self, method='sureshrink', mode='soft', wavelet='db8', level=7):
        # 小波阈值去噪
        wave = pywt.Wavelet(wavelet)
        (cA, cD) = pywt.dwt(data=self.ppg_raw, wavelet=wave)
        var = self.get_var(cD)
        coeffs = pywt.wavedec(data=self.ppg_raw, wavelet=wavelet, level=level)  # 小波分解

        for idx, coeff in enumerate(coeffs):
            if idx == 0:
                coeff.fill(0)  # coeff是array格式
                continue
            if method == 'visushrink':
                thd = self.visu_shrink(var, coeff)
            elif method == 'sureshrink':
                thd = self.sure_shrink(var, coeff)
            elif method == 'heursure':
                thd = self.heur_sure(var, coeff)
            else:
                thd = self.mini_max(var, coeff)
            # thd = methods_dict[method](var, coeff)

            coeffs[idx] = pywt.threshold(coeff, thd, mode=mode)

        thd_data = pywt.waverec(coeffs, wavelet=wavelet)
        thd_data = signal.savgol_filter(thd_data, 11, 1, mode='nearest')  # 平滑滤波
        # thd_data = heartpy.filter_signal(thd_data, cutoff=10.0, sample_rate=300, order=4, filtertype='lowpass')
        return thd_data


class DataSift:
    """数据筛选"""

    def __init__(self, origin_root_path, save_root_path):
        """
        :param origin_root_path: 原始数据根目录
        :param save_root_path: 存储根目录
        """
        self.origin_root_path = origin_root_path
        self.save_root_path = save_root_path

    def judge_signal(self, data, signal='ppg'):
        """判断信号是否合格"""
        # 'signal': [diff_zero_num, val_max, val_min]
        # 'signal': [导数0的数量，信号最大值，信号最小值]
        signal_para = {'ppg': [400, 4, 0],
                       'abp': [400, 300, 10],
                       'ecg': [400, 2, -1]}  # 3000采样点的片段，400个导数为0，占比13.3%

        data_diff = np.diff(data)  # 一阶导数
        # diff_max, diff_min = np.max(data_diff), np.min(data_diff)  # 导数范围
        data_max, data_min = np.max(data), np.min(data)  # 信号范围
        diff_zero_num = np.sum(data_diff == 0)  # 导数0的数量

        # print(diff_min, diff_max, diff_zero_num)

        # cond1 = diff_max <= signal_para[signal][0]
        # cond2 = diff_min >= signal_para[signal][1]
        cond3 = diff_zero_num <= signal_para[signal][0]
        cond4 = data_max <= signal_para[signal][1]
        cond5 = data_min >= signal_para[signal][2]

        judge_cond = cond3 and cond4 and cond5  # cond1 and cond2 and

        return judge_cond

    def assess_quality(self, data):
        """信号质量评估"""
        ppg = data[0]
        abp = data[1]
        ecg = data[2]

        quality_cond = self.judge_signal(ppg, 'ppg') and self.judge_signal(abp, 'abp') and self.judge_signal(ecg, 'ecg')

        return quality_cond

    def data_sift(self, interval=3000):
        """数据筛选"""
        # interval: 先将信号切分成3000采样点的片段，再进行质量评估
        for cur, dirs, files in os.walk(self.origin_root_path):
            sbj_id = cur.split("\\")[-1]  # p000107
            npy_files = fnmatch.filter(files, "*.npy")

            print(f"sifting {sbj_id} ...")
            for nf in npy_files:
                origin_data = get_data(os.path.join(cur, nf))
                origin_wav = origin_data['wav_data']
                # 病人全部数据
                sbj_data = {'gender': origin_data['gender'], 'age': origin_data['age'], 'height': origin_data['height'],
                            'weight': origin_data['weight'], 'wav_data': None, 'into_date': origin_data['into_date']}

                wav_data = []  # 单个住院记录的所有波形数据
                length = origin_wav.shape[1]  # 长度
                win_num = length // interval

                for i in range(win_num):
                    start_idx = i * interval
                    end_idx = (i + 1) * interval
                    data = origin_wav[:, start_idx: end_idx]  # ppg, abp, ecg
                    # 质量评估
                    if self.assess_quality(data):
                        # 需要的信号，存储下来
                        wav_data.append(data)

                if wav_data:  # 非空才进行存储
                    save_path = os.path.join(self.save_root_path, sbj_id)
                    if not os.path.exists(save_path):
                        os.mkdir(save_path)
                    sbj_data['wav_data'] = np.asarray(wav_data)  # (N, 3, 3000)
                    # print(sbj_data['wav_data'].shape)
                    np.save(os.path.join(save_path, f"{nf[:-4]}_sift.npy"), sbj_data)  # into_id_sift.npy


class DataFilter:
    """滤波"""

    def __init__(self, origin_root_path, save_root_path):
        """
        :param origin_root_path: 原始数据根目录
        :param save_root_path: 存储根目录
        """
        self.origin_root_path = origin_root_path
        self.save_root_path = save_root_path

    def ecg_filter(self, ecg_data):
        """心电信号滤波"""
        # 带通滤波器即可
        # 原始采样率125，无失真滤波频率至少为250
        return heartpy.filter_signal(ecg_data, cutoff=[1.0, 35.0], sample_rate=250, order=3, filtertype='bandpass')

    def abp_filter(self, abp_data):
        return signal.savgol_filter(abp_data, 11, 1, mode='nearest')

    def exec_filter(self):
        """滤波"""
        # ppg用小波变换
        # ecg用带通滤波器
        # abp用平滑滤波
        for cur, dirs, files in os.walk(self.origin_root_path):
            sbj_id = cur.split("\\")[-1]  # p000107
            npy_files = fnmatch.filter(files, "*sift.npy")

            print(f"filtering {sbj_id} ...")
            for nf in npy_files:
                sift_data = get_data(os.path.join(cur, nf))
                sift_wav = sift_data['wav_data']  # (N, C, L) == (N, 3, 3000)
                # 病人全部数据
                sbj_data = {'gender': sift_data['gender'], 'age': sift_data['age'], 'height': sift_data['height'],
                            'weight': sift_data['weight'], 'wav_data': None, 'into_date': sift_data['into_date']}

                wav_data = []
                # print(sift_wav.shape)
                for i in range(sift_wav.shape[0]):
                    # print(sift_wav[i][0].shape)
                    # print(sift_wav[i][1].shape)
                    ppg_after = PPGFilter(sift_wav[i][0]).denoise()
                    abp_after = self.abp_filter(sift_wav[i][1])
                    ecg_after = self.ecg_filter(sift_wav[i][2])

                    tmp = [ppg_after, abp_after, ecg_after]
                    wav_data.append(tmp)

                save_path = os.path.join(self.save_root_path, sbj_id)
                if not os.path.exists(save_path):
                    os.mkdir(save_path)
                sbj_data['wav_data'] = np.asarray(wav_data)
                # print(sbj_data['wav_data'].shape)  # (N, 3, 3000)
                np.save(os.path.join(save_path, f"{nf[:-9]}_filter.npy"), sbj_data)


class PostProcess:
    """
    后处理，重组数据格式，筛选生理信息
    """

    def __init__(self, origin_root_path, save_root_path):
        self.origin_root_path = origin_root_path
        self.save_root_path = save_root_path

    def recombination(self, need_hw=False):
        # 对滤波后的数据进行重组，使之适合训练
        # 两种选择：
        # 1. 包含性别年龄身高体重
        # 2. 只包含性别年龄
        for cur, dirs, files in os.walk(self.origin_root_path):
            sbj_id = cur.split("\\")[-1]  # p000107
            npy_files = fnmatch.filter(files, "*filter.npy")

            print(f"post processing {sbj_id} ...")
            for nf in npy_files:
                filter_data = get_data(os.path.join(cur, nf))
                gender, age = filter_data['gender'], filter_data['age']
                height, weight = filter_data['height'], filter_data['weight']

                # 编码: 男性为1，女性为0
                gender = 1 if gender == 'M' else 0
                # age = age / 100  # 归一化
                # 这里保持原始输入，使用NN提取特征后输出scale到0-1之间，即使用sigmoid

                # 是否需要身高体重信息
                if need_hw:
                    # 为0或者None都不符合条件
                    if not height or not weight:
                        continue
                    else:
                        phy_data = np.asarray([gender, age, height, weight])  # 病人的生理数据
                else:
                    phy_data = np.asarray([gender, age])  # 病人的生理数据

                # 病人全部数据
                sbj_data = {'phy_data': phy_data, 'wav_data': filter_data['wav_data'],
                            'into_date': filter_data['into_date']}

                save_path = os.path.join(self.save_root_path, sbj_id)
                if not os.path.exists(save_path):
                    os.mkdir(save_path)
                # print(sbj_data['wav_data'].shape)
                np.save(os.path.join(save_path, f"{nf[:7]}_recmb.npy"), sbj_data)


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
        arr_sig = np.asarray(signal)
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
    
    def quality_assess(self, wav, sbp, dbp):
        """
        质量评估
        """
        wav = np.asarray(wav)
        PPG_THD = int(wav.shape[1] * 0.03)
        ECG_THD = int(wav.shape[1] * 0.03)

        # 超过范围的不要
        if sbp < 60 or sbp > 200 or dbp < 20 or dbp > 130:
            return False

        wav_diff = np.diff(wav, axis=1)
        zero_nums = np.sum(wav_diff == 0, axis=1)
        nan_nums = np.sum(np.isnan(wav_diff))

        if nan_nums > 0 or zero_nums[0] > PPG_THD or zero_nums[1] > ECG_THD:
            return False

        return True

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
        interval = sample_num * self.fs  # 5表示5秒
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

            if not self.quality_assess(tmp, sbp, dbp):
                continue
            # MIMIC-III: SBP: 55-195  DBP: 25-120
            # MIMIC-II: SBp: 80-180  DBP: 50-110

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


class MyDatasets:
    def __init__(self, root_path, save_path):
        """
        root_path: 存放滤波数据npy文件的根目录
        save_path: 存储路径
        """
        self.root_path = root_path
        self.save_path = save_path

    def random_sample(self, wav_data, bp_data, need_sample):
        # 随机选择切分好的数据中的若干个样本
        # print(wav_data.shape)  # (N, 3, 3000)
        # print(bp_data.shape)   # (N, 2)
        sample_num = wav_data.shape[0]  # 样本数
        sample_index = np.random.choice(sample_num, need_sample, replace=False)

        # 采样
        res_wav = wav_data[sample_index]
        res_bp = bp_data[sample_index]
        # print(res_wav.shape)  # (need_sample, 3, 3000)
        # print(res_bp.shape)   # (need_sample, 2)

        return res_wav, res_bp

    def create_wave_ga(self, feat_num=10, overlap=5, need_sample=400):
        # wave_data, gender, age
        """
        feat_num: 切分的特征数量
        overlap: 重叠数量
        need_sample: 需要的切片样本数量, 400 <=> 400 * 10 / 3600 = 1.11小时, 500就是1.38小时
        """
        # 先定义变量
        all_wav_data = []  # 波形数据，(N, L, C)
        all_bp_data = []   # 血压数据，(N, 2)
        all_phy_data = []  # 生理数据，(N, 2)
        judge_list = [self.root_path.split("\\")[-1][:3], 'p00']  # part 名称

        for cur, dirs, files in os.walk(self.root_path):
            # 因为文件夹下只有npy文件，因此遍历到的文件一定是npy_files
            # 若有其他文件，请解注释下面这句话
            npy_files = fnmatch.filter(files, '*.npy')

            # 分块存储
            # 否则需要一块较大的内存才能一次性存下来(至少64GB)
            # 这是一种后处理的策略，因此最后一个part不会在循环内被处理
            sbj_id = cur.split('\\')[-1]
            part_name = sbj_id[:3]  # p00
            if part_name not in judge_list:
                judge_list.append(part_name)

                all_wav_data = np.concatenate(all_wav_data, axis=0)
                all_bp_data = np.concatenate(all_bp_data, axis=0)
                all_phy_data = np.concatenate(all_phy_data, axis=0)

                np.save(os.path.join(self.save_path, f"{judge_list[-2]}_wav_data.npy"), all_wav_data)
                np.save(os.path.join(self.save_path, f"{judge_list[-2]}_bp_data.npy"), all_bp_data)
                np.save(os.path.join(self.save_path, f"{judge_list[-2]}_ga_data.npy"), all_phy_data)
                print(f"{judge_list[-2]} wav shape: {all_wav_data.shape}")
                print(f"{judge_list[-2]} bp shape: {all_bp_data.shape}")
                print(f"{judge_list[-2]} ga shape: {all_phy_data.shape}")

                all_wav_data = tuple()
                all_bp_data = tuple()
                all_phy_data = tuple()

            print(f"processing {sbj_id} ...")
            for nf in npy_files:
                wav_data = tuple()  # 一次住院的波形数据
                bp_data = tuple()  # 血压数据

                npy_path = os.path.join(cur, nf)
                # (N, C, L)
                # (样本数，通道数, 片段长度)
                sbj_data = get_data(npy_path)  # 一次住院的所有数据，可以认为是一个病例的数据
                clinical_data = sbj_data['wav_data']  # (N, C, L)
                person_data = sbj_data['phy_data']    # (2, )

                my_split = SplitData(fs=125)
                for i in tqdm(range(clinical_data.shape[0])):
                    ppg_data = clinical_data[i][0]  # ith sample: 0 channel
                    abp_data = clinical_data[i][1]
                    ecg_data = clinical_data[i][2]

                    intv_wav_data, intv_bp_data = my_split.split_by_time(wav=[ppg_data, ecg_data],
                                                                         abp=abp_data,
                                                                         sample_num=feat_num,
                                                                         overlap=overlap)
                    # 还是按照时间切片更好
                    # 按照特征切片会引入fake data，降低模型可解释性

                    if intv_bp_data.size == 0:
                        # 必须判断size
                        # counter case: [[]] , shape: (1, 0)
                        continue  # 该片段无法进行10秒切片

                    wav_data += (intv_wav_data,)
                    bp_data += (intv_bp_data,)

                if len(wav_data) == 0:
                    # 应该可以去掉
                    continue  # p083608 有问题

                wav_data = np.concatenate(wav_data, axis=0)  # (S, C, L)
                bp_data = np.concatenate(bp_data, axis=0)    # (S, 2)
                if len(bp_data) >= need_sample:  # 超过一定数量才存储
                    # print(f"{sbj_id} is saving ...")
                    sample_wav, sample_bp = self.random_sample(wav_data, bp_data, need_sample)  # 随机选择样本
                    phy_data = np.tile(person_data, (need_sample, 1))  # need_sample行，1列

                    # sample_wav: (need_sample, 2, 1250) --> ppg, ecg
                    # sample_bp: (need_sample, 2)        --> sbp, dbp
                    # phy_data: (need_sample, 2)         --> gender, age

                    all_wav_data += (sample_wav, )
                    all_bp_data += (sample_bp, )
                    all_phy_data += (phy_data, )

        if len(all_bp_data) > 0:
            # 其实这里的判断没啥必要
            # 一个part的数据为0的概率太小了
            all_wav_data = np.concatenate(all_wav_data, axis=0)
            all_bp_data = np.concatenate(all_bp_data, axis=0)
            all_phy_data = np.concatenate(all_phy_data, axis=0)
            # 存储最后一个part
            np.save(os.path.join(self.save_path, f"{judge_list[-1]}_wav_data.npy"), all_wav_data)
            np.save(os.path.join(self.save_path, f"{judge_list[-1]}_bp_data.npy"), all_bp_data)
            np.save(os.path.join(self.save_path, f"{judge_list[-1]}_ga_data.npy"), all_phy_data)
            print(f"{judge_list[-1]} wav shape: {all_wav_data.shape}")
            print(f"{judge_list[-1]} bp shape: {all_bp_data.shape}")
            print(f"{judge_list[-1]} ga shape: {all_phy_data.shape}")


def create_h5():
    path = r"I:\dataset\MIMIC-III\bp_torch_final"
    f = h5py.File(os.path.join(path, "bp_dataset_iii.h5"), "a")  # append mode
    # for i in range(10):
    for i in range(3, 7):
        wav_data = np.load(os.path.join(path, f"p0{i}_wav_data.npy"))
        ga_data = np.load(os.path.join(path, f"p0{i}_ga_data.npy")).astype(np.float32)
        bp_data = np.load(os.path.join(path, f"p0{i}_bp_data.npy"))

        ga_data[:, 1] = ga_data[:, 1] / 100  # 还是要先归一化的

        if i == 3:
            f.create_dataset("wav", data=wav_data, compression=6, chunks=True, maxshape=(None, 2, 1250))
            f.create_dataset("ga", data=ga_data, compression=6, chunks=True, maxshape=(None, 2))  # 6th level gzip
            f.create_dataset("bp", data=bp_data, compression=6, chunks=True, maxshape=(None, 2))
        else:
            f["wav"].resize(f["wav"].shape[0] + wav_data.shape[0], axis=0)
            f["wav"][-wav_data.shape[0]:] = wav_data

            f["ga"].resize(f["ga"].shape[0] + ga_data.shape[0], axis=0)
            f["ga"][-ga_data.shape[0]:] = ga_data

            f["bp"].resize(f["bp"].shape[0] + bp_data.shape[0], axis=0)
            f["bp"][-bp_data.shape[0]:] = bp_data

        print(f"add {i}th , chunk wav shape: {f['wav'].shape}")

    all_bp = f["bp"]
    bp_df = pd.DataFrame(all_bp)
    bp_df.to_csv(os.path.join(path, "bp.csv"))

    f.close()


def get_constant_para():
    """
    获取常数参数
    feat_num = 10, overlap = 0
    """
    root_path = r"I:\dataset\MIMIC-III\by_time_8"
    # root_path = r"D:\Projects\Py_projects\blood_pressure\Blood_Pressure\my_bp_torch\mimic_iii\dataset"
    dset = h5py.File(os.path.join(root_path, "bp_dataset_iii.h5"))
    bp = dset['bp'][:, 0]
    # print(len(bp))

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


def get_data(npy_path):
    # 读取npy
    # sbj_data = {'gender': None, 'age': None, 'height': None,
    #             'weight': None, 'wav_data': None, 'into_date': None}
    all_data = np.load(npy_path, allow_pickle=True).item()
    # print(type(all_data.item()))   # dict
    # clinical_data = all_data.item()['wav_data']  # 获取全部数据

    return all_data


def my_train_test_split():
    # inter-patient做法
    # 按照病例分开，而不是按照样本数划分数据集
    # 按照病例，给定1000人，800人做训练集，200人做测试集(本函数作用)
    # 按照样本，给定1000样本，800样本做训练集，200样本做测试集
    path = r"H:\datasets\1000sample_795p"
    all_wav_data = np.load(os.path.join(path, "all_wav_data.npy"))
    all_sbp_data = np.load(os.path.join(path, "all_sbp_data.npy"))
    all_ga_data = np.load(os.path.join(path, "all_ga_data.npy"))

    patient_num = all_sbp_data.shape[0]    # 患者数
    test_set_num = int(patient_num * 0.2)  # 测试集患者数
    test_index = np.random.choice(patient_num, test_set_num, replace=False)
    train_index = np.delete(np.arange(patient_num), test_index)

    train_wav = all_wav_data[train_index, :]
    test_wav = all_wav_data[test_index, :]

    train_ga = all_ga_data[train_index, :]
    test_ga = all_ga_data[test_index, :]

    train_sbp = all_sbp_data[train_index, :]
    test_sbp = all_sbp_data[test_index, :]

    train_wav_final = tuple()
    train_ga_final = tuple()
    train_sbp_final = tuple()
    for i in range(train_wav.shape[0]):
        train_wav_final += (train_wav[i], )
        train_ga_final += (train_ga[i], )
        train_sbp_final += (train_sbp[i], )

    test_wav_final = tuple()
    test_ga_final = tuple()
    test_sbp_final = tuple()
    for j in range(test_wav.shape[0]):
        test_wav_final += (test_wav[j], )
        test_ga_final += (test_ga[j], )
        test_sbp_final += (test_sbp[j], )

    print("train_wav shape: ", train_wav.shape)
    print("train_ga shape: ", train_ga.shape)
    print("train_sbp shape: ", train_sbp.shape)
    print("test_wav shape: ", test_wav.shape)
    print("test_ga shape: ", test_ga.shape)
    print("test_sbp shape: ", test_sbp.shape)

    train_wav_final = np.concatenate(train_wav_final, axis=0)
    train_ga_final = np.concatenate(train_ga_final, axis=0)
    train_sbp_final = np.concatenate(train_sbp_final, axis=0)
    test_wav_final = np.concatenate(test_wav_final, axis=0)
    test_ga_final = np.concatenate(test_ga_final, axis=0)
    test_sbp_final = np.concatenate(test_sbp_final, axis=0)

    train_wav_final, train_ga_final, train_sbp_final = shuffle(train_wav_final,
                                                               train_ga_final,
                                                               train_sbp_final,
                                                               random_state=42)
    test_wav_final, test_ga_final, test_sbp_final = shuffle(test_wav_final,
                                                            test_ga_final,
                                                            test_sbp_final,
                                                            random_state=42)

    print("train_wav_final shape: ", train_wav_final.shape)
    print("train_ga_final shape: ", train_ga_final.shape)
    print("train_sbp_final shape: ", train_sbp_final.shape)
    print("test_wav_final shape: ", test_wav_final.shape)
    print("test_ga_final shape: ", test_ga_final.shape)
    print("test_sbp_final shape: ", test_sbp_final.shape)

    np.save("train_wav.npy", train_wav_final)
    np.save("train_ga.npy", train_ga_final)
    np.save("train_sbp.npy", train_sbp_final)
    np.save("test_wav.npy", test_wav_final)
    np.save("test_ga.npy", test_ga_final)
    np.save("test_sbp.npy", test_sbp_final)


def main():
    # 主函数
    # origin_npy_path: 读取hea和dat文件保存成的npy数据，每次住院记录有多个数据包
    # origin_sbj_path: 一次住院记录保存成一个包，结果仍是npy数据
    # sift_sbj_path: 筛选后的病人数据，每次住院记录有一个包，尺寸为N x 3000 x 3，N为样本数，3000为片段长度，3为通道数
    # filter_sbj_path: 滤波后的病人数据，形式同筛选后的数据
    # recomb_path: 重组后的病人数据，生理数据为1d array: [gender, age, (height, weight)]
    # 性别男1女0，均为标准单位，身高体重可选，原因是数据量较少

    # DataExtract("origin_npy_path", "origin_sbj_path").extract("csv_path")
    # DataSift("origin_sbj_path", "sift_sbj_path").data_sift()
    # DataFilter("sift_sbj_path", "filter_sbj_path").exec_filter()
    # PostProcess(r"filter_sbj_path", r"recomb_path").recombination(need_hw=True)
    # MyDatasets("recomb_path").create_datasets(feat_num=10, overlap_num=5, need_bp_type='SBP', need_sample=500)

    # DataExtract(r"H:\last_part", r"G:\process_data").extract(r"H:\data.csv")
    # DataSift(r"H:\extract_data", r"I:\sift_data").data_sift(interval=3000)
    # DataFilter(r"I:\sift_data", r"I:\filter_data").exec_filter()
    # PostProcess(r"I:\filter_data", r"I:\recmb_data").recombination(need_hw=False)  # 保持为False，不然数据太少
    MyDatasets(r"I:\recmb_data", r"I:\dataset\MIMIC-III\bp_torch_final").create_wave_ga(feat_num=10, overlap=0, need_sample=1000)


if __name__ == "__main__":
    # main()
    create_h5()
    # get_constant_para()
