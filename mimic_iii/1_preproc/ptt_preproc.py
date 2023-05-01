import os
import numpy as np
import matplotlib.pyplot as plt
import h5py
import pandas as pd
from sklearn.model_selection import train_test_split
from wfdb.processing import xqrs_detect


def create_norm_sub(start, end, mu, sigma):
    """
    创建正态子集
    """
    # data_path = r"D:\Projects\Py_projects\blood_pressure\Blood_Pressure\bp_torch_final\mimic_ii\dataset"
    data_path = r"I:\dataset\MIMIC-III\by_time_8"
    dset = h5py.File(os.path.join(data_path, "bp_dataset_iii.h5"))

    save_path = r"D:\Projects\Py_projects\blood_pressure\Blood_Pressure\bp_torch_final\mimic_iii\norm_dataset"
    all_bp = np.asarray(dset['bp'])
    all_wav = np.asarray(dset['wav'])

    old_train_wav, old_test_wav, old_train_bp, old_test_bp = train_test_split(all_wav, all_bp,
                                                              test_size=0.3,
                                                              shuffle=True,
                                                              random_state=0)
    
    train_wav, test_wav, train_bp, test_bp = train_test_split(old_test_wav, old_test_bp,
                                                              test_size=0.4,
                                                              shuffle=True,
                                                              random_state=0)

    test_dbp = test_bp[:, 1]

    sub_wav = tuple()
    sub_bp = tuple()

    for i in range(start, end + 1):
        print(f"processing interval {i}...")
        need_sample = int(norm_pdf(i, mu, sigma))
        cond = np.where((test_dbp >= i) & (test_dbp < i + 1))  # 处于(i, i+1)区间内的所有样本
        aim_bp = test_bp[cond]
        aim_wav = test_wav[cond]

        sample_num = aim_wav.shape[0]
        if sample_num > need_sample: 
            sample_wav, sample_bp = random_sample(aim_wav, aim_bp, need_sample)
        else:
            sample_wav, sample_bp = aim_wav, aim_bp

        sub_wav += (sample_wav, )
        sub_bp += (sample_bp, )
    
    sub_wav = np.concatenate(sub_wav, axis=0)
    sub_bp = np.concatenate(sub_bp, axis=0)

    np.save(os.path.join(save_path, "train_wav.npy"), train_wav)
    np.save(os.path.join(save_path, "train_bp.npy"), train_bp)
    np.save(os.path.join(save_path, "sub_wav.npy"), sub_wav)
    np.save(os.path.join(save_path, "sub_bp.npy"), sub_bp)

    bp_df = pd.DataFrame(sub_bp)
    bp_df.to_csv(os.path.join(save_path, "sub_bp.csv"))

    print(sub_wav.shape)
    print(sub_bp.shape)


def norm_pdf(x, mu, sigma): 
    # 计算概率密度
    pdf = 1/(sigma*np.sqrt(2*np.pi))*np.exp(-(x-mu)**2/(2*sigma**2)) * 20000
    # 75000 = 3000 / 0.04
    return pdf


def create_bal_sub(start, end, need_sample=100):
    """
    创建平衡均匀分布子集
    """
    data_path = r"D:\Projects\Py_projects\blood_pressure\Blood_Pressure\bp_torch_final\mimic_ii\bal_dataset"
    dset = h5py.File(os.path.join(data_path, "bp_dataset.h5"))
    all_bp = np.asarray(dset['bp'])
    all_wav = np.asarray(dset['wav'])

    train_wav, test_wav, train_bp, test_bp = train_test_split(all_wav, all_bp,
                                                              test_size=0.4,
                                                              shuffle=True,
                                                              random_state=0)
    test_dbp = test_bp[:, 1]

    sub_wav = tuple()
    sub_bp = tuple()

    for i in range(start, end + 1):
        print(f"processing interval {i}...")
        cond = np.where((test_dbp >= i) & (test_dbp < i + 1))  # 处于(i, i+1)区间内的所有样本
        aim_bp = test_bp[cond]
        aim_wav = test_wav[cond]

        sample_num = aim_wav.shape[0]
        if sample_num > need_sample: 
            sample_wav, sample_bp = random_sample(aim_wav, aim_bp, need_sample)
        else:
            sample_wav, sample_bp = aim_wav, aim_bp

        sub_wav += (sample_wav, )
        sub_bp += (sample_bp, )
    
    sub_wav = np.concatenate(sub_wav, axis=0)
    sub_bp = np.concatenate(sub_bp, axis=0)

    np.save(os.path.join(data_path, "train_wav.npy"), train_wav)
    np.save(os.path.join(data_path, "train_bp.npy"), train_bp)
    np.save(os.path.join(data_path, "sub_wav.npy"), sub_wav)
    np.save(os.path.join(data_path, "sub_bp.npy"), sub_bp)

    bp_df = pd.DataFrame(sub_bp)
    bp_df.to_csv(os.path.join(data_path, "sub_bp.csv"))

    print("sub wav shape", sub_wav.shape)
    print("sub bp shape", sub_bp.shape)


def random_sample(wav_data, bp_data, need_sample):
    # 随机选择切分好的数据中的若干个样本
    # print(wav_data.shape)  # (N, 3, 1250)
    # print(bp_data.shape)   # (N, 2)
    sample_num = wav_data.shape[0]  # 样本数
    sample_index = np.random.choice(sample_num, need_sample, replace=False)

    # 采样
    res_wav = wav_data[sample_index]
    res_bp = bp_data[sample_index]
    # print(res_wav.shape)  # (need_sample, 3, 1250)
    # print(res_bp.shape)   # (need_sample, 2)

    return res_wav, res_bp


def create_h5():
    # mimic-ii preprocessed part data path
    path = r"D:\Projects\Py_projects\blood_pressure\Blood_Pressure\bp_torch_final\mimic_iii\norm_dataset"
    train_wav = np.load(os.path.join(path, "train_wav.npy"))
    train_bp = np.load(os.path.join(path, "train_bp.npy"))
    wav_data = np.load(os.path.join(path, "sub_wav.npy"))
    bp_data = np.load(os.path.join(path, "sub_bp.npy"))

    wav_val, wav_test, bp_val, bp_test = train_test_split(wav_data, bp_data,
                                                          test_size=0.5,
                                                          random_state=0,
                                                          shuffle=True)
    f_train = h5py.File(os.path.join(path, "bp_train.h5"), "w")  # append mode
    f_train.create_dataset("wav", data=train_wav, compression=6, chunks=True, maxshape=(None, 2, 1250))
    f_train.create_dataset("bp", data=train_bp, compression=6, chunks=True, maxshape=(None, 2))

    f_val = h5py.File(os.path.join(path, "bp_val.h5"), "w")  # append mode
    f_val.create_dataset("wav", data=wav_val, compression=6, chunks=True, maxshape=(None, 2, 1250))
    f_val.create_dataset("bp", data=bp_val, compression=6, chunks=True, maxshape=(None, 2))

    f_test = h5py.File(os.path.join(path, "bp_test.h5"), "w")  # append mode
    f_test.create_dataset("wav", data=wav_test, compression=6, chunks=True, maxshape=(None, 2, 1250))
    f_test.create_dataset("bp", data=bp_test, compression=6, chunks=True, maxshape=(None, 2))

    test_bp = f_test["bp"]
    print("test_bp shape: ", test_bp.shape)
    test_bp_df = pd.DataFrame(test_bp)
    test_bp_df.to_csv(os.path.join(path, "bp_test.csv"))

    train_bp = f_train["bp"]
    print("train bp shape: ", train_bp.shape)
    train_bp_df = pd.DataFrame(train_bp)
    train_bp_df.to_csv(os.path.join(path, "bp_train.csv"))

    f_train.close()
    f_test.close()
    f_val.close()


if __name__ == "__main__":
    create_norm_sub(50, 109, 80, 10)
    # create_bal_sub(50, 109, need_sample=200)
    # create_h5()

