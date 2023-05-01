import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import os
from scipy.stats import gmean
from scipy.optimize import curve_fit
from collections import defaultdict


def compute_metrics(y_hat, y_true, y_train, save_path, configs):
    """
    y_hat: predict label, [sbp, dbp]
    y_true: true label, [sbp, dbp]
    y_train: train label, [sbp, dbp]
    """
    length = y_true.shape[0]

    me = np.sum((y_hat - y_true), axis=0) / length
    mae = np.sum(np.abs(y_hat - y_true), axis=0) / length
    std = np.std(np.abs(y_hat - y_true), axis=0, ddof=1)
    sbp_r = np.corrcoef(y_hat[:, 0], y_true[:, 0])[0][1]
    dbp_r = np.corrcoef(y_hat[:, 1], y_true[:, 1])[0][1]

    sbp_shot_dict = shot_metrics(y_hat[:, 0], y_true[:, 0], y_train[:, 0],
                                 many_shot=configs["sbp_many"], low_shot=configs["sbp_low"])
    dbp_shot_dict = shot_metrics(y_hat[:, 1], y_true[:, 1], y_train[:, 1],
                                 many_shot=configs["dbp_many"], low_shot=configs["dbp_low"])

    with open(os.path.join(save_path, 'bp_metric.txt'), "w+") as f:
        # This mode will clear the original content each time.
        f.write("[SBP]\n")
        f.write(f"ME: {me[0]:.2f}\t")
        f.write(f"MAE: {mae[0]:.2f}\t")
        f.write(f"STD: {std[0]:.2f}\t")
        f.write(f"r: {sbp_r:.4f}\n")
        for sk, sv in sbp_shot_dict.items():
            f.write(f"{sk}: \t")
            for sk1, sv1 in sv.items():
                f.write(f"{sk1}: {sv1:.2f}\t")
            f.write("\n")

        f.write("\n\n")

        f.write("[DBP]\n")
        f.write(f"ME: {me[1]:.2f}\t")
        f.write(f"MAE: {mae[1]:.2f}\t")
        f.write(f"STD: {std[1]:.2f}\t")
        f.write(f"r: {dbp_r:.4f}\n")
        for dk, dv in dbp_shot_dict.items():
            f.write(f"{dk}: \t")
            for dk1, dv1 in dv.items():
                f.write(f"{dk1}: {dv1:.2f}\t")
            f.write("\n")


def plot_hist(history, save_path='.', font_size=30):
    """
    history: [train_loss, val_loss]
    """
    train_loss, val_loss = history

    plt.figure(figsize=(15, 12))
    plt.plot(train_loss, linewidth=1, label="train_loss")
    plt.plot(val_loss, linewidth=1, label="val_loss")
    plt.xlabel("Epoch", fontsize=font_size)
    plt.ylabel("Loss", fontsize=font_size)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)

    plt.savefig(os.path.join(save_path, "history.png"))


def plot_corr(y_hat, y_true, bp='SBP', save_path='.', font_size=30):
    """
    correlation figure
    bp: "SBP" or "DBP"
    """
    def func(x, a, b):
        return a * x + b

    fig, ax = plt.subplots(figsize=(15, 15))
    ax.plot(y_true, y_true, 'k:', linewidth=5, label='y=x')  # true value
    ax.scatter(y_true, y_hat, s=1, c='b', marker='.', label='data')  # prediction
    # 添加趋势线
    popt1, _ = curve_fit(func, y_true, y_hat)
    prediction_fit = func(y_true, *popt1)

    ax.plot(y_true, prediction_fit, 'r--', linewidth=5, label='fit curve')  # curve fit

    r = np.corrcoef(y_true, y_hat)
    # print("r = %.4f" % r[0][1])
    ax.text(0.3, 0.8, "r = %.4f" % r[0][1], fontsize=font_size, transform=ax.transAxes)

    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    ax.set_xlabel(f'Reference {bp}(mmHg)', fontsize=font_size)
    ax.set_ylabel(f'Estimated {bp}(mmHg)', fontsize=font_size)
    # ax.legend()
    fig.savefig(os.path.join(save_path, f'{bp}_corr.png'))


def plot_ba(y_hat, y_true, bp='SBP', save_path='.', font_size=30):

    mean = np.mean([y_hat, y_true], axis=0)
    diff = y_hat - y_true  # Difference between data1 and data2
    md = np.mean(diff)     # Mean of the difference
    sd = np.std(diff, axis=0)  # Standard deviation of the difference
    up_bound = md + 1.96 * sd
    low_bound = md - 1.96 * sd
    offset = (up_bound - low_bound) / 100 * 1.5

    fig, ax = plt.subplots(figsize=(15, 15))
    trans = transforms.blended_transform_factory(ax.transAxes, ax.transData)

    ax.scatter(mean, diff, s=1, c='b', marker='.')
    ax.axhline(md, color='g', linestyle='--', linewidth=5)
    ax.axhline(up_bound, color='r', linestyle='--', linewidth=5)
    ax.axhline(low_bound, color='r', linestyle='--', linewidth=5)

    ax.text(0.98, md + offset, 'Mean', fontsize=font_size, color='r', ha='right', va='bottom', transform=trans)
    ax.text(0.98, md - offset, f'{md:.2f}', fontsize=font_size, color='r', ha='right', va='top', transform=trans)
    ax.text(0.98, up_bound + offset, '+1.96 STD', fontsize=font_size, ha='right', va='bottom', transform=trans)
    ax.text(0.98, up_bound - offset, f'{up_bound:.2f}', fontsize=font_size, ha='right', va='top', transform=trans)
    ax.text(0.98, low_bound + offset, '-1.96 STD', fontsize=font_size, ha='right', va='bottom', transform=trans)
    ax.text(0.98, low_bound - offset, f'{low_bound:.2f}', fontsize=font_size, ha='right', va='top', transform=trans)

    # plt.title('Bland Altman')
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    ax.set_xlabel(f"Average {bp}(mmHg)", fontsize=font_size)
    ax.set_ylabel(f"Difference {bp}(mmHg)", fontsize=font_size)
    fig.savefig(os.path.join(save_path, f'{bp}_bland_altman.png'))


def shot_metrics(y_hat, y_true, y_train, many_shot=6000, low_shot=1000):
    y_train = y_train.astype(int)
    y_hat = y_hat.astype(int)
    y_true = y_true.astype(int)

    train_class_count, test_class_count = [], []
    mse_per_class, l1_per_class, me_per_class, l1_all_per_class = [], [], [], []
    for l in np.unique(y_true):
        train_class_count.append(len(y_train[y_train == l]))
        test_class_count.append(len(y_true[y_true == l]))
        mse_per_class.append(np.sum((y_hat[y_true == l] - y_true[y_true == l]) ** 2))
        l1_per_class.append(np.sum(np.abs(y_hat[y_true == l] - y_true[y_true == l])))
        me_per_class.append(np.sum(y_hat[y_true == l] - y_true[y_true == l]))
        l1_all_per_class.append(np.abs(y_hat[y_true == l] - y_true[y_true == l]))

    many_shot_mse, median_shot_mse, low_shot_mse = [], [], []
    many_shot_l1, median_shot_l1, low_shot_l1 = [], [], []
    many_shot_me, median_shot_me, low_shot_me = [], [], []
    many_shot_gmean, median_shot_gmean, low_shot_gmean = [], [], []
    many_shot_cnt, median_shot_cnt, low_shot_cnt = [], [], []

    for i in range(len(train_class_count)):
        if train_class_count[i] > many_shot:
            many_shot_mse.append(mse_per_class[i])
            many_shot_l1.append(l1_per_class[i])
            many_shot_me.append(me_per_class[i])
            many_shot_gmean += list(l1_all_per_class[i])
            many_shot_cnt.append(test_class_count[i])
        elif train_class_count[i] < low_shot:
            low_shot_mse.append(mse_per_class[i])
            low_shot_l1.append(l1_per_class[i])
            low_shot_me.append(me_per_class[i])
            low_shot_gmean += list(l1_all_per_class[i])
            low_shot_cnt.append(test_class_count[i])
        else:
            median_shot_mse.append(mse_per_class[i])
            median_shot_l1.append(l1_per_class[i])
            median_shot_me.append(me_per_class[i])
            median_shot_gmean += list(l1_all_per_class[i])
            median_shot_cnt.append(test_class_count[i])

    many_shot_gmean = np.hstack(many_shot_gmean)
    many_shot_gmean[many_shot_gmean == 0] += 1
    median_shot_gmean = np.hstack(median_shot_gmean)
    median_shot_gmean[median_shot_gmean == 0] += 1
    low_shot_gmean = np.hstack(low_shot_gmean)
    low_shot_gmean[low_shot_gmean == 0] += 1

    shot_dict = defaultdict(dict)
    shot_dict['many']['mse'] = np.sum(many_shot_mse) / np.sum(many_shot_cnt)
    shot_dict['many']['l1'] = np.sum(many_shot_l1) / np.sum(many_shot_cnt)
    shot_dict['many']['me'] = np.sum(many_shot_me) / np.sum(many_shot_cnt)
    shot_dict['many']['gmean'] = gmean(np.hstack(many_shot_gmean), axis=None).astype(float)
    shot_dict['median']['mse'] = np.sum(median_shot_mse) / np.sum(median_shot_cnt)
    shot_dict['median']['l1'] = np.sum(median_shot_l1) / np.sum(median_shot_cnt)
    shot_dict['median']['me'] = np.sum(median_shot_me) / np.sum(median_shot_cnt)
    shot_dict['median']['gmean'] = gmean(np.hstack(median_shot_gmean), axis=None).astype(float)
    shot_dict['low']['mse'] = np.sum(low_shot_mse) / np.sum(low_shot_cnt)
    shot_dict['low']['l1'] = np.sum(low_shot_l1) / np.sum(low_shot_cnt)
    shot_dict['low']['me'] = np.sum(low_shot_me) / np.sum(low_shot_cnt)
    shot_dict['low']['gmean'] = gmean(np.hstack(low_shot_gmean), axis=None).astype(float)

    return shot_dict


def post_process(data_path, configs):
    y_hat = np.load(os.path.join(data_path, "y_hat.npy"))
    y_true = np.load(os.path.join(data_path, "y_true.npy"))
    y_train = np.load(os.path.join(data_path, "y_train.npy"))
    history = np.load(os.path.join(data_path, "history.npy"))

    plot_corr(y_hat[:, 0], y_true[:, 0], "SBP", save_path=data_path, font_size=30)
    plot_corr(y_hat[:, 1], y_true[:, 1], "DBP", save_path=data_path, font_size=30)

    plot_ba(y_hat[:, 0], y_true[:, 0], "SBP", save_path=data_path, font_size=30)
    plot_ba(y_hat[:, 1], y_true[:, 1], "DBP", save_path=data_path, font_size=30)

    plot_hist(history, save_path=data_path, font_size=30)
    compute_metrics(y_hat, y_true, y_train, save_path=data_path, configs=configs)


if __name__ == "__main__":
    pass

