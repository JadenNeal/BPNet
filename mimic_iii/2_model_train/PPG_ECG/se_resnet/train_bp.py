from bp_net import bpnet18
from post_process import post_process
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch.backends.cudnn as cudnn
import numpy as np
import torch
import h5py
import os


SEED = 666
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
cudnn.deterministic = True
cudnn.benchmark = True

configs = {
    "epochs": 100,
    "batch_size": 64,
    "model_width": 64,
    "device_ids": [0, 1],
    "early_stop_patience": 5,
    "test_size": 0.2,
    "result_save_path": "./result",
    # boxcox trans
    "sbp_many": 8000,
    "sbp_low": 1000,
    "dbp_many": 12000,
    "dbp_low": 3000,
    "bias": 605.0
}

if not os.path.exists(configs['result_save_path']):
    os.mkdir(configs['result_save_path'])

device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
bp_model = bpnet18(model_width=configs['model_width']).to(device=device)
bp_model = torch.nn.DataParallel(bp_model, device_ids=configs['device_ids'])


class BPDataset(Dataset):
    def __init__(self, wav_data, bp_data):
        # initialization
        ppg = np.expand_dims(wav_data[:, 0, :], axis=1)
        ecg = np.expand_dims(wav_data[:, 1, :], axis=1)
        self.ppg = torch.as_tensor(ppg, dtype=torch.float32)
        self.ecg = torch.as_tensor(ecg, dtype=torch.float32)

        self.bp = torch.as_tensor(bp_data, dtype=torch.float32)  # (N, 2)

    def __len__(self):
        return len(self.bp)

    def __getitem__(self, idx):
        return self.ppg[idx], self.ecg[idx], self.bp[idx]


class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        # print("val_loss={}".format(val_loss))
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'best_model.pth')
        self.val_loss_min = val_loss


def train(train_x, train_y):
    """
    训练函数
    :param train_x: 归一化后的 x_train 数据, npy格式
    :param train_y: 归一化后的 y_train 数据, npy格式
    :return: None
    """
    EPOCHS = configs['epochs']
    batch_size = configs['batch_size']
    criterion = torch.nn.MSELoss()  # mse loss
    optimizer = torch.optim.Adam(bp_model.parameters(), lr=0.001)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=10, eta_min=1e-8)
    # 这里的patience和keras中的不一样
    # torch中patience设置为2，实际上3个epoch的val_loss还没有下降才会在第3个epoch降低lr
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                           factor=0.2,
                                                           patience=1,
                                                           min_lr=1e-5,
                                                           verbose=True)
    early_stopping = EarlyStopping(patience=configs['early_stop_patience'], verbose=False)

    x_train, x_val, y_train, y_val = train_test_split(train_x,
                                                      train_y,
                                                      test_size=configs['test_size'],
                                                      shuffle=True,
                                                      random_state=0)

    train_set = BPDataset(x_train, y_train)
    train_loader = DataLoader(train_set, batch_size=batch_size)  # 训练集
    val_set = BPDataset(x_val, y_val)
    val_loader = DataLoader(val_set, batch_size=batch_size)  # 验证集

    # 开始训练
    train_loss = []  # 存储每个epoch的loss结果
    valid_loss = []
    val_sbp_loss = []
    val_dbp_loss = []
    for epoch in range(EPOCHS):
        # ================== train mode ================== #
        bp_model.train()
        train_epoch_loss = []  # 每个epoch的loss
        train_loop = tqdm(enumerate(train_loader, 0), total=len(train_loader), colour='blue')
        for i, [x1, x2, y] in train_loop:
            # print(x.shape)  # (batch_size, 2, 1250)
            # print(y.shape)  # (batch_size, 2)
            x1 = x1.to(device)
            x2 = x2.to(device)
            y = y.to(device)

            y_hat = bp_model(x1, x2)
            y_hat_0 = y_hat[0].squeeze(dim=1)
            y_hat_1 = y_hat[1].squeeze(dim=1)
            # print(len(y_pred))   # 2
            # print(type(y_pred))  # tuple

            # Compute and print loss
            sbp_loss = criterion(y_hat_0, y[:, 0])
            dbp_loss = criterion(y_hat_1, y[:, 1])
            tot_loss = sbp_loss + dbp_loss / (dbp_loss / sbp_loss).detach()  # total loss

            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            tot_loss.backward()
            optimizer.step()

            train_epoch_loss.append(tot_loss.item())

            # train 过程展示
            loss_dict = {"train_loss": tot_loss.item(), "sbp_loss": sbp_loss.item(), "dbp_loss": dbp_loss.item()}
            train_loop.set_description(f"Epoch {epoch + 1}/{EPOCHS}")
            train_loop.set_postfix(loss_dict)

        train_loss.append(np.average(train_epoch_loss))

        # ================== valid mode ================== #
        bp_model.eval()
        valid_epoch_loss = []
        sbp_epoch_loss = []
        dbp_epoch_loss = []
        with torch.no_grad():
            for i, [x1, x2, y] in enumerate(val_loader, 0):
                x1 = x1.to(device)
                x2 = x2.to(device)
                y = y.to(device)

                y_hat = bp_model(x1, x2)
                y_hat_0 = y_hat[0].squeeze(dim=1)
                y_hat_1 = y_hat[1].squeeze(dim=1)

                sbp_loss = criterion(y_hat_0, y[:, 0])
                dbp_loss = criterion(y_hat_1, y[:, 1])
                tot_loss = sbp_loss + dbp_loss / (dbp_loss / sbp_loss).detach()  # total loss

                sbp_epoch_loss.append(sbp_loss.item())
                dbp_epoch_loss.append(dbp_loss.item())
                valid_epoch_loss.append(tot_loss.item())

        val_sbp_loss.append(np.average(sbp_epoch_loss))
        val_dbp_loss.append(np.average(dbp_epoch_loss))
        valid_loss.append(np.average(valid_epoch_loss))
        print(
            f"val_loss: {valid_loss[-1]:.4f} val_sbp_loss: {val_sbp_loss[-1]:.8f} val_dbp_loss: {val_dbp_loss[-1]:.8f} "
            f"lr: {optimizer.state_dict()['param_groups'][0]['lr']:.8f}")

        early_stopping(valid_loss[-1], model=bp_model, path=configs['result_save_path'])
        if early_stopping.early_stop:
            print("========== Early Stopped ==========")
            break

        scheduler.step(valid_loss[-1])  # 学习率衰减

    hist = [train_loss, valid_loss]
    np.save(os.path.join(configs["result_save_path"], "history.npy"), hist)
    np.save(os.path.join(configs["result_save_path"], "y_train.npy"), train_y)


def test(test_x, test_y):
    test_set = BPDataset(test_x, test_y)
    test_loader = DataLoader(test_set, batch_size=configs['batch_size'])

    bp_model.eval()
    y_pred = tuple()  # 预测结果
    test_loop = tqdm(enumerate(test_loader, 0), total=len(test_loader), colour='yellow')
    with torch.no_grad():
        for i, [x1, x2, _] in test_loop:
            x1 = x1.to(device)
            x2 = x2.to(device)

            y_hat = bp_model(x1, x2)
            y_hat_0 = y_hat[0].squeeze(dim=1)
            y_hat_1 = y_hat[1].squeeze(dim=1)

            tmp = [y_hat_0.cpu().numpy(), y_hat_1.cpu().numpy()]
            y_pred += (tmp,)

            test_loop.set_description(f"testing process")

    # 存储结果
    y_pred = np.concatenate(y_pred, axis=1)
    y_pred = y_pred.T  # (N, 2)
    np.save(os.path.join(configs['result_save_path'], "y_hat.npy"), y_pred)
    np.save(os.path.join(configs['result_save_path'], "y_true.npy"), test_y)


def _prepare_data(path):
    dset = h5py.File(os.path.join(path, "bp_dataset_ii.h5"))
    wav = dset['wav']  # (N, 2, 1250)
    bp = dset['bp']  # (N, 2)
    X = np.asarray(wav)
    Y = np.asarray(bp)

    print("all X shape: ", X.shape)
    print("all Y shape: ", Y.shape)

    x_train, x_test, y_train, y_test = train_test_split(X, Y,
                                                        test_size=configs["test_size"],
                                                        shuffle=True,
                                                        random_state=0)
    print("load success ...")
    return x_train, x_test, y_train, y_test


def main():
    # 全部数据
    root_path = r"/data_new/lwc/mimic_ii/dataset_best"
    x_train, x_test, y_train, y_test = _prepare_data(root_path)

    print("=" * 15, "start training ...", "=" * 15)
    train(train_x=x_train, train_y=y_train)

    print("=" * 15, "start testing ...", "=" * 15)
    test(test_x=x_test, test_y=y_test)

    post_process(data_path=configs['result_save_path'], configs=configs)


if __name__ == '__main__':
    main()
