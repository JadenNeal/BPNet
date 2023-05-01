import torch
import torch.nn as nn
from torchinfo import summary
import torchviz


class BPNet(nn.Module):
    def __init__(self, model_width):
        super(BPNet, self).__init__()
        self.share_net = nn.GRU(input_size=2, hidden_size=model_width,
                                num_layers=3, batch_first=True)

        # SBP branch
        self.linear_1 = nn.Linear(model_width, 1)

        # DBP branch
        self.linear_2 = nn.Linear(model_width, 1)

    def forward(self, ppg_x, ecg_x):
        fusion = torch.cat((ppg_x, ecg_x), dim=1)  # (N, C, L)
        fusion = fusion.permute(0, 2, 1)  # (N, L, C)
        share_x, _ = self.share_net(fusion)

        final_out_1 = self.linear_1(share_x)[:, -1, :]
        final_out_2 = self.linear_2(share_x)[:, -1, :]

        return final_out_1, final_out_2


def bpnet18(model_width):
    """
    return a BPNet object
    """
    return BPNet(model_width)


if __name__ == '__main__':
    model = bpnet18(model_width=64)
    # x1 = torch.rand(3, 1, 1250)
    # x2 = torch.rand(3, 1, 1250)
    # y = model(x1, x2)
    # g = torchviz.make_dot(y)
    # g.view()
    summary(model, input_size=[(3, 1, 1250), (3, 1, 1250)])  # 3是样本数

