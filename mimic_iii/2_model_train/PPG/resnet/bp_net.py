import torch
import torch.nn as nn
from torchinfo import summary
import torchviz


class BasicBlock(nn.Module):
    """
    Basic Block for resnet 18 and resnet 34
    """
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, ratio=16):
        super(BasicBlock, self).__init__()

        self.residual = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_channels * BasicBlock.expansion),
            # nn.ReLU(inplace=True)
        )

        self.act = nn.ReLU(inplace=True)

        # the shortcut output dimension is not the same with residual function
        # use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels * BasicBlock.expansion)
            )
        else:
            self.downsample = nn.Sequential()

    def forward(self, inputs):
        identity = self.downsample(inputs)
        residual = self.residual(inputs)

        output = residual + identity
        output = self.act(output)

        return output


class ResNet(nn.Module):
    def __init__(self, model_width, layer_dims, ratio=16):
        super(ResNet, self).__init__()
        self.in_channels = model_width

        # stem
        self.stem = nn.Sequential(
            # two channel, ppg & ecg
            nn.Conv1d(1, model_width, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(model_width),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1))

        # residual block
        self.layer_1 = self._make_layer(model_width, layer_dims[0], stride=1, ratio=ratio)
        self.layer_2 = self._make_layer(model_width * 2, layer_dims[1], stride=2, ratio=ratio)
        self.layer_3 = self._make_layer(model_width * 4, layer_dims[2], stride=2, ratio=ratio)
        self.layer_4 = self._make_layer(model_width * 8, layer_dims[3], stride=2, ratio=ratio)

    def _make_layer(self, out_channels, num_blocks, stride, ratio):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_channels, out_channels, stride, ratio))
            self.in_channels = out_channels * BasicBlock.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        # 单输入
        # 从上到下
        out = self.stem(x)
        out = self.layer_1(out)
        out = self.layer_2(out)
        out = self.layer_3(out)
        out = self.layer_4(out)

        return out


class BPNet(nn.Module):
    def __init__(self, model_width, layer_dims, ratio):
        super(BPNet, self).__init__()
        self.share_net = ResNet(model_width, layer_dims, ratio)

        # SBP branch
        self.mlp_1 = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(model_width * 8, 1)
        )

        # DBP branch
        self.mlp_2 = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(model_width * 8, 1)
        )

    def forward(self, ppg_x):
        fusion = ppg_x
        share_x = self.share_net(fusion)

        final_out_1 = self.mlp_1(share_x)
        final_out_2 = self.mlp_2(share_x)

        return final_out_1, final_out_2


def bpnet18(model_width, ratio=16):
    """
    return a BPNet 18 object
    """
    return BPNet(model_width, [2, 2, 2, 2], ratio)


def bpnet34(model_width, ratio=16):
    """
    return a BPNet 34 object
    """
    return BPNet(model_width, [3, 4, 6, 3], ratio)


if __name__ == '__main__':
    model = bpnet18(model_width=64)
    # x1 = torch.rand(3, 1, 1250)
    # x2 = torch.rand(3, 1, 1250)
    # y = model(x1, x2)
    # g = torchviz.make_dot(y)
    # g.view()
    summary(model, input_size=(3, 1, 1250))  # 3是样本数

