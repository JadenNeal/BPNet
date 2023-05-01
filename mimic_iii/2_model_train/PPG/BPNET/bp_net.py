import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
import torchviz


# add Large kernel attention


def CBR(in_channel, out_channel, kernel_size, stride=1, padding=0, groups=1, dilation=1, bias=False):
    """
    conv --> bn --> relu
    """
    return nn.Sequential(
        nn.Conv1d(in_channels=in_channel, out_channels=out_channel,
                  kernel_size=kernel_size, stride=stride, padding=padding,
                  groups=groups, dilation=dilation, bias=bias),
        nn.BatchNorm1d(out_channel),
        nn.ReLU(inplace=True)
    )


class LKA(nn.Module):
    def __init__(self, in_channel):
        super(LKA, self).__init__()
        self.identity = nn.Sequential()
        self.conv1 = CBR(in_channel, in_channel, 5, 1, 2, groups=in_channel)
        self.conv_spatial = CBR(in_channel, in_channel, 7, 1, 9, groups=in_channel, dilation=3)
        self.conv2 = CBR(in_channel, in_channel, 1, 1, 0)
        # self.corrcoef = nn.Sigmoid()
        self.bn = nn.BatchNorm1d(in_channel)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.identity(x)
        out = self.conv1(x)
        out = self.conv_spatial(out)
        out = self.conv2(out)
        # out = self.corrcoef(out)

        out = out * identity
        out = self.bn(out)
        out = self.act(out)

        return out


class LKABlock(nn.Module):
    """
    large kernel attention block
    """
    def __init__(self, in_channel, out_channel, stride):
        super(LKABlock, self).__init__()
        self.proj_1 = CBR(in_channel, out_channel, 1, stride, 0)
        self.spatial_attn = LKA(out_channel)
        self.proj_2 = nn.Sequential(
            nn.Conv1d(out_channel, out_channel, 1, 1, 0),
            nn.BatchNorm1d(out_channel)
            # relu放到basic block中添加
        )

    def forward(self, x):
        out = self.proj_1(x)
        out = self.spatial_attn(out)
        out = self.proj_2(out)

        return out


class BasicBlock(nn.Module):
    """
    Basic Block for resnet 18 and resnet 34
    """
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, ratio=16):
        super(BasicBlock, self).__init__()
        self.residual = LKABlock(in_channels, out_channels, stride=stride)

        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(out_channels, out_channels // ratio),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels // ratio, out_channels),
            nn.Sigmoid()
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

        squeeze = self.squeeze(residual)
        # print(squeeze.shape)  # (2, 64, 1)
        squeeze = squeeze.view(squeeze.size(0), -1)
        excitation = self.excitation(squeeze)
        excitation = excitation.view(residual.size(0), residual.size(1), 1)
        x = residual * excitation

        output = x + identity
        output = self.act(output)

        return output


class SEResNet(nn.Module):
    def __init__(self, model_width, layer_dims, ratio=16):
        super(SEResNet, self).__init__()
        self.in_channels = model_width

        # stem
        self.stem = nn.Sequential(
            # one channel, namely ppg
            CBR(1, model_width, 7, 2, 3),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1))

        # residual block
        # SBP branch
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
        c1 = self.stem(x)
        c2 = self.layer_1(c1)
        c3 = self.layer_2(c2)
        c4 = self.layer_3(c3)
        c5 = self.layer_4(c4)

        return c2, c3, c4, c5


class BPNet(nn.Module):
    def __init__(self, model_width, layer_dims, ratio):
        super(BPNet, self).__init__()
        self.ppg_net = SEResNet(model_width, layer_dims, ratio)
        self.ecg_net = SEResNet(model_width, layer_dims, ratio)

        self.top_layer = nn.Conv1d(model_width * 8, model_width, 1, 1, 0, bias=False)
        self.latlayer_1 = nn.Conv1d(model_width * 4, model_width, 1, 1, 0, bias=False)
        self.latlayer_2 = nn.Conv1d(model_width * 2, model_width, 1, 1, 0, bias=False)
        self.latlayer_3 = nn.Conv1d(model_width, model_width, 1, 1, 0, bias=False)

        self.share = CBR(model_width, model_width, 3, 1, 1)

        # SBP branch
        self.cbr_1 = nn.Sequential(
            CBR(model_width, model_width * 2, 3, 2, 1),
            CBR(model_width * 2, model_width * 2, 3, 1, 1)
        )
        self.mlp_1 = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(model_width * 2, 1)
        )

        # DBP branch
        self.cbr_2 = nn.Sequential(
            CBR(model_width, model_width * 2, 3, 2, 1),
            CBR(model_width * 2, model_width * 2, 3, 1, 1)
        )
        self.mlp_2 = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(model_width * 2, 1)
        )

    def _upsmaple_add(self, x, y):
        length = y.shape[2]
        out = F.interpolate(x, size=length, mode="linear", align_corners=True) + y
        return out

    def forward(self, ppg_x):
        ppg_c2, ppg_c3, ppg_c4, ppg_c5 = self.ppg_net(ppg_x)

        p5 = self.top_layer(ppg_c5)
        p4 = self._upsmaple_add(p5, self.latlayer_1(ppg_c4))
        p3 = self._upsmaple_add(p4, self.latlayer_2(ppg_c3))
        p2 = self._upsmaple_add(p3, self.latlayer_3(ppg_c2))

        share_x = self.share(p2)

        out_1 = self.cbr_1(share_x)
        final_out_1 = self.mlp_1(out_1)

        out_2 = self.cbr_2(share_x)
        final_out_2 = self.mlp_2(out_2)

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

