# encoding=utf-8

import torch
import torch.nn.functional as F
import torch.nn as nn
import math


# from collections import OrderedDict

def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]


def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, std=0.001)
            # torch.nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            # torch.nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


# -------------------------------------------------#
#   MISH激活函数
# -------------------------------------------------#
class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


# -------------------------------------------------#
#   卷积块
#   CONV + BATCHNORM + MISH
# -------------------------------------------------#
class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(BasicConv, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = Mish()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


# ---------------------------------------------------#
#   CSPdarknet的结构块的组成部分
#   内部堆叠的残差块
# ---------------------------------------------------#
class Resblock(nn.Module):
    def __init__(self, channels, hidden_channels=None, residual_activation=nn.Identity()):
        super(Resblock, self).__init__()

        if hidden_channels is None:
            hidden_channels = channels

        self.block = nn.Sequential(
            BasicConv(channels, hidden_channels, 1),
            BasicConv(hidden_channels, channels, 3)
        )

    def forward(self, x):
        return x + self.block(x)


# ---------------------------------------------------#
#   CSPdarknet的结构块
#   存在一个大残差边
#   这个大残差边绕过了很多的残差结构
# ---------------------------------------------------#
class Resblock_body(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks, first):
        """
        :param in_channels:
        :param out_channels:
        :param num_blocks:
        :param first:
        """

        super(Resblock_body, self).__init__()

        self.downsample_conv = BasicConv(in_channels, out_channels, 3, stride=2)

        if first:
            self.split_conv0 = BasicConv(out_channels, out_channels, 1)
            self.split_conv1 = BasicConv(out_channels, out_channels, 1)
            self.blocks_conv = nn.Sequential(
                Resblock(channels=out_channels, hidden_channels=out_channels // 2),
                BasicConv(out_channels, out_channels, 1)
            )
            self.concat_conv = BasicConv(out_channels * 2, out_channels, 1)
        else:
            self.split_conv0 = BasicConv(out_channels, out_channels // 2, 1)
            self.split_conv1 = BasicConv(out_channels, out_channels // 2, 1)

            self.blocks_conv = nn.Sequential(
                *[Resblock(out_channels // 2) for _ in range(num_blocks)],
                BasicConv(out_channels // 2, out_channels // 2, 1)
            )
            self.concat_conv = BasicConv(out_channels, out_channels, 1)

    def forward(self, x):
        x = self.downsample_conv(x)

        x0 = self.split_conv0(x)
        x1 = self.split_conv1(x)
        x1 = self.blocks_conv(x1)

        x = torch.cat([x1, x0], dim=1)
        x = self.concat_conv(x)

        return x


class CSPDarkNet(nn.Module):
    def __init__(self, layers, heads, head_conv):
        """
        :param layers:
        :param heads:
        :param head_conv:
        """
        super(CSPDarkNet, self).__init__()

        self.heads = heads
        self.inplanes = 32
        self.conv1 = BasicConv(3, self.inplanes, kernel_size=3, stride=1)
        self.feature_channels = [64, 128, 256, 512, 1024]

        self.stages = nn.ModuleList([
            Resblock_body(self.inplanes, self.feature_channels[0], layers[0], first=True),
            Resblock_body(self.feature_channels[0], self.feature_channels[1], layers[1], first=False),
            # Resblock_body(self.feature_channels[1], self.feature_channels[2], layers[2], first=False),
            # Resblock_body(self.feature_channels[2], self.feature_channels[3], layers[3], first=False),
            # Resblock_body(self.feature_channels[3], self.feature_channels[4], layers[4], first=False)
        ])

        self.num_features = 1

        # 进行权值初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        # set heads
        for head in self.heads:
            channels = self.heads[head]
            if head_conv > 0:
                head_out = nn.Sequential(nn.Conv2d(128, head_conv, kernel_size=3, padding=1, bias=True),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(head_conv, channels, kernel_size=1, stride=1, padding=0, bias=True))
                if 'hm' in head:
                    head_out[-1].bias.data.fill_(-2.19)
                else:
                    fill_fc_weights(head_out)
            else:
                head_out = nn.Conv2d(128, channels, kernel_size=1, stride=1, padding=0, bias=True)
                if 'hm' in head:
                    head_out.bias.data.fill_(-2.19)
                else:
                    fill_fc_weights(head_out)

            # set each head
            self.__setattr__(head, head_out)

    def forward(self, x):
        x = self.conv1(x)

        x = self.stages[0](x)
        x = self.stages[1](x)

        # out3 = self.stages[2](x)
        # out4 = self.stages[3](out3)
        # out5 = self.stages[4](out4)
        # return out3, out4, out5

        # get heads
        ret = {}
        for head in self.heads:  # get each head
            ret[head] = self.__getattr__(head)(x)

        return [ret]


def darknet53(pre_trained, **kwargs):
    model = CSPDarkNet([1, 2, 8, 8, 4])

    if pre_trained:
        if isinstance(pre_trained, str):
            model.load_state_dict(torch.load(pre_trained))
        else:
            raise Exception('darknet request a pre_trained path. got [{}]'.format(pre_trained))

    return model


def get_csp_darknet(num_layers, heads, head_conv):
    """
    :param num_layers:
    :param heads:
    :param head_conv:
    :return:
    """
    model = CSPDarkNet(layers=[1, 2, 8, 8, 4], heads=heads, head_conv=head_conv)
    return model
