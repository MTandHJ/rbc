

import torch
import torch.nn as nn


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, 3, stride, padding=1)

def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, 1, stride)

class WideBasicBlock(nn.Module):

    def __init__(self, in_channels, out_channels, 
                    stride=1, shortcut=None):
        super(WideBasicBlock, self).__init__()

        self.bn0 = nn.BatchNorm2d(in_channels)
        self.active = nn.ReLU(inplace=True)
        self.conv0 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv1 = conv3x3(out_channels, out_channels)

        self.shortway = shortcut

    def forward(self, x):
        
        inputs = self.active(self.bn0(x))
        c0 = self.active(self.bn1(self.conv0(inputs)))
        outs1 = self.conv1(c0)
        outs2 = x if not self.shortway else self.shortway(inputs)
        return outs1 + outs2


class WideResnet(nn.Module):

    def __init__(
        self, depth=34, width=10,
        num_classes=10, block=WideBasicBlock
    ):
        super(WideResnet, self).__init__()

        self.addition = False

        assert (depth - 4) % 6 == 0, "depth should be 6n+4"
        num_blocks = (depth - 4) // 6
        self.width = width
        self.cur_channels = 16

        self.conv0 = conv3x3(3, 16)
        self.group1 = self._make_group(block, 16 * width, num_blocks, 1)
        self.group2 = self._make_group(block, 32 * width, num_blocks, 2)
        self.group3 = self._make_group(block, 64 * width, num_blocks, 2)

        self.bn = nn.BatchNorm2d(64 * width)
        self.active = nn.ReLU(inplace=True)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * width, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_group(self, block, out_channels, num_blocks, stride):

        shortcut = None
        if stride != 1 or self.cur_channels != out_channels:
            shortcut = conv1x1(self.cur_channels, out_channels, stride)

        layers = [
            block(self.cur_channels, out_channels, stride, shortcut)
        ]
        self.cur_channels = out_channels
        for _ in range(num_blocks-1):
            layers.append(
                block(out_channels, out_channels)
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        c0 = self.conv0(x)
        g1 = self.group1(c0)
        g2 = self.group2(g1)
        g3 = self.group3(g2)

        features = self.avg_pool(self.active(self.bn(g3))).flatten(1)
        outs = self.fc(features)
        if self.training or self.addition:
            return outs, features
        else:
            return outs


class WideResnetrbc(nn.Module):

    def __init__(
        self, depth=34, width=10,
        num_classes=10, block=WideBasicBlock
    ):
        super(WideResnetrbc, self).__init__()

        self.addition = False

        assert (depth - 4) % 6 == 0, "depth should be 6n+4"
        num_blocks = (depth - 4) // 6
        self.width = width
        self.cur_channels = 16

        self.conv0 = conv3x3(3, 16)
        self.group1 = self._make_group(block, 16 * width, num_blocks, 1)
        self.group2 = self._make_group(block, 32 * width, num_blocks, 2)
        self.group3 = self._make_group(block, 64 * width, num_blocks, 2)

        self.bn = nn.BatchNorm2d(64 * width)
        self.active = nn.Tanh()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.rbc = nn.Linear(64 * width, num_classes, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_group(self, block, out_channels, num_blocks, stride):

        shortcut = None
        if stride != 1 or self.cur_channels != out_channels:
            shortcut = conv1x1(self.cur_channels, out_channels, stride)

        layers = [
            block(self.cur_channels, out_channels, stride, shortcut)
        ]
        self.cur_channels = out_channels
        for _ in range(num_blocks-1):
            layers.append(
                block(out_channels, out_channels)
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        c0 = self.conv0(x)
        g1 = self.group1(c0)
        g2 = self.group2(g1)
        g3 = self.group3(g2)

        features = self.active(self.bn(self.avg_pool(g3))).flatten(1)
        outs = self.rbc(features)
        if self.training or self.addition:
            return outs, features
        else:
            return outs


if __name__ == "__main__":

    model = WideResnet()
    x = torch.randn(1, 3, 32, 32)
    out = model(x)
    print(out)







