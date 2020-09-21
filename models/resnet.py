


import torch
import torch.nn as nn


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, 3, stride, padding=1)

def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, 1, stride)

def get_depth(depth):
    depth_lst = [18, 34, 50, 101, 152]
    assert (depth in depth_lst), "Error : Resnet depth should be either 18, 34, 50, 101, 152"
    cf_dict = {
        '18': (BasicBlock, [2, 2, 2, 2]),
        '34': (BasicBlock, [3, 4, 6, 3]),
        '50': (Bottleneck, [3, 4, 6, 3]),
        '101':(Bottleneck, [3, 4, 23, 3]),
        '152':(Bottleneck, [3, 8, 36, 3]),
    }

    return cf_dict[str(depth)]

class BasicBlock(nn.Module):

    expansion = 1

    def __init__(self, in_channels, out_channels, 
                    stride=1, shortcut=None):
        super(BasicBlock, self).__init__()
        
        self.active = nn.ReLU(inplace=True)
        self.longway = nn.Sequential(
            conv3x3(in_channels, out_channels, stride),
            nn.BatchNorm2d(out_channels),
            self.active,
            conv3x3(out_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            self.active
        )
        self.shortway = shortcut

    def forward(self, x):

        residual = self.longway(x)
        identity = x if not self.shortway else self.shortway(x)
        return self.active(residual + identity)

class Bottleneck(nn.Module):

    expansion = 4

    def __init__(self, in_channels, out_channels,
                    stride=1, shortcut=None):
        super(Bottleneck, self).__init__()

        self.active = nn.ReLU(inplace=True)
        self.longway = nn.Sequential(
            conv1x1(in_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            self.active,
            conv3x3(out_channels, out_channels, stride),
            nn.BatchNorm2d(out_channels),
            self.active,
            conv3x3(out_channels, out_channels * self.expansion),
            nn.BatchNorm2d(out_channels * self.expansion),
            self.active
        )
        self.shortway = shortcut

    def forward(self, x):

        residual = self.longway(x)
        identity = x if not self.shortway else self.shortway(x)
        return self.active(residual + identity)

class Resnet(nn.Module):

    def __init__(self, depth, num_classes=10):
        super(Resnet, self).__init__()

        self.cur_channels = 64
        block, num_blocks = get_depth(depth)

        self.active = nn.ReLU(inplace=True)
        self.layer0 = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64),
            self.active,
            nn.MaxPool2d(3, 2, 1)
        )
        self.layer1 = self._make_layer(block, 64, num_blocks[0], 1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], 2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], 2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], 2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, channels, num_blocks, stride):

        shortcut = None
        if stride != 1 or self.cur_channels != channels * block.expansion:
            shortcut = nn.Sequential(
                conv1x1(self.cur_channels, channels * block.expansion, stride),
                nn.BatchNorm2d(channels * block.expansion)
            )
        
        layers = [
            block(self.cur_channels, channels, stride, shortcut)
        ]
        self.cur_channels = channels * block.expansion
        for _ in range(num_blocks-1):
            layers.append(
                block(self.cur_channels, channels)
            )
        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.flatten(start_dim=1)
        outs = self.fc(x)

        return outs
















