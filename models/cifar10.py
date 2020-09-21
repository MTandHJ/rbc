


import torch
import torch.nn as nn




class CIFAR10(nn.Module):

    def __init__(self):
        super(CIFAR10, self).__init__()

        self.addition = False

        self.conv = nn.Sequential(  # 3 x 32 x 32
            nn.Conv2d(3, 64, 3),  # 64 x 30 x 30
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3),  # 64 x 28 x 28
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 64 x 14 x 14
            nn.Conv2d(64, 128, 3),  # 128 x 12 x 12
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3),  # 128 x 10 x 10
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)  # 128 x 5 x 5
        )

        self.dense = nn.Sequential(
            nn.Linear(128 * 5 * 5, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256)
        )
        self.active = nn.ReLU(inplace=True),
        self.fc = nn.Linear(256, 10)

    def forward(self, x):
        x = self.conv(x).flatten(start_dim=1)
        features = self.active(self.dense(x))
        outs = self.fc(features)
        if self.training or self.addition:
            return outs, features
        else:
            return outs



class CIFAR10rbc(nn.Module):

    def __init__(self, num_features=256):
        super(CIFAR10rbc, self).__init__()

        self.addition = False

        self.conv = nn.Sequential(  # 3 x 32 x 32
            nn.Conv2d(3, 64, 3),  # 64 x 30 x 30
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3),  # 64 x 28 x 28
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 64 x 14 x 14
            nn.Conv2d(64, 128, 3),  # 128 x 12 x 12
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3),  # 128 x 10 x 10
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)  # 128 x 5 x 5
        )

        self.dense = nn.Sequential(
            nn.Linear(128 * 5 * 5, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_features),
            nn.BatchNorm1d(num_features)
        )
        # For RBC, the following activation function
        # and fc layer are speical.
        self.active = nn.Tanh()
        self.rbc = nn.Linear(num_features, 10, bias=False)

    def forward(self, x):
        x = self.conv(x).flatten(start_dim=1)
        features = self.active(self.dense(x))
        outs = self.rbc(features)
        if self.training or self.addition:
            return outs, features
        else:
            return outs