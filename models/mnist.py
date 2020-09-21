






import torch
import torch.nn as nn

class MNIST(nn.Module):

    def __init__(self):
        super(MNIST, self).__init__()

        self.addition = False

        self.conv = nn.Sequential( # 1 x 28 x 28
            nn.Conv2d(1, 32, 3),   # 32 x 26 x 26
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3),  # 32 x 24 x 24
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),       # 32 x 12 x 12
            nn.Conv2d(32, 64, 3),  # 64 x 10 x 10
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3),  # 64 x 8 x 8
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)        # 64 x 4 x 4
        )

        self.dense = nn.Sequential(
            nn.Linear(64 * 4 * 4, 200),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(200),
            nn.Linear(200, 200),
            nn.BatchNorm1d(200)
        )
        self.active = nn.ReLU(inplace=True),
        self.fc = nn.Linear(200, 10)


    def forward(self, x):
        x = self.conv(x).flatten(start_dim=1)
        features = self.active(self.dense(x))
        outs = self.fc(features)
        if self.training or self.addition:
            return outs, features
        else:
            return outs

class MNISTrbc(nn.Module):

    def __init__(self, num_features=200):
        super(MNISTrbc, self).__init__()

        self.addition = False

        self.conv = nn.Sequential( # 1 x 28 x 28
            nn.Conv2d(1, 32, 3),   # 32 x 26 x 26
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3),  # 32 x 24 x 24
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),       # 32 x 12 x 12
            nn.Conv2d(32, 64, 3),  # 64 x 10 x 10
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3),  # 64 x 8 x 8
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)        # 64 x 4 x 4
        )

        self.dense = nn.Sequential(
            nn.Linear(64 * 4 * 4, 200),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(200),
            nn.Linear(200, num_features),
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
