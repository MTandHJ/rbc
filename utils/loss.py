


import torch
import torch.nn as nn


class CrossEntropyLoss:
    """
    To filter some useless kwargs.
    """
    def __call__(self, outs, labels, **kwargs):
        return nn.functional.cross_entropy(outs, labels)


class RbcLoss:
    """
    Rbcloss is the basic loss (loss1) together with a regularizer (loss2):
                loss1 + leverage * loss2
    """
    def __init__(
        self, device, loss1=None,  scale=0.01,
        num_classes=10, leverage=100) -> None:
        if not loss1:
            self.loss1 = CrossEntropyLoss()
        else:
            self.loss1 = loss1
        self.device = device
        self.leverage = leverage
        self.rbc_target(num_classes, scale)
    
    def rbc_target(self, num_classes, scale) -> None:
        rbz_target = torch.ones((num_classes, num_classes)) / (1 - num_classes) + \
                    (torch.ones(num_classes) * (num_classes / (num_classes - 1))).diag()
        self.target = rbz_target.to(self.device) * (scale ** 2)

    def loss2(self, model):
        if hasattr(model, "rbc"):
            temp = model.rbc.weight @ model.rbc.weight.t()
            return (temp - self.target).norm()
        else:
            return torch.tensor(0.)

    def __call__(self, model, outs, features, labels):
        loss2 = self.loss2(model)
        loss1 = self.loss1(
            outs=outs,
            features=features,
            labels=labels
        )
        loss = loss1 + self.leverage * loss2
        return loss

class MMCLoss:
    """
    MMCLoss:
    $$
    \\frac{1}{2} \| z - w_y \|_2^2.
    $$
    """
    def __init__(self, model, scale):
        if hasattr(model, "rbc"):
            self.weight = model.rbc.weight
        else:
            self._generate_opt_means(model, scale)
            self.weight = model.fc.weight    
        
    @classmethod
    def initialize(cls, num_features, num_classes, scale):
        weight = torch.zeros((num_classes, num_features))
        weight[0, 0] = 1.
        l_r_1 = num_classes - 1
        for i in range(1, num_classes):
            for j in range(i):
                weight[i, j] = -(1 + weight[i] @ weight[j] * l_r_1) \
                                    / (weight[j, j] * l_r_1)
            weight[i, i] = (1 - weight[i] @ weight[i]).abs().sqrt()
        return weight * scale

    def _generate_opt_means(self, model, scale):
        num_classes, num_features = model.fc.weight.size()
        device = model.fc.weight.device
        weight = self.initialize(num_features, num_classes, scale)
        model.fc.weight.data = weight.to(device)
        model.fc.weight.requires_grad_(False)

    def __call__(self, features, labels, **kwargs):
        n = len(labels)
        temp = self.weight[labels]
        loss = nn.functional.mse_loss(features, temp)
        return loss
    