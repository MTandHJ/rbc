






import torch
import torch.nn as nn


# import nn.Parameter as Parameter


# NORM_C = 0.1

# class RBC(nn.Module):
#     def __init__(self, num_features, num_classes, 
#                     allowed_grad=True):
#         super(RBC, self).__init__()
#         w = self.initialize(num_features, num_classes)
#         self.weight = nn.Parameter(w, requires_grad=allowed_grad)

#     def initialize(self, num_features, num_classes):
#         weights = torch.zeros((num_classes, num_features))
#         weights[0, 0] = 1.
#         l_r_1 = num_classes - 1
#         for i in range(1, num_classes):
#             for j in range(i):
#                 weights[i, j] = -(1 + weights[i] @ weights[j] * l_r_1) \
#                                     / (weights[j, j] * l_r_1)
#             weights[i, i] = (1 - weights[i] @ weights[i]).abs().sqrt()
#         return weights * NORM_C

#     def forward(self, x):
#         return x @ self.weight.t()








