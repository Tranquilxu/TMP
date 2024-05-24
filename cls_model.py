import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


torch.manual_seed(1)


class Linear(nn.Module):
    def __init__(self, input_dim, output_dim, parameter_momentum=0.1):
        super(Linear, self).__init__()

        self.L1 = nn.Linear(input_dim, output_dim, bias=False)
        init.xavier_uniform_(self.L1.weight)



    def forward(self, x):
        x = x.view(-1, self.num_flat_features(x))
        # print("x size", x.size())
        x = self.L1(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class image_input(nn.Module):
    def __init__(self):
        super(image_input, self).__init__()
        self.c1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.c2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.c3 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.c4 = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(64)


    def forward(self, x):
        # print("x size", x.size())
        x = self.c1(x)
        x = self.bn1(x)
        x = F.leaky_relu(x, negative_slope=0.05)
        x = F.dropout2d(x, p=0.001)
        x = self.c2(x)
        x = self.bn2(x)
        x = F.leaky_relu(x, negative_slope=0.05)
        x = F.dropout2d(x, p=0.001)
        x = self.c3(x)
        x = self.bn3(x)
        x = F.leaky_relu(x, negative_slope=0.05)
        x = self.c4(x)
        return x
