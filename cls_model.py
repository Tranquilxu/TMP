import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.autograd import Variable
import torch.nn.init as init
import torch.nn.functional as F


torch.manual_seed(1)
# np.random.seed(1)
class Linear(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Linear, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = x.view(-1, self.num_flat_features(x))
        out = self.linear(out)
        return out

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, parameter_momentum=0.1):
        super(MLP, self).__init__()

        self.L1 = nn.Linear(input_dim, 300, bias=False)
        init.xavier_uniform_(self.L1.weight)
        self.bn1 = nn.BatchNorm1d(300, momentum=parameter_momentum)
        init.ones_(self.bn1.weight)

        self.L2 = nn.Linear(300, 301, bias=False)
        init.xavier_uniform_(self.L2.weight)
        self.bn2 = nn.BatchNorm1d(301, momentum=parameter_momentum)
        init.ones_(self.bn2.weight)

        self.L3 = nn.Linear(301, 302, bias=False)
        init.xavier_uniform_(self.L3.weight)
        self.bn3 = nn.BatchNorm1d(302, momentum=parameter_momentum)
        init.ones_(self.bn3.weight)

        self.L4 = nn.Linear(302, 303, bias=False)
        init.xavier_uniform_(self.L4.weight)
        self.bn4 = nn.BatchNorm1d(303, momentum=parameter_momentum)
        init.ones_(self.bn4.weight)

        self.L5 = nn.Linear(303, output_dim, bias=True)
        init.xavier_uniform_(self.L5.weight)
        init.zeros_(self.L5.bias)

    def forward(self, x):
        x = x.view(-1, self.num_flat_features(x))
        x = self.L1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.L2(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.L3(x)
        x = self.bn3(x)
        x = F.relu(x)

        x = self.L4(x)
        x = self.bn4(x)
        x = F.sigmoid(x)

        x = self.L5(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features



class MLP_no_BN(nn.Module):
    def __init__(self, input_dim, output_dim, parameter_momentum=0.1):
        super(MLP_no_BN, self).__init__()

        self.L1 = nn.Linear(input_dim, 300, bias=False)
        self.D1 = nn.Dropout1d(0.5)
        init.xavier_uniform_(self.L1.weight)

        self.L2 = nn.Linear(300, 301, bias=False)
        self.D2 = nn.Dropout1d(0.5)
        init.xavier_uniform_(self.L2.weight)

        self.L3 = nn.Linear(301, 302, bias=False)
        init.xavier_uniform_(self.L3.weight)

        self.L4 = nn.Linear(302, 303, bias=False)
        init.xavier_uniform_(self.L4.weight)

        self.L5 = nn.Linear(303, output_dim, bias=True)
        init.xavier_uniform_(self.L5.weight)
        init.zeros_(self.L5.bias)


    def forward(self, x):
        x = x.view(-1, self.num_flat_features(x))
        x = self.L1(x)
        x = self.D1(x)
        x = F.relu(x)

        x = self.L2(x)
        x = self.D2(x)
        x = F.relu(x)

        x = self.L3(x)
        x = F.relu(x)

        x = self.L4(x)
        x = F.relu(x)

        x = self.L5(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class MLP_clip(nn.Module):
    def __init__(self, input_dim, output_dim, parameter_momentum=0.1):
        super(MLP_clip, self).__init__()

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

class text_input(nn.Module):
    def __init__(self):
        super(text_input, self).__init__()
        self.L1 = nn.Linear(10, 77, bias=False)
        init.xavier_uniform_(self.L1.weight)

    def forward(self, x):
        # print("x size", x.size())
        x = self.L1(x)
        return x

class Cnn(nn.Module):
    def __init__(self, input_channels, n_outputs, dropout_rate):
        self.dropout_rate = dropout_rate
        super(Cnn, self).__init__()

        self.c1 = nn.Conv2d(input_channels, 128, kernel_size=3, stride=1, padding=1)
        self.c2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.c3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.c4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.c5 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.c6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.c7 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=0)
        self.c8 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=0)
        self.c9 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=0)
        self.l_c1 = nn.Linear(128, n_outputs)
        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(256)
        self.bn6 = nn.BatchNorm2d(256)
        self.bn7 = nn.BatchNorm2d(512)
        self.bn8 = nn.BatchNorm2d(256)
        self.bn9 = nn.BatchNorm2d(128)

    def forward(self, x):
        x = self.c1(x)
        x = self.bn1(x)
        x = F.leaky_relu(x, negative_slope=0.01)
        x = self.c2(x)
        x = self.bn2(x)
        x = F.leaky_relu(x, negative_slope=0.01)
        x = self.c3(x)
        x = self.bn3(x)
        x = F.leaky_relu(x, negative_slope=0.01)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.dropout2d(x, p=self.dropout_rate)

        x = self.c4(x)
        x = self.bn4(x)
        x = F.leaky_relu(x, negative_slope=0.01)
        x = self.c5(x)
        x = self.bn5(x)
        x = F.leaky_relu(x, negative_slope=0.01)
        x = self.c6(x)
        x = self.bn6(x)
        x = F.leaky_relu(x, negative_slope=0.01)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.dropout2d(x, p=self.dropout_rate)

        x = self.c7(x)
        x = self.bn7(x)
        x = F.leaky_relu(x, negative_slope=0.01)
        x = self.c8(x)
        x = self.bn8(x)
        x = F.leaky_relu(x, negative_slope=0.01)
        x = self.c9(x)
        x = self.bn9(x)
        x = F.leaky_relu(x, negative_slope=0.01)
        x = F.avg_pool2d(x, kernel_size=x.data.shape[2])

        x = x.view(x.size(0), x.size(1))
        x = self.l_c1(x)

        return x
