import torch
import torch.nn as nn
from layers import *
from ops import *
from preprocessing import normalize_adj_torch
import torch.nn.functional as F


class AGSRNet(nn.Module):

    def __init__(self, ks, args):
        super(AGSRNet, self).__init__()

        self.lr_dim = args.lr_dim
        self.hr_dim = args.hr_dim
        self.hidden_dim = args.hidden_dim
        self.layer = GSRLayer(self.hr_dim)
        self.net = GraphUnet(ks, self.lr_dim, self.hr_dim)
        self.gc1 = GraphConvolution(
            self.hr_dim, self.hidden_dim, 0, act=F.relu)
        self.gc2 = GraphConvolution(
            self.hidden_dim, self.hr_dim, 0, act=F.relu)

    def forward(self, lr, lr_dim, hr_dim):
        with torch.autograd.set_detect_anomaly(True):

            I = torch.eye(self.lr_dim).type(torch.FloatTensor)
            A = normalize_adj_torch(lr).type(torch.FloatTensor)

            self.net_outs, self.start_gcn_outs = self.net(A, I)

            self.outputs, self.Z = self.layer(A, self.net_outs)

            self.hidden1 = self.gc1(self.Z, self.outputs)
            self.hidden2 = self.gc2(self.hidden1, self.outputs)
            z = self.hidden2

            z = (z + z.t())/2
            z = z.fill_diagonal_(1)

        return torch.abs(z), self.net_outs, self.start_gcn_outs, self.outputs


class Dense(nn.Module):
    def __init__(self, n1, n2, args):
        super(Dense, self).__init__()
        self.weights = torch.nn.Parameter(
            torch.FloatTensor(n1, n2), requires_grad=True)
        nn.init.normal_(self.weights, mean=args.mean_dense, std=args.std_dense)

    def forward(self, x):
        np.random.seed(1)
        torch.manual_seed(1)

        out = torch.mm(x, self.weights)
        return out


class Discriminator(nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()
        self.dense_1 = Dense(args.hr_dim, args.hr_dim, args)
        self.relu_1 = nn.ReLU(inplace=False)
        self.dense_2 = Dense(args.hr_dim, args.hr_dim, args)
        self.relu_2 = nn.ReLU(inplace=False)
        self.dense_3 = Dense(args.hr_dim, 1, args)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        np.random.seed(1)
        torch.manual_seed(1)
        dc_den1 = self.relu_1(self.dense_1(inputs))
        dc_den2 = self.relu_2(self.dense_2(dc_den1))
        output = dc_den2
        output = self.dense_3(dc_den2)
        output = self.sigmoid(output)
        return torch.abs(output)


def gaussian_noise_layer(input_layer, args):
    z = torch.empty_like(input_layer)
    noise = z.normal_(mean=args.mean_gaussian, std=args.std_gaussian)
    z = torch.abs(input_layer + noise)

    z = (z + z.t())/2
    z = z.fill_diagonal_(1)
    return z
