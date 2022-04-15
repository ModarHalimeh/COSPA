import torch.nn.functional as F
import torch.nn as nn
import model.complexnn as complexnn
import torch

class InstanceNorm(torch.nn.Module):
    def __init__(self, size):
        super(InstanceNorm, self).__init__()
        self.IN_r = torch.nn.InstanceNorm1d(size)
        self.IN_i = torch.nn.InstanceNorm1d(size)

    def forward(self, xr, xi):
        xr, xi = xr.transpose(1,2), xi.transpose(1,2)
        xr, xi = self.IN_r(xr), self.IN_i(xi)
        xr, xi = xr.transpose(1, 2), xi.transpose(1, 2)
        return xr, xi

class decoder(torch.nn.Module):
    def __init__(self, cfg):
        super(decoder, self).__init__()

        self.FC1    = complexnn.complex_Linear(cfg['layer1_in'], cfg['layer1_out'])
        self.norm_1 = InstanceNorm(cfg['layer1_out'])

        self.FC2    = complexnn.complex_Linear(cfg['layer2_in'], cfg['layer2_out'])
        self.norm_2 = InstanceNorm(cfg['layer2_out'])

        self.FC3    = complexnn.complex_Linear(cfg['layer3_in'], cfg['layer3_out'], bias=True)

        self.act    = torch.nn.LeakyReLU(0.1)



    def forward(self, xr, xi):

        xr, xi = self.FC1(xr, xi)
        xr, xi = self.norm_1(xr, xi)
        xr, xi = self.act(xr), self.act(xi)

        xr, xi = self.FC2(xr, xi)
        xr, xi = self.norm_2(xr, xi)
        xr, xi = self.act(xr), self.act(xi)

        xr, xi = self.FC3(xr, xi)

        xr, xi = xr.transpose(1,2), xi.transpose(1,2)

        return xr, xi