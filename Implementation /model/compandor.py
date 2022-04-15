import torch.nn.functional as F
import torch.nn as nn
import model.complexnn as complexnn
import torch



device = 'cuda:0'


def pad2d_as(x1, x2):
    # Pad x1 to have same size with x2
    # inputs are NCHW
    diffH = x2.size()[2] - x1.size()[2]
    diffW = x2.size()[3] - x1.size()[3]

    return F.pad(x1, (0, diffW, 0, diffH))

def padded_cat(x1, x2, dim):
    x1 = pad2d_as(x1, x2)
    x1 = torch.cat([x1, x2], dim=dim)
    return x1


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


class ComplexGRU(nn.Module):
    def __init__(self, inputSize= 64*17*2, hiddenSize= 128, numLayers =1, batchSize=1):
        super(ComplexGRU, self).__init__()

        self.hiddenSize = hiddenSize

        self.FCLr_in      = torch.nn.Linear(in_features=inputSize, out_features=hiddenSize, bias=False)
        self.FCLi_in      = torch.nn.Linear(in_features=inputSize, out_features=hiddenSize, bias=False)

        self.grulayerR = torch.nn.GRU(input_size=hiddenSize, hidden_size=hiddenSize, num_layers=numLayers, bias=False, batch_first=True)
        self.grulayerI = torch.nn.GRU(input_size=hiddenSize, hidden_size=hiddenSize, num_layers=numLayers, bias=False, batch_first=True)

        self.FCLr      = torch.nn.Linear(in_features=hiddenSize, out_features=inputSize, bias=False)
        self.FCLi      = torch.nn.Linear(in_features=hiddenSize, out_features=inputSize, bias=False)
        self.act       = nn.LeakyReLU(0.1)

    def forward(self, xr, xi):



        xr_gru_in = self.act(self.FCLr_in(xr) - self.FCLi_in(xi))
        xi_gru_in = self.act(self.FCLi_in(xr) + self.FCLr_in(xi))

        GRU_rr, _ = self.grulayerR(xr_gru_in)
        GRU_ri, _ = self.grulayerI(xr_gru_in)

        GRU_ii, _ = self.grulayerI(xi_gru_in)
        GRU_ir, _ = self.grulayerR(xi_gru_in)

        xr_gru = GRU_rr - GRU_ii
        xi_gru = GRU_ir + GRU_ri

        xr = self.act(self.FCLr(xr_gru) - self.FCLi(xi_gru))
        xi = self.act(self.FCLi(xr_gru) + self.FCLr(xi_gru))


        return xr, xi


class compandor(torch.nn.Module):
    def __init__(self, cfg, nfft, nr_channels):
        super(compandor, self).__init__()

        self.FC1    = complexnn.complex_Linear(cfg['layer1_in'], cfg['layer1_out'])
        self.norm_1 = InstanceNorm(cfg['layer1_out'])

        self.gru    = ComplexGRU(cfg['gru_in'], cfg['gru_out'])

        self.FC2    = complexnn.complex_Linear(cfg['layer2_in'], cfg['layer2_out'])
        self.norm_2 = InstanceNorm(cfg['layer2_out'])

        self.act    = torch.nn.LeakyReLU(0.1)

        self.nfft           = nfft
        self.nr_channels    = nr_channels

    def dearrange_channels(self, xr, xi):
        xr_unwrapped  = xr[..., :self.nfft].unsqueeze(dim=3)
        xi_unwrapped  = xi[..., :self.nfft].unsqueeze(dim=3)
        for ind in range(1, self.nr_channels):
            xr_unwrapped = torch.cat((xr_unwrapped, xr[..., (ind*self.nfft):((ind+1)*self.nfft)].unsqueeze(dim=3)), dim=3)
            xi_unwrapped = torch.cat((xi_unwrapped, xi[..., (ind * self.nfft):((ind + 1) * self.nfft)].unsqueeze(dim=3)), dim=3)
        return xr_unwrapped, xi_unwrapped

    def forward(self, xr, xi):

        xr, xi = xr.transpose(1,2), xi.transpose(1,2)
        xr, xi = self.FC1(xr, xi)
        xr, xi = self.norm_1(xr, xi)
        xr, xi = self.act(xr), self.act(xi)

        xr, xi = self.gru(xr, xi)

        xr, xi = self.FC2(xr, xi)
        xr, xi = self.norm_2(xr, xi)
        xr, xi = self.act(xr), self.act(xi)

        xr, xi = self.dearrange_channels(xr, xi)


        return xr, xi









