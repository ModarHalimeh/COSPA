from model.istft import ISTFT
import torch.nn.functional as F
import torch.nn as nn
import model.complexnn as complexnn
import torch
# Complex CNN for online processing.
# Written for: Microsoft AEC Challenge 2021
# Modar Halimeh. LMS. 2020.
# Modified after: Phase-ware speech enhancement and Deep Complex U-net


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
        self.IN_r = torch.nn.InstanceNorm2d(size)
        self.IN_i = torch.nn.InstanceNorm2d(size)

    def forward(self, xr, xi):

        return self.IN_r(xr), self.IN_i(xi)


class Encoder(nn.Module): #describes one encoder layer
    def __init__(self, conv_cfg, leaky_slope):
        super(Encoder, self).__init__()
        self.conv   = complexnn.ComplexConvWrapper(nn.Conv2d, *conv_cfg, bias=False)
        self.bn     = InstanceNorm(conv_cfg[1])#complexnn.ComplexBatchNorm(conv_cfg[1])#
        self.act    = complexnn.CLeakyReLU(leaky_slope, inplace=True)

    def forward(self, xr, xi):
        xr, xi = self.act(*self.bn(*self.conv(xr, xi)))
        return xr, xi

class Decoder(nn.Module): #describes one decoder layer
    def __init__(self, dconv_cfg, leaky_slope):
        super(Decoder, self).__init__()
        self.dconv  = complexnn.ComplexConvWrapper(nn.ConvTranspose2d, *dconv_cfg, bias=False)
        self.bn     = InstanceNorm(dconv_cfg[1])#complexnn.ComplexBatchNorm(dconv_cfg[1])#
        self.act    = complexnn.CLeakyReLU(leaky_slope, inplace=True)

    def forward(self, xr, xi, skip=None):
        if skip is not None:
            xr, xi = padded_cat(xr, skip[0], dim=1), padded_cat(xi, skip[1], dim=1)
        xr, xi = self.act(*self.bn(*self.dconv(xr, xi)))
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

        originalShape = xr.shape

        xr = torch.reshape(xr, [-1, originalShape[3], originalShape[2]*originalShape[1]])
        xi = torch.reshape(xi, [-1, originalShape[3], originalShape[2]*originalShape[1]])

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

        xr, xi = torch.reshape(xr, [originalShape[0], originalShape[1], originalShape[2],
                                          originalShape[3]]), torch.reshape(xi, [originalShape[0], originalShape[1], originalShape[2], originalShape[3]])
        return xr, xi

class CRUNet(nn.Module):
    def __init__(self, cfg):
        super(CRUNet, self).__init__()

        self.encoders = nn.ModuleList()

        for layer in range(0, 4):
            self.encoders.append(Encoder(cfg['encoders'][layer], 0.1))

        self.decoders = nn.ModuleList()
        for layer in range(0, 3):
            self.decoders.append(Decoder(cfg['decoders'][layer], 0.1))

        self.final_dec = complexnn.ComplexConvWrapper(nn.ConvTranspose2d, *cfg['decoders'][3], bias=True)

        self.gruLayer = ComplexGRU(inputSize=cfg['GRUIN'], hiddenSize=cfg['GRUdim'], numLayers=1, batchSize=1)


    def get_ratio_mask(self, o_real, o_imag):
        mag = torch.sqrt(o_real ** 2 + o_imag ** 2)
        phase = torch.atan2(o_imag, o_real)
        mag = torch.tanh(mag)
        return mag, phase

    def forward(self, xr, xi):
        input_real, input_imag = xr, xi
        skips = list()
        for encoder in self.encoders:
            xr, xi = encoder(xr, xi)
            skips.append((xr, xi))


        xr_gru, xi_gru = self.gruLayer(xr, xi)
        xr, xi = torch.cat((xr, xr_gru), dim=1), torch.cat((xi, xi_gru), dim=1)

        skip = skips.pop()
        skip = None
        for decoder in self.decoders:
            xr, xi = decoder(xr, xi, skip)
            skip = skips.pop()

        xr, xi = padded_cat(xr, skip[0], dim=1), padded_cat(xi, skip[1],
                                                            dim=1)  # ensures skip connection sizes are compatible
        xr, xi = self.final_dec(xr, xi)

        xr, xi = pad2d_as(xr, input_real), pad2d_as(xi, input_imag)

        return xr, xi








