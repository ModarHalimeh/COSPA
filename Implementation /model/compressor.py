import torch.nn.functional as F
import torch.nn as nn
import model.complexnn as complexnn
import torch


class compressor(nn.Module): #describes one encoder layer
    def __init__(self, conv_cfg):
        super(compressor, self).__init__()
        self.conv   = complexnn.ComplexConvWrapper(nn.Conv2d, *conv_cfg, bias=False)

    def forward(self, xr, xi):
        xr, xi = self.conv(xr, xi)
        return xr, xi