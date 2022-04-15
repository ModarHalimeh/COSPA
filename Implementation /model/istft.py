import torch
import torch.nn.functional as F
import numpy as np
import scipy.signal

class ISTFT(torch.nn.Module):
    def __init__(self, filter_length=400, hop_length=200, window='hanning', center=False):
        super(ISTFT, self).__init__()

        self.filter_length = filter_length
        self.hop_length = hop_length
        self.center = center

        win_cof = scipy.signal.get_window(window, filter_length)
        self.inv_win = self.inverse_stft_window(win_cof, hop_length)

        fourier_basis = np.fft.fft(np.eye(self.filter_length))
        cutoff = int((self.filter_length / 2 + 1))
        fourier_basis = np.vstack([np.real(fourier_basis[:cutoff, :]),
                                   np.imag(fourier_basis[:cutoff, :])])
        inverse_basis = torch.FloatTensor(self.inv_win * \
                np.linalg.pinv(fourier_basis).T[:, None, :])

        self.register_buffer('inverse_basis', inverse_basis.float())

    # Use equation 8 from Griffin, Lim.
    # Paper: "Signal Estimation from Modified Short-Time Fourier Transform"
    # Reference implementation: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/signal/spectral_ops.py
    # librosa use equation 6 from paper: https://github.com/librosa/librosa/blob/0dcd53f462db124ed3f54edf2334f28738d2ecc6/librosa/core/spectrum.py#L302-L311
    def inverse_stft_window(self, window, hop_length):
        window_length = len(window)
        denom = window ** 2
        overlaps = -(-window_length // hop_length)  # Ceiling division.
        denom = np.pad(denom, (0, overlaps * hop_length - window_length), 'constant')
        denom = np.reshape(denom, (overlaps, hop_length)).sum(0)
        denom = np.tile(denom, (overlaps, 1)).reshape(overlaps * hop_length)
        return window / denom[:window_length]

    def forward(self, real_part, imag_part, length=None):
        if (real_part.dim() == 2):
            real_part = real_part.unsqueeze(0)
            imag_part = imag_part.unsqueeze(0)

        recombined = torch.cat([real_part, imag_part], dim=1)

        inverse_transform = F.conv_transpose1d(recombined,
                                               self.inverse_basis,
                                               stride=self.hop_length,
                                               padding=0)

        padded = int(self.filter_length // 2)
        if length is None:
            if self.center:
                inverse_transform = inverse_transform[:, :, padded:-padded]
        else:
            if self.center:
                inverse_transform = inverse_transform[:, :, padded:]
            inverse_transform = inverse_transform[:, :, :length]

        return inverse_transform
