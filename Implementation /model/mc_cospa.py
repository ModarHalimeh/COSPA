from model.istft import ISTFT
import torch.optim.lr_scheduler as lrSched
import helpers.loss_functions as loss_f
from torch.optim import Adam
import scipy.io
import torch
import pytorch_lightning as pl
import numpy as np
from model.CRUNet import CRUNet
from model.compressor import compressor
from model.compandor import compandor
from model.decoder import decoder
import os

n_fft       = 1024
hop_length  = 512
device = 'cuda:0' # This can be changed to whatever device you have
window = torch.hann_window(n_fft).to(device)
stft = lambda x: torch.stft(x, n_fft, hop_length, window=window,  center=True)
istft = ISTFT(n_fft, hop_length, window='hanning').to(device)

class COSPA(pl.LightningModule):
    def __init__(self, cfg, cfgEncoder, cfgCompressor, cfgCompandor, cfgDecoder):
        super(COSPA, self).__init__()
        self.nr_channels = cfg['nr_channels']
        self.net_name    = cfg['NetName']
        self.nfft        = hop_length + 1

        self.encoder           = CRUNet(cfgEncoder) 

        self.compressor_noise  = compressor(cfgCompressor) # define one conv1d to compress noise signal estiamtes
        self.compressor_source = compressor(cfgCompressor) # define one conv1d to compress source signal estiamtes

        self.compandor         = compandor(cfgCompandor, self.nfft, self.nr_channels) 

        self.decoder           = decoder(cfgDecoder)


    def get_ratio_mask(self, o_real, o_imag):
        # processes unbounded mask O to get the final mask M 
        mag = torch.sqrt(o_real ** 2 + o_imag ** 2)
        phase = torch.atan2(o_imag, o_real)
        mag = torch.tanh(mag)
        return mag, phase

    def apply_mask(self, xr, xi, mag, phase):
        # apply the complex-valued to mask to a complex valued signal (xr, xi)
        mag = mag * torch.sqrt(xr ** 2 + xi ** 2)
        phase = phase + torch.atan2(xi, xr)
        return mag * torch.cos(phase), mag * torch.sin(phase)

    def forward(self, xr, xi):
        
        ########## Spatial encoders ###################
        # estimate single-channel complex mask 
        # -> apply it to each channel to obtain noise/source estimates
        # -> collect said estimates in a complex-valued vector
        
        # extract first mic signal 
        xr_sc, xi_sc     = xr[..., 0], xi[..., 0]
        # estimate single-channel mask
        o_r_sc, o_i_sc   = self.encoder(xr_sc, xi_sc)
        mag_sc, phase_sc = self.get_ratio_mask(o_r_sc, o_i_sc)
        mag_sc, phase_sc = mag_sc.squeeze(), phase_sc.squeeze()

        # apply mask to first mic channel to obtain initial source and noise signal estimates
        xr_ch, xi_ch    = xr[..., 0].squeeze(dim=1), xi[..., 0].squeeze(dim=1)
        xr_source, xi_source = self.apply_mask(xr_ch, xi_ch, mag_sc, phase_sc)
        xr_noise,  xi_noise  = xr_ch-xr_source, xi_ch-xi_source

        xr_source, xi_source = xr_source.unsqueeze(dim=1), xi_source.unsqueeze(dim=1)
        xr_source, xi_source = self.compressor_source(xr_source, xi_source)
        xr_source, xi_source = xr_source.squeeze(), xi_source.squeeze()

        xr_noise, xi_noise  = xr_noise.unsqueeze(dim=1), xi_noise.unsqueeze(dim=1)
        xr_noise,  xi_noise =  self.compressor_noise(xr_noise,  xi_noise)
        xr_noise, xi_noise  = xr_noise.squeeze(), xi_noise.squeeze()
        
        # apply mask to the other mic channels to obtain initial source and noise signal estimates
        # and then append these estimates 
        for ind in range(1, self.nr_channels):
            xr_ch, xi_ch = xr[..., ind].squeeze(dim=1), xi[..., ind].squeeze(dim=1)
            xr_source_local, xi_source_local = self.apply_mask(xr_ch, xi_ch, mag_sc, phase_sc)
            xr_noise_local, xi_noise_local   = xr_ch-xr_source_local, xi_ch-xi_source_local

            xr_source_local, xi_source_local = xr_source_local.unsqueeze(dim=1), xi_source_local.unsqueeze(dim=1)
            xr_source_local, xi_source_local = self.compressor_source(xr_source_local, xi_source_local)
            xr_source_local, xi_source_local = xr_source_local.squeeze(), xi_source_local.squeeze()

            xr_noise_local, xi_noise_local = xr_noise_local.unsqueeze(dim=1), xi_noise_local.unsqueeze(dim=1)
            xr_noise_local, xi_noise_local = self.compressor_noise(xr_noise_local, xi_noise_local)
            xr_noise_local, xi_noise_local = xr_noise_local.squeeze(), xi_noise_local.squeeze()

            xr_source = torch.cat((xr_source, xr_source_local), dim=1)
            xi_source = torch.cat((xi_source, xi_source_local), dim=1)

            xr_noise = torch.cat((xr_noise, xr_noise_local), dim=1)
            xi_noise = torch.cat((xi_noise, xi_noise_local), dim=1)

        xr, xi = torch.cat((xr_source, xr_noise), dim=1), torch.cat((xi_source, xi_noise), dim=1)
        
        ########## Spatial Compandor ###################
        xr, xi = self.compandor(xr, xi)

        ########## Spatial Decoders ####################
        o_r, o_i    = self.decoder(xr[..., 0], xi[..., 0])
        
        # Process and collect final masks
        mag, phase  = self.get_ratio_mask(o_r, o_i)
        mag, phase  = mag.unsqueeze(dim=3), phase.unsqueeze(dim=3)
        for ind in range(1, self.nr_channels):
            o_r, o_i = self.decoder(xr[..., ind], xi[..., ind])
            mag_local, phase_local = self.get_ratio_mask(o_r, o_i)
            mag_local, phase_local = mag_local.unsqueeze(dim=3), phase_local.unsqueeze(dim=3)
            mag, phase             = torch.cat((mag, mag_local), dim=3),  torch.cat((phase, phase_local), dim=3)
        return mag, phase


    def training_step(self, batch, batch_idx):
        mix_mic_logf, mix_mic_f, mix_target, source_target, mix_mic, source_mic, noise_mic, _ = batch #some of these signals are not used and can be removed later
        
        # extract real/imaginary components of the microphone signals 
        xr, xi      = mix_mic_f[..., 0, :self.nr_channels].unsqueeze(dim=1), mix_mic_f[..., 1, :self.nr_channels].unsqueeze(dim=1) 
        
        # Estimate the multichannel complex-valued masks by the COSPA 
        mag, phase   = self(xr, xi) 

        # Apply said masks to the noisy microphone signals 
        y = 0
        xr_m, xi_m  = self.apply_mask(xr[..., 0].squeeze(), xi[..., 0].squeeze(), mag[..., 0], phase[..., 0])
        
        # use istft to obtain the time-domain signal estimate 
        estimated   = istft(xr_m.squeeze(), xi_m.squeeze())
        y           = y + estimated
        for ind in range(1, self.nr_channels):
            xr_m, xi_m  = self.apply_mask(xr[..., ind].squeeze(), xi[..., ind].squeeze(), mag[..., ind], phase[..., ind])
            estimated   = istft(xr_m.squeeze(), xi_m.squeeze())
            y           = y + estimated
        estimated = y.squeeze()

        # define training target signal 
        source_target    = source_target
        source_target    = stft(source_target).squeeze()
        source_target    = istft(source_target[..., 0], source_target[..., 1])
        source_target    = source_target.squeeze()

        # calculate the shortest signal length
        L = np.minimum(len(estimated[0, :]), len(source_target[0, :]))
        
        # Calculate loss 
        loss = loss_f.SNR_loss(source_target[:, :L].squeeze(), estimated[:, :L].squeeze())

        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=False)

        return loss

    def validation_step(self, batch, batch_idx):

        mix_mic_logf, mix_mic_f, mix_target, source_target, mix_mic, source_mic, noise_mic, _ = batch # some of these signals is not used and can be removed/ignored

        with torch.no_grad():
            xr, xi = mix_mic_f[..., 0, :self.nr_channels].unsqueeze(dim=1), mix_mic_f[..., 1,
                                                                            :self.nr_channels].unsqueeze(dim=1)

            mag, phase = self(xr, xi)

            y = 0
            xr_m, xi_m = self.apply_mask(xr[..., 0].squeeze(), xi[..., 0].squeeze(), mag[..., 0], phase[..., 0])
            estimated = istft(xr_m.squeeze(), xi_m.squeeze())
            y = y + estimated
            for ind in range(1, self.nr_channels):
                xr_m, xi_m = self.apply_mask(xr[..., ind].squeeze(), xi[..., ind].squeeze(), mag[..., ind], phase[..., ind])
                estimated = istft(xr_m.squeeze(), xi_m.squeeze())
                y = y + estimated
            estimated = y.squeeze()

            source_target = source_target
            source_target = stft(source_target).squeeze()
            source_target = istft(source_target[..., 0], source_target[..., 1])
            source_target = source_target.squeeze()

            L = np.minimum(len(estimated[0, :]), len(source_target[0, :]))

            loss = loss_f.SNR_loss(source_target[:, :L].squeeze(), estimated[:, :L].squeeze())

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=False)
        return loss

    def test_step(self, batch, batch_idx):
        mix_mic_logf, mix_mic_f, source_target, mix_mic, source_mic, noise_mic, dir_noise, filename = batch

        xr, xi = mix_mic_f[..., 0, :self.nr_channels].unsqueeze(dim=1), mix_mic_f[..., 1, :self.nr_channels].unsqueeze(dim=1)

        mag, phase = self(xr, xi)

        y = 0
        xr_m, xi_m = self.apply_mask(xr[..., 0].squeeze(), xi[..., 0].squeeze(), mag[..., 0], phase[..., 0])
        estimated = istft(xr_m.squeeze(), xi_m.squeeze())
        y = y + estimated
        for ind in range(1, self.nr_channels):
            xr_m, xi_m = self.apply_mask(xr[..., ind].squeeze(), xi[..., ind].squeeze(), mag[..., ind], phase[..., ind])
            estimated = istft(xr_m.squeeze(), xi_m.squeeze())
            y = y + estimated
        estimated = y.squeeze()


        # save results as .mat structures into a given results directory 
        for fileInd in range(0, len(filename)):
            s_hat    = estimated[fileInd, :]
            y_mix    = mix_mic[fileInd, :, :]
            file     = filename[fileInd]

            sig= {}
            sig['s_hat']            = s_hat.detach().cpu().numpy()
            sig['y']                = y_mix.detach().cpu().numpy()

            if not os.path.isdir('./results/' + self.net_name):
                os.mkdir('./results/' + self.net_name)

            scipy.io.savemat('./results/' + self.net_name + '/' + file + '.mat', sig)

    def configure_optimizers(self):
        optimizer= Adam(self.parameters(), lr=1e-3)
        lrLambda = lambda epoch: 0.98**epoch
        scheduler= lrSched.LambdaLR(optimizer, lrLambda)
        return [optimizer], [scheduler]





