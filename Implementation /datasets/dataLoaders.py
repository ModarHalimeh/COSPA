#This dataloader is specific to the data structure which we've used. You are strongly advised to use your own dataloader.

import torch
import os
import h5py
import numpy as np
from torch.utils import data
import scipy.io as io

hdf5_directory = 'hdf5-filepath' # filepath used to save the dataset 

sig_duration    = 7
sample_freq     = 16000
pi              = 3.14159265359
cfg             = {}

cfg['frame_shift']     = 512
cfg['block_length']    = cfg['frame_shift']*2
cfg['nfft']            = cfg['block_length']
window = torch.hann_window(cfg['nfft'] )
stft = lambda x: torch.stft(x, cfg['nfft'], cfg['frame_shift'], window=window,  center=True, return_complex=False)


def load_data_list(folder):
    # Load_data_list is used to search for .mat files within a give -folder directory- and return a list of found .mat files as -dataList-
    directory = folder
    filelist = os.listdir(directory)
    dataList = [f for f in filelist if f.endswith(".mat")]

    print("datalist loaded...")
    return dataList


def create_dataset(directory, outFile):
    # This functions writes the .mat files (contained in directory) into a single hdf5 file outFile 
    dataList = load_data_list(directory)

    print('extracting data from MATLAB files. \n')

    if not (os.path.isdir(hdf5_directory)):
        os.mkdir(hdf5_directory)

    nrDataFiles     = len(dataList)
    fullAxis        = np.random.permutation(nrDataFiles)
    trainingList    = fullAxis
    trainingDatalist = [dataList[i] for i in trainingList]
    sigLength = int(np.floor(sig_duration * sample_freq))

    with h5py.File(hdf5_directory + outFile , "w", swmr=True, libver='latest') as f:
        dt = 'f'
        for fileName in trainingDatalist:
            # each .mat file contains a structure sig, which includes: 
            #       s_mic: source image in the microphone signals 
            #       y: noisy mic signals 
            #       m_mic: music source spatial image
            #       n_mic: microphone self-noise signals 
            #       n_bg_mic: noise source spatial images 
            #       s_mvdr: training target signal as described in the paper
            data = io.loadmat(os.path.join(directory, fileName), struct_as_record=False)
            source_mic      = data['sig'][0, 0].s_mic[0:sigLength, :].astype(np.float32)
            mix_mic         = data['sig'][0, 0].y[0:sigLength, :].astype(np.float32)
            noise_mic       = data['sig'][0, 0].m_mic[0:sigLength, :].astype(np.float32)+ data['sig'][0, 0].n_mic[0:sigLength, :].astype(np.float32) + data['sig'][0, 0].n_bg_mic[0:sigLength, :].astype(np.float32)
            target_mvdr     = data['sig'][0, 0].s_mvdr[0:sigLength, :].astype(np.float32)
            
            a_opt = data['sig'][0, 0].a_opt.astype(np.float32)
            grp = f.create_group(fileName[:-4])
            grp.create_dataset('a_opt', data=a_opt)

            grp.create_dataset('source_mic',    data=source_mic)
            grp.create_dataset('mix_mic',       data=mix_mic)
            grp.create_dataset('noise_mic',     data=noise_mic)
            grp.create_dataset('target_mvdr',     data=target_mvdr)
            
            # if this is an hfdf5 for the test data, save the additional white-noise sources signals too
            if "test" in outFile:
                dir_noise = data['sig'][0, 0].dir_noise[0:sigLength, :, :].astype(np.float32)
                grp.create_dataset('dir_noise', data=dir_noise)

    print('saved data. \n')

class mcDNN_train_dataset(data.Dataset):

    def __init__(self, cfg):
        self.frame_shift    = cfg['frame_shift']
        self.win_length     = cfg['block_length']
        self.fs             = sample_freq
        self.sig_duration   = sig_duration
        self.outPATH        = cfg['outPATH']
        self.inPATH         = cfg['inPATH']

        if not os.path.isfile(hdf5_directory+self.outPATH):
            print('============= Creating Dataset ===================')
            create_dataset(self.inPATH, self.outPATH)

        with h5py.File(hdf5_directory+self.outPATH, 'r', swmr=True, libver='latest') as f:
            self.samples = list(f.keys())
            self.nr_samples = len(f)

        self.reader = None

    def __len__(self):
        return self.nr_samples

    def __getitem__(self, idx):

        if self.reader is None:
            self.reader = h5py.File(hdf5_directory+self.outPATH, 'r', swmr=True, libver='latest')

        if torch.is_tensor(idx):
            idx = idx.tolist()

        item_name = self.samples[idx]
        item      = self.reader[item_name]

        mix_mic         = item['mix_mic'][()].astype(np.float32)
        mix_mic         = torch.from_numpy(mix_mic)
        norm_fact       = torch.max(torch.max(torch.abs(mix_mic)))
        mix_mic_f       = STFT_across_channels(mix_mic)


        source_target   = item['target_mvdr'][()].astype(np.float32)
        source_target   = source_target[511:, :]
        source_target   = torch.from_numpy(source_target)


        source_mic      = item['source_mic'][()].astype(np.float32)
        source_mic      = torch.from_numpy(source_mic)
        source_mic      = STFT_across_channels(source_mic)

        noise_mic      = item['noise_mic'][()].astype(np.float32)
        noise_mic      = torch.from_numpy(noise_mic)
        noise_mic      = STFT_across_channels(noise_mic)


        mix_target      = item['mix_mvdr'][()].astype(np.float32)
        #mix_target      = mix_target[511:, :] # might need to account for algorithmic delays, depends on your target
        mix_target      = torch.from_numpy(mix_target)


        mix_mic_amp     = mix_mic_f[..., 1, :]**2 + mix_mic_f[..., 0, :]**2
        mix_mic_phase   = torch.atan2(mix_mic_f[..., 1, :], mix_mic_f[..., 0, :])
        mix_mic_f_log   = torch.sqrt(mix_mic_amp)
        mix_mic_f_log   = torch.div(mix_mic_f_log, torch.max(torch.abs(mix_mic_f_log[..., :])))
        mix_mic_phase   = torch.div(mix_mic_phase, pi)

        mix_mic_logf    = torch.cat((mix_mic_f_log.unsqueeze(dim=2), mix_mic_phase.unsqueeze(dim=2)), dim=2)

        assert(not torch.isnan(torch.max(mix_mic_f_log)))
        assert (not torch.isnan(torch.max(mix_mic_phase)))

        batch = mix_mic_logf, mix_mic_f, mix_target, source_target, mix_mic, source_mic, noise_mic, item_name


        return batch

class mcDNN_test_dataset(data.Dataset):

    def __init__(self, cfg):
        self.frame_shift = cfg['frame_shift']
        self.win_length = cfg['block_length']
        self.fs = sample_freq
        self.sig_duration = sig_duration
        self.outPATH = cfg['outPATH']
        self.inPATH = cfg['inPATH']

        if not os.path.isfile(hdf5_directory + self.outPATH):
            print('============= Creating Dataset ===================')
            create_dataset(self.inPATH, self.outPATH)

        with h5py.File(hdf5_directory+ self.outPATH, 'r', swmr=True, libver='latest') as f:
            self.samples = list(f.keys())
            self.nr_samples = len(f)

        self.reader = None

    def __len__(self):
        return self.nr_samples

    def __getitem__(self, idx):

        if self.reader is None:
            self.reader = h5py.File(hdf5_directory + self.outPATH, 'r', swmr=True, libver='latest')

        if torch.is_tensor(idx):
            idx = idx.tolist()

        item_name = self.samples[idx]
        item = self.reader[item_name]

        mix_mic = item['mix_mic'][()].astype(np.float32)
        mix_mic = torch.from_numpy(mix_mic)
        mix_mic_f = STFT_across_channels(mix_mic)

        mix_mic_amp = mix_mic_f[:, :, 1, :] ** 2 + mix_mic_f[:, :, 0, :] ** 2
        mix_mic_phase = torch.atan2(mix_mic_f[:, :, 1, :], mix_mic_f[:, :, 0, :])
        mix_mic_f_log = torch.sqrt(mix_mic_amp)
        mix_mic_f_log = torch.div(mix_mic_f_log, torch.max(torch.abs(mix_mic_f_log[..., :])))#normalize_across_channels(mix_mic_f_log)
        mix_mic_phase = torch.div(mix_mic_phase, pi)
        mix_mic_logf  = torch.cat((mix_mic_f_log.unsqueeze(dim=2), mix_mic_phase.unsqueeze(dim=2)), dim=2)

        noise_mic = item['noise_mic'][()].astype(np.float32)
        noise_mic = torch.from_numpy(noise_mic)
        noise_mic = STFT_across_channels(noise_mic)

        source_mic = item['source_mic'][()].astype(np.float32)
        source_mic = torch.from_numpy(source_mic)
        source_mic = STFT_across_channels(source_mic)

        dir_noise = item['dir_noise'][()].astype(np.float32) # dir_noise are the white noise sources used in the paper to generated the beampattern
        dir_noise = torch.from_numpy(dir_noise)
        nr_angles = len(dir_noise[0, 0, :])
        nr_frames = len(mix_mic_f[0, :, 0, 0])
        nr_f_bins = len(mix_mic_f[:, 0, 0, 0])

        dir_noise_f = torch.zeros((nr_f_bins, nr_frames, 2, 5, 0))
        for ind in range(1, nr_angles):
            dir_noise_curr = STFT_across_channels(dir_noise[:, :, ind])
            dir_noise_f = torch.cat((dir_noise_f,
                                     dir_noise_curr.unsqueeze(dim=4)), dim=4)
        batch = mix_mic_logf, mix_mic_f, mix_mic, source_mic, noise_mic, dir_noise_f, item_name

        assert (not torch.isnan(torch.max(mix_mic_f_log)))
        assert (not torch.isnan(torch.max(mix_mic_phase)))

        return batch


def STFT_across_channels(input):
    nr_channels = input.shape
    nr_channels = nr_channels[-1]

    output      = stft(input[:, 0]).unsqueeze(dim=3)
    for channelInd in range(1, nr_channels):
        output = torch.cat((output, stft(input[:, channelInd]).unsqueeze(dim=3)), dim=3)

    return output

def normalize_across_channels(input):
    nr_channels = input.shape
    nr_channels = nr_channels[-1]

    output              = input
    output[..., 0]      = torch.div(input[..., 0], torch.max(torch.abs(input[..., 0])))
    for channelInd in range(1, nr_channels):
        output[..., channelInd]      = torch.div(input[..., channelInd], torch.max(torch.abs(input[..., channelInd])))

    return output
