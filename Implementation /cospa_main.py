networkName = 'cospa_shorttest'

cfg = {}
cfg['nr_channels']          = 5
cfg['NetName']              = networkName + '_' + str(cfg['nr_channels'])

cfg['frame_shift']     = 512
cfg['block_length']    = cfg['frame_shift']*2
cfg['nfft']            = cfg['block_length']



cfgEncoder = {}
cfgEncoder['encoders'] = {}
cfgEncoder['encoders'][0] = [1,  64, [7, 2], [4, 1], [6, 1]]
cfgEncoder['encoders'][1] = [64, 32, [5, 1], [2, 1], [4, 0]]
cfgEncoder['encoders'][2] = [32, 16, [4, 2], [2, 1], [3, 1]]
cfgEncoder['encoders'][3] = [16, 16, [4, 1], [2, 1], [3, 0]]

cfgEncoder['decoders'] = {}
cfgEncoder['decoders'][0] = [16+16,   16,  [4, 1], [2, 1], [3, 0]]
cfgEncoder['decoders'][1] = [16+16,   32,  [4, 2], [2, 1], [3, 1]]
cfgEncoder['decoders'][2] = [32+32,   64,  [5, 1], [2, 1], [4, 0]]
cfgEncoder['decoders'][3] = [64+64,    1,  [7, 2], [4, 1], [6, 1]]

cfgEncoder['GRUdim']        = 120
cfgEncoder['GRUIN']         = 304


cfgCompressor = {}
cfgCompressor    = [1,  1, [7, 1], [2, 1], [6, 0]]

cfgCompandor = {}
cfgCompandor['layer1_in']  = 260*cfg['nr_channels']*2
cfgCompandor['layer1_out'] = 128

cfgCompandor['gru_in']  = 128
cfgCompandor['gru_out'] = 128

cfgCompandor['layer2_in']  = 128
cfgCompandor['layer2_out'] = 513*cfg['nr_channels']

cfgDecoder = {}
cfgDecoder['layer1_in']  = 513
cfgDecoder['layer1_out'] = 256

cfgDecoder['layer2_in']  = 256
cfgDecoder['layer2_out'] = 256

cfgDecoder['layer3_in']  = 256
cfgDecoder['layer3_out'] = cfg['frame_shift']+1


cfg['outPATH']          = 'train_data.hdf5'
cfg['inPATH']           = 'Directory_path_to_source_training_mat_files'
cfg['outPATH_test']     = 'test_data2.hdf5'
cfg['inPATH_test']      = 'Directory_path_to_source_testing_mat_files'


cfg['hop_length']    = cfg['frame_shift']
cfg['n_fft']         = cfg['block_length']

cfg['batch_size']       =10 


from datasets import dataLoaders
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
import numpy as np
from model import mc_cospa
from pytorch_lightning import loggers as pl_loggers


synth_set = dataLoaders.mcDNN_train_dataset(cfg)
train_len = len(synth_set) - np.floor_divide(len(synth_set), 100)
val_len = np.floor_divide(len(synth_set), 100)
train_set, val_set = torch.utils.data.random_split(synth_set, [train_len, val_len])
trainLoader, valLoader = DataLoader(train_set, batch_size=cfg['batch_size'], num_workers=4, shuffle=True, drop_last=True), \
                         DataLoader(val_set, batch_size=3, num_workers=4, drop_last=True)


cfg['outPATH']          = cfg['outPATH_test']
cfg['inPATH']           = cfg['inPATH_test']
test_set = dataLoaders.mcDNN_test_dataset(cfg)
testLoader             = DataLoader(test_set, batch_size=5, num_workers=3)


net = mc_cospa.COSPA(cfg, cfgEncoder, cfgCompressor, cfgCompandor, cfgDecoder)
TBlogger = pl_loggers.TensorBoardLogger('logs/' + cfg['NetName'])
cfg['checkpoint_name']       = './results/model_baseline.tar'

trainer = pl.Trainer(logger=TBlogger, log_every_n_steps=4, gpus=1, max_epochs=26)

trainer.fit(net, trainLoader, val_dataloaders=valLoader)
trainer.test(net, test_dataloaders=testLoader)

torch.save(net, './results/cospa_net')
