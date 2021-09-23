from gnet.model.train import *


train('b0_ce', 'stft_1024_magn', 'small', 'data')
# successive_train('paper', 'paper_wd', pre_cfg_name='stft', dm_cfg_name='small', data_path='data')