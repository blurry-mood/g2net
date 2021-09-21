from gnet.model.train import *


# train('paper', 'stft', 'small', 'data')
successive_train('paper', 'paper_wd', pre_cfg_name='stft', dm_cfg_name='small', data_path='data')