from gnet.model.train import *


train('b0_bce', 'stft', 'small', 'data')
# successive_train('paper', 'paper_wd', pre_cfg_name='stft', dm_cfg_name='small', data_path='data')