from omegaconf import OmegaConf
import pandas
import torch
import os
from tqdm.auto import tqdm
tqdm.pandas()

from .litmodel import PredictLitModel
from ..loader.datamodule import PredictDataModule

torch.backends.cudnn.benchmark = True

_HERE = os.path.split(__file__)[0]


@torch.no_grad()
def predict(chkpt_path, model_cfg_name, pre_cfg_name, dm_cfg_name, data_path):
    # configuration
    cfg = os.path.join(_HERE, 'config', model_cfg_name+'.yaml')
    cfg = OmegaConf.load(cfg)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # model
    litmodel = PredictLitModel.load_from_checkpoint(
        chkpt_path, config=cfg, preprocess_config_name=pre_cfg_name)
    litmodel.eval()
    litmodel = litmodel.to(device)

    # datamodule
    dm = PredictDataModule(data_path, dm_cfg_name)
    dm.prepare_data()

    # run inference
    data = []
    for paths, xs in tqdm(dm.predict_dataloader(), desc='Running inference'):
        preds = litmodel(xs.to(device)).cpu().tolist()
        paths = [path.decode('utf-8') for path in paths]
        data.extend(list(zip(paths, preds)))

    # save to csv
    df = pandas.DataFrame(data, columns=['id', 'target'])
    df['id'] = df['id'].progress_apply(lambda x: x.split('.')[0])
    df.to_csv('submission.csv', index=False)