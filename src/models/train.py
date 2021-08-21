import argparse
from pytorch_lightning import Trainer
from litmodel import LitModel
import os

parser = argparse.ArgumentParser(description='Train Lightning Module.')
parser.add_argument('--yaml', type=str, required=True,
                    help='YAML config file containing the lit model description.')
parser.add_argument('--data', type=str, required=True, 
                    help='Path to training dataset. The training folder should contain train_labels.csv with keys [id, target]. Wildcards are supported.')
args = parser.parse_args()

cfg = os.path.join(os.path.split(__file__)[0], 'config', args.yaml)
data_path = args.data

litmodel = LitModel(cfg)