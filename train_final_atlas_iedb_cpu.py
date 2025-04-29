import os
import sys
from argparse import ArgumentParser
import torch
from torch.utils.data import DataLoader
import torch.multiprocessing
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, StochasticWeightAveraging, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
import wandb

from model.MLP_f1_strong_dropout import MLP 
from dataset.Dataset import csv_dataset
import csv
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import LabelEncoder
from collections import Counter
from imblearn.over_sampling import SMOTE
from pandas import read_csv


def get_next_version(root_dir):
  _fs = pl.utilities.cloud_io.get_filesystem(root_dir)

  try:
      listdir_info = _fs.listdir(root_dir)
  except OSError:
    print("Missing logger folder: %s", root_dir)
    return 0

  existing_versions = []
  for listing in listdir_info:
      d = listing["name"]
      bn = os.path.basename(d)
      if _fs.isdir(d) and bn.startswith("version_"):
          dir_ver = bn.split("_")[1].replace("/", "")
          existing_versions.append(int(dir_ver))
  if len(existing_versions) == 0:
      return 0

  return max(existing_versions) + 1

torch.multiprocessing.set_sharing_strategy('file_system')
torch.backends.cudnn.determinstic = True
torch.backends.cudnn.benchmark = False
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


parser = ArgumentParser()
parser.add_argument('--num_workers', type=int, default=56) # 58
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--val_check_n', type=int, default=5)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--train_path', type=str)
parser.add_argument('--val_path', type=str)
parser.add_argument('--test_path', type=str)
parser.add_argument('--num_classes', type=int, default=3)
parser.add_argument('--checkpoint', type=str, default=None)
parser.add_argument('--metric', type=str, default='f1.2')
parser.add_argument('--test', action='store_true')
parser.add_argument('--tune', default = False) # , action='store_true'
parser = MLP.add_model_specific_args(parser) # CAUTION: STATIC METHOD. IF ADDING MODEL SPECIFIC ARGS, MUST BE DONE IN THIS CLASS
parser = pl.Trainer.add_argparse_args(parser)
hparams = parser.parse_args()

pl.seed_everything(hparams.seed)

def train():
  model_name = 'BindinMLP_final_atlas_iedb'
  log_dir = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])),'logs',model_name)
  train_dataset = csv_dataset(hparams.train_path)
  val_dataset = csv_dataset(hparams.val_path)

  ## do the smote resampling
  df = read_csv(hparams.train_path, sep='\t' )
  data = df.values
  X, y = data[:, :-1], data[:, -1]
  #y = LabelEncoder().fit_transform(y)
  counter = Counter(y)
  new_sizes_groups = round(len(y)/3)
  #undersample = RandomUnderSampler(sampling_strategy={0.0:new_sizes_groups, 1.0:counter[1], 2.0:counter[2]})
  undersample = RandomUnderSampler(sampling_strategy={0.0:new_sizes_groups, 1.0:counter[1], 2.0:new_sizes_groups})

  #undersample = RandomUnderSampler(sampling_strategy={0.0:new_sizes_groups, 1.0:counter[1], 2.0:new_sizes_groups})
  X, y = undersample.fit_resample(X, y)
  oversample = SMOTE(sampling_strategy={0.0:new_sizes_groups, 1.0:new_sizes_groups, 2.0:new_sizes_groups})
  #oversample = SMOTE(sampling_strategy={0.0:new_sizes_groups, 1.0:new_sizes_groups, 2.0:new_sizes_groups})

  X, y = oversample.fit_resample(X, y)

  xpd = pd.DataFrame(X)
  xpd['11'] = y.tolist()
  with open(os.path.dirname(hparams.train_path) + '/train_data_balanced.tsv', 'w') as f:
    writer = csv.writer(f, delimiter='\t')
    writer.writerows(pd.DataFrame.to_numpy(xpd))
  #train_dataset = xpd
  train_dataset = csv_dataset(os.path.dirname(hparams.train_path) + '/train_data_balanced.tsv')
  hparams.input_size=train_dataset.get_input_size()


  del xpd, X, df, data


  version = get_next_version(log_dir)
  log_folder = os.path.join(log_dir,f'version_{version}')

  callbacks = []
  callbacks.append(ModelCheckpoint(monitor=f'val.{hparams.metric}',mode='max',filename=f'best_val_{hparams.metric}_{version}',dirpath=log_folder))
  callbacks.append(LearningRateMonitor(logging_interval='epoch'))
  callbacks.append(EarlyStopping(monitor=f'val.{hparams.metric}', min_delta=0.00, patience=20/hparams.val_check_n, verbose=False, mode='max'))

  model = MLP(hparams)

  train_loader = DataLoader(train_dataset,  num_workers=hparams.num_workers, batch_size=hparams.batch_size, pin_memory=True, persistent_workers=True, shuffle=True)
  val_loader = DataLoader(val_dataset,  num_workers=hparams.num_workers, batch_size=hparams.batch_size, pin_memory=True, persistent_workers=True, shuffle=False)


  if not hparams.checkpoint:
    trainer = pl.Trainer.from_argparse_args(hparams, logger=logger, gpus=1, check_val_every_n_epoch=hparams.val_check_n, callbacks=callbacks, enable_progress_bar=True, max_epochs=hparams.max_epochs, accelerator='cpu')
    trainer.fit(model, train_loader, val_loader)
    logger.log_metrics({f'best.val.{hparams.metric}': model.best_val_metric})
    ckpt_path = os.path.join(log_folder,f'best_val_{hparams.metric}_{model.best_val_metric}.ckpt')
  else:
    trainer = pl.Trainer.from_argparse_args(hparams, logger=logger, gpus=1, resume_from_checkpoint=hparams.checkpoint)
    ckpt_path = hparams.checkpoint

  # Test if needed
  if hparams.test:
    test_dataset = csv_dataset(hparams.test_path)
    test_loader = DataLoader(test_dataset,  num_workers=hparams.num_workers, batch_size=hparams.batch_size, pin_memory=True, persistent_workers=True, shuffle=False)
    trainer.test(model, test_loader, ckpt_path=ckpt_path)

sweep_config = {
    'method': 'bayes',
    'metric': {
      'name': 'val.f1.2',
      'goal': 'maximize'
    },
    'parameters': {
        'learning_rate': {
            'values': [3e-5, 1e-5, 3e-6, 1e-6, 3e-7, 1e-7]
        },
        'weight_decay': {
            'values': [1e-3, 1e-4, 1e-5, 1e-6, 0]
        },
        'num_layers': {
            'values': [1, 2, 3, 4]
        },
        'num_nodes_per_layer': {
            'values': [64, 128, 256, 512, 1024]
        },
        'dropout_prob': {
            'values': [0.3, 0.2, 0.1, 0.05, 0.01, 0.005]
        },
        'init_type': {
            'values': ['normal', 'xavier', 'kaiming', 'orthogonal']
        }
      }

}

logger = WandbLogger(project='BindinMLP_final_atlas_iedb')
sweep_id = wandb.sweep(
    sweep=sweep_config, 
    project="BindinMLP_final_atlas_iedb")

if hparams.tune:
  wandb.agent(sweep_id, function=train)
else:
  train()
