import argparse
import collections
import os
import numpy as np
import pandas as pd
import model.loss as module_loss
import model.metric as module_metric
from model.models import SubgroupTE
from parse_config import ConfigParser
from trainer.trainer import Trainer
from utils import Load_split_dataset

import torch
import torch.nn as nn

# fix random seeds for reproducibility
SEED = 1111
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def main(m_type, config):
    config, train_set, valid_set, test_set = Load_split_dataset(config)
        
    # build model architecture, initialize weights, then print to console    
    model = SubgroupTE(config['hyper_params'])
    model.weights_init()  
    
    logger = config.get_logger('train') 
    logger.info(model)
    logger.info("-"*100)

    # get function handles of metrics
    metrics = [getattr(module_metric, met) for met in ['IPTW', 'Acc_treatment', 'Acc_outcome']]

    # build optimizer
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)

    trainer = Trainer(model, 
                      optimizer,
                      metrics,
                      config,
                      train_set,
                      valid_set,
                      test_set)

    log = trainer.train()
    return log


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', type=str, 
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default="0", type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('--data', type=str)
    params = args.parse_args()
    
    config = ConfigParser.from_args(args)
    config['data_loader']['data'] = params.data

    
    log = main(params.model, config)
        

    if os.path.isfile(save_file):
        save_df.to_csv(save_file, index=False, mode='a', header=False)
    else:
        save_df.to_csv(save_file, index=False, header=True)
