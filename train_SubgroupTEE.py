import argparse
import collections
import os
import numpy as np
import pandas as pd
import model.loss as module_loss
import model.metric as module_metric
import model.models as model_arch
from parse_config import ConfigParser
from trainer.trainer_SubgroupTEE import Trainer_SubgroupTEE
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
    model = getattr(model_arch, m_type)
    model = model(config['hyper_params'])
    model.weights_init()  
    
    logger = config.get_logger('train') 
    logger.info('='*100)
    logger.info('    {:25s}: {}'.format("Model", m_type))
    logger.info("-"*100)
    for key, value in config['data_loader'].items():
        logger.info('    {:25s}: {}'.format(str(key), value))
    logger.info("-"*100)
    for key, value in config['hyper_params'].items():
        logger.info('    {:25s}: {}'.format(str(key), value))
    logger.info("-"*100)
    logger.info(model)
    logger.info("-"*100)

    # get function handles of loss and metrics
    if config['data_loader']['data'] == 'OUD':
        metrics = [getattr(module_metric, met) for met in ['IPTW', 'Acc_treatment', 'Acc_outcome']]

    else:
        metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)

    trainer = Trainer_SubgroupTEE(model, 
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
    args.add_argument('--model', type=str)
    args.add_argument('--shared_hidden', type=int)
    args.add_argument('--outcome_hidden', type=int)
    args.add_argument('--n_layers', type=int)
    args.add_argument('--n_clusters', type=int)
    args.add_argument('--target_drug', default = None, type=str)
    args.add_argument('--alpha', type=float, default=1.0)
    args.add_argument('--beta', type=float, default=1.0)
    args.add_argument('--gamma', type=float, default=1.0)
    params = args.parse_args()
    
    n_iters = 1
    log_dict = dict()
    for iter in range(n_iters):
        exper_name = '{}/target_drug_{}/{}/n_clusters_{}_n_layers_{}_shared_hidden_{}_outcome_hidden_{}/iter_{}'.format(
                        params.data,params.target_drug, params.model, params.n_clusters, params.n_layers, params.shared_hidden, params.outcome_hidden, iter)
        config = ConfigParser.from_args(args, exper_name)
        config['data_loader']['data'] = params.data
        if params.data == 'OUD':
            config['data_loader']['batch_size'] = 32
        config['data_loader']['target_drug'] = params.target_drug
        config['hyper_params']['n_clusters'] = params.n_clusters
        config['hyper_params']['n_layers'] = params.n_layers
        config['hyper_params']['shared_hidden'] = params.shared_hidden
        config['hyper_params']['outcome_hidden'] = params.outcome_hidden
        config['hyper_params']['alpha'] = params.alpha
        config['hyper_params']['beta'] = params.beta
        config['hyper_params']['gamma'] = params.gamma
        
        log = main(params.model, config)
        
        for key, value in log.items():
            if iter == 0:
                log_dict[key] = [value]
            else:
                log_dict[key].append(value)
                    
    save_dict = dict()
    save_dict['model'] = [params.model]
    save_dict['data'] = [params.data]
    save_dict['target_drug'] = [params.target_drug]
    save_dict['n_iters'] = [n_iters]
    for key, value in config['hyper_params'].items():
        save_dict[key] = [value]
    for key, value in config['optimizer'].items():
        save_dict[key] = [str(value)]
    for key, value in log_dict.items():
        if 'clusters' in key or 'counts' in key:
            save_dict[key+'_avg'] = str(np.mean(value, 0))
        else:
            save_dict[key+'_avg'] = np.mean(value, 0)
            save_dict[key+'_std'] = np.std(value, 0)
        
    save_df = pd.DataFrame(save_dict) 
    
    save_dir = './results'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)  
    save_file =  os.path.join(save_dir, '{}.csv'.format(params.model))
    if params.data == 'OUD':
        save_file =  os.path.join(save_dir, '{}_OUD.csv'.format(params.model))
    if os.path.isfile(save_file):
        save_df.to_csv(save_file, index=False, mode='a', header=False)
    else:
        save_df.to_csv(save_file, index=False, header=True)