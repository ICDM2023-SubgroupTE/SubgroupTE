import numpy as np
import torch
from base.base_trainer import BaseTrainer
from utils import inf_loop, MetricTracker
from model.utils.utils import padding
import torch.nn as nn
import pandas as pd
from model.metric import compute_variances

class Trainer_SubgroupTEE(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, 
                      optimizer,
                      metric_ftns,
                      config,
                      train_set,
                      valid_set,
                      test_set):
        super().__init__(model, metric_ftns, optimizer, config)
        self.config = config
        self.batch_size = config["data_loader"]["batch_size"]
        self.d_type = self.config['data_loader']['data']
        self.is_OUD = True if self.d_type == 'OUD' else False
        self.n_clusters = self.model.n_clusters
        self.input_dim = self.model.input_dim
        self.target_drug = config['data_loader']['target_drug']
        
        self.train_set = train_set
        self.valid_set = valid_set
        self.test_set = test_set
        
        self.train_n_batches = int(np.ceil(float(config["data_loader"]["n_train"]) / float(self.batch_size)))
        self.valid_n_batches = int(np.ceil(float(config["data_loader"]["n_valid"]) / float(self.batch_size)))
        self.test_n_batches = int(np.ceil(float(config["data_loader"]["n_test"]) / float(self.batch_size)))

        self.do_validation = self.valid_set is not None
        self.lr_scheduler = optimizer
        self.log_step = 4  # reduce this if you want more logs

        self.metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns])
        self.loss_dict = {
            'epoch': [],
            'PEHE': [],
            'IPTW': [],
            'within_var': [],
            'across_var': []}
        

        
    def _train_epoch(self, epoch):
        x_lengths = None                       
        
        self.metrics.reset()                
        y0_outs, y1_outs, t_outs = torch.tensor([]).to(self.device), torch.tensor([]).to(self.device), torch.tensor([]).to(self.device)
        y_trgs, t_trgs, te_trgs = torch.tensor([]).to(self.device), torch.tensor([]).to(dtype=torch.int64).to(self.device), torch.tensor([]).to(self.device)
        y0y1_outs, assigned_clusters = torch.tensor([]).to(self.device), torch.tensor([]).to(self.device)
        for index in range(self.train_n_batches):
            self.model.train()
            x = self.train_set['X'][index*self.batch_size:(index+1)*self.batch_size]
            if self.is_OUD:
                x, x_lengths = padding(x, self.input_dim)
            x = torch.from_numpy(x).float().to(self.device)
            t = torch.Tensor(
                    self.train_set['T'][index*self.batch_size:(index+1)*self.batch_size]                    
                ).to(self.device)            
            y = torch.Tensor(
                    self.train_set['Y'][index*self.batch_size:(index+1)*self.batch_size]                    
                ).to(self.device)
            
            if not self.is_OUD:
                te = torch.Tensor(
                        self.train_set['TE'][index*self.batch_size:(index+1)*self.batch_size]                    
                    ).to(self.device)

            self.optimizer.zero_grad()
            loss, y0_pred, y1_pred, t_pred, clusters = self.model.predict(x, t, y, x_lengths)
            if self.is_OUD:
                y0_pred = torch.where(y0_pred > 0.5, 1.0, 0.0)
                y1_pred = torch.where(y1_pred > 0.5, 1.0, 0.0)
            loss.backward(retain_graph=True)
            self.optimizer.step()
            self.metrics.update('loss', loss.item())
            
            if index % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f} '.format(
                    epoch,
                    self._progress(index),
                    loss.item(),
                ))
                
            clusters = torch.argmax(clusters, dim=1)
            assigned_clusters = torch.cat([assigned_clusters, clusters], 0)
            
            y0_outs = torch.cat([y0_outs, y0_pred])
            y1_outs = torch.cat([y1_outs, y1_pred])
            y_trgs = torch.cat([y_trgs, y])
            t_trgs = torch.cat([t_trgs, t])
            if not self.is_OUD:
                te_trgs = torch.cat([te_trgs, te])
            else:
                t_outs = torch.cat([t_outs, t_pred])
            
            with torch.no_grad():
                self.model.eval()
                self.model.update_centers(x, x_lengths)


        for met in self.metric_ftns:
            if not self.is_OUD:
                self.metrics.update(met.__name__, met(t_trgs, y_trgs, te_trgs, y0_outs, y1_outs))
            else:
                self.metrics.update(met.__name__, met(t_trgs, y_trgs, y0_outs, y1_outs, t_outs))
                
        log = self.metrics.result() 
        
        if self.is_OUD:
            TE_ = y1_outs - y0_outs
        else:
            TE_ = te_trgs
        within_var, across_var = compute_variances(TE_, assigned_clusters, self.n_clusters)
        clusters, counts = assigned_clusters.unique(return_counts=True, sorted=True)
        
        log.update({'clusters': clusters.tolist()})
        log.update({'counts': counts.tolist()})
        log.update({'within_var':within_var})
        log.update({'across_var':across_var})  
        
        if self.do_validation:
            val_log = self._infer(self.valid_set, self.valid_n_batches, is_valid=True, epoch=epoch)
            log.update(**{'val_' + k: v for k, v in val_log.items()})
            
        return log

        
    def _infer(self, data_set, n_batches, is_valid=False, is_test=False, epoch=None):
        x_lengths = None
        self.model.eval()
        self.metrics.reset()
        with torch.no_grad():
            x_lengths = None
            y0_outs, y1_outs, t_outs = torch.tensor([]).to(self.device), torch.tensor([]).to(self.device), torch.tensor([]).to(self.device)
            y_trgs, t_trgs, te_trgs = torch.tensor([]).to(self.device), torch.tensor([]).to(dtype=torch.int64).to(self.device), torch.tensor([]).to(self.device)
            assigned_clusters = torch.tensor([]).to(self.device)
            for index in range(n_batches):
                x = data_set['X'][index*self.batch_size:(index+1)*self.batch_size]
                if self.is_OUD:
                    x, x_lengths = padding(x, self.input_dim)
                x = torch.from_numpy(x).float().to(self.device)
                t = torch.Tensor(
                        data_set['T'][index*self.batch_size:(index+1)*self.batch_size]                    
                    ).to(self.device)            
                y = torch.Tensor(
                        data_set['Y'][index*self.batch_size:(index+1)*self.batch_size]                    
                    ).to(self.device)
                if not self.is_OUD:
                    te = torch.Tensor(
                            data_set['TE'][index*self.batch_size:(index+1)*self.batch_size]                    
                        ).to(self.device)
                
                loss, y0_pred, y1_pred, t_pred, clusters = self.model.predict(x, t, y, x_lengths)
                if self.is_OUD:
                    y0_pred = torch.where(y0_pred > 0.5, 1.0, 0.0)
                    y1_pred = torch.where(y1_pred > 0.5, 1.0, 0.0)
                self.metrics.update('loss', loss.item())
                
                clusters = torch.argmax(clusters, dim=1)
                y0_outs = torch.cat([y0_outs, y0_pred])
                y1_outs = torch.cat([y1_outs, y1_pred])
                y_trgs = torch.cat([y_trgs, y])
                t_trgs = torch.cat([t_trgs, t])
                if not self.is_OUD:
                    te_trgs = torch.cat([te_trgs, te])
                else:
                    t_outs = torch.cat([t_outs, t_pred])
                assigned_clusters = torch.cat([assigned_clusters, clusters])
                
        for met in self.metric_ftns:
            if not self.is_OUD:
                self.metrics.update(met.__name__, met(t_trgs, y_trgs, te_trgs, y0_outs, y1_outs))
            else:
                self.metrics.update(met.__name__, met(t_trgs, y_trgs, y0_outs, y1_outs, t_outs))
                
        log = self.metrics.result()
        
        if self.is_OUD:
            TE_ = y1_outs - y0_outs
        else:
            TE_ = te_trgs
        within_var, across_var = compute_variances(TE_, assigned_clusters, self.n_clusters)

        clusters, counts = assigned_clusters.unique(return_counts=True, sorted=True)
        log.update({'clusters': clusters.tolist()})
        log.update({'counts': counts.tolist()})
        log.update({'across_var':across_var})
        log.update({'within_var':within_var})
        

        if is_valid:
            self.loss_dict['epoch'].append(epoch)
            self.loss_dict['within_var'].append(within_var)
            self.loss_dict['across_var'].append(across_var)
            if self.is_OUD:
                self.loss_dict['IPTW'].append(log['IPTW'])
            else:
                self.loss_dict['PEHE'].append(log['PEHE'])

        if is_test and self.is_OUD:
            np.save(f"results/Clusters_{self.target_drug}", assigned_clusters.cpu().numpy())
            np.save(f"results/TE_{self.target_drug}", TE_.cpu().numpy())
            np.save(f"results/Dataset_{self.target_drug}", data_set)
            
        elif is_test:
            np.save(f"results/Clusters_{self.d_type}", assigned_clusters.cpu().numpy())
            np.save(f"results/TE_{self.d_type}", TE_.cpu().numpy())
            
        return log
        

    def _test_epoch(self):
        PATH = str(self.checkpoint_dir / 'model_best.pth')
        self.model.load_state_dict(torch.load(PATH)['state_dict'])
        self.model.eval()
        
        log = {}        
        train_log = self._infer(self.train_set, self.train_n_batches)
        valid_log = self._infer(self.valid_set, self.valid_n_batches)
        test_log = self._infer(self.test_set, self.test_n_batches, is_test=True)
        log.update(**{'train_' + k: v for k, v in train_log.items()})
        log.update(**{'val_' + k: v for k, v in valid_log.items()})
        log.update(**{'test_' + k: v for k, v in test_log.items()})
        
        np.save(f"results/loss_dict_{self.d_type}", self.loss_dict)
                
        self.logger.info('='*100)
        self.logger.info('Inference is completed')
        self.logger.info('-'*100)
        for key, value in log.items():
            self.logger.info('    {:20s}: {}'.format(str(key), value))  
        self.logger.info('='*100)

            
        return log
        
    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        current = batch_idx * self.batch_size
        total = self.config['data_loader']['n_train']
        return base.format(current, total, 100.0 * current / total)
        
        