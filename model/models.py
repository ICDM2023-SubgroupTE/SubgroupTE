import numpy as np
import random
from model.loss import SubgroupTE_loss
from copy import deepcopy
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.clustering import ClusterAssignment

SEED = 1111
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class SubgroupTE(nn.Module):
    def __init__(self, config):
        super(SubgroupTE, self).__init__()
        
        self.input_dim = config['input_dim']
        emb_dim = config['emb_dim']
        out_dim = config['out_dim']
        n_layer = config['n_layers']
        self.n_clusters = config['n_clusters']
        nhead = 5
        alpha, beta, gamma = config['alpha'],config['beta'],config['gamma']
        self.w = 1
        self.criterion = partial(SubgroupTE_loss, alpha=alpha, beta=beta, gamma=gamma)
        
        self.embedding = nn.Sequential(
            nn.Linear(self.input_dim, self.input_dim), nn.ReLU()
            )
            
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.input_dim, nhead=nhead, batch_first=True, dim_feedforward=emb_dim) 
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=n_layer)
        self.init_treat_out = nn.Linear(self.input_dim, 1)
        self.init_control_out = nn.Linear(self.input_dim, 1)
        
        self.assignment = ClusterAssignment(self.n_clusters, self.w)

        outcomemodel = nn.Sequential(
            nn.Linear(self.input_dim+self.n_clusters, out_dim), nn.ReLU(),
            nn.Linear(out_dim, out_dim), nn.ReLU(),
            nn.Linear(out_dim, 1)
            ) 
        
        self.control_out = deepcopy(outcomemodel)
        self.treat_out = deepcopy(outcomemodel)
        self.propensity = nn.Sequential(
            nn.Linear(self.input_dim+self.n_clusters, 1), nn.Sigmoid()
            )
    
    def get_features(self, x): #in: (batch, feature)
        n_samples = x.size(0)
        x = self.embedding(x)
        if len(x.shape) == 2:
            x = x.unsqueeze(1) #in: (batch, seq=1, feature)
        features = self.transformer_encoder(x)[:,-1]
        y1_pred_init = self.init_treat_out(features)
        y0_pred_init = self.init_control_out(features)
        return features, y0_pred_init, y1_pred_init

    def subgrouping(self, x):
        return self.assignment(x)
    
    def predict(self, x, t, y):
        features, y0_pred_init, y1_pred_init = self.get_features(x)
        te = y1_pred_init - y0_pred_init
        clusters = self.subgrouping(te)               # subgroup probability
        features = torch.cat((features, clusters), 1) # concat features and subgroup probability
        
        y0_pred = self.control_out(features)
        y1_pred = self.treat_out(features)
        t_pred = self.propensity(features)
        loss = self.criterion(y, t, t_pred, y0_pred, y1_pred, y0_pred_init, y1_pred_init)
        return loss, y0_pred, y1_pred, t_pred, clusters
    
    def update_centers(self, x, x_lengths): 
        _, y0_pred_init, y1_pred_init = self.get_features(x, x_lengths)
        te = y1_pred_init - y0_pred_init
        self.assignment._adjust_centers(te)
        
        cluster_out = self.subgrouping(te)
        cluster_out = torch.argmax(cluster_out, dim=1)
        self.assignment._update_centers(te, cluster_out)
        
    def weights_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
                    
                    
                    
                    
                    
