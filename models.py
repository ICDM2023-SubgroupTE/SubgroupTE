import numpy as np
import random
from model.loss import dragonnet_loss, vcnet_loss, SubgroupTEE_loss, target_distribution
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.utils.vcnet_block import Dynamic_FC, Density_Block
from model.utils.transformers import TransformerEncoder, TransformerEncoderLayer, TransformerDecoderLayer, TransformerDecoder
from model.utils.mlp import MLP
from model.utils.trans_ci import TransformerModel, Embeddings
from model.utils.clustering import ClusterAssignment

SEED = 1111
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class SubgroupTEE(nn.Module):
    def __init__(self, config):
        super(SubgroupTEE, self).__init__()
        
        self.input_dim = config['input_dim']
        shared_hidden = config['shared_hidden']
        outcome_hidden = config['outcome_hidden']
        n_layer = config['n_layers']
        self.n_clusters = config['n_clusters']
        nhead = 5
        if self.input_dim > 100:
            nhead = 2
        #alpha, beta, gamma = config['alpha'],config['beta'],config['gamma']
        alpha, beta, gamma = 1.0,1.0,1.0
        self.w = 1
        self.criterion = partial(SubgroupTEE_loss, alpha=alpha, beta=beta, gamma=gamma)
        
        self.embedding = nn.Sequential(
            nn.Linear(self.input_dim, self.input_dim), nn.ReLU()
            )
            
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.input_dim, nhead=nhead, batch_first=True, dim_feedforward=shared_hidden) 
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=n_layer)
        self.init_treat_out = nn.Linear(self.input_dim, 1)
        self.init_control_out = nn.Linear(self.input_dim, 1)
        
        self.assignment = ClusterAssignment(self.n_clusters, self.w)
        
        self.control_out = nn.Sequential(
            nn.Linear(in_features=self.input_dim+self.n_clusters, out_features=outcome_hidden), nn.ReLU(),
            nn.Linear(in_features=outcome_hidden, out_features=outcome_hidden), nn.ReLU(),
            nn.Linear(in_features=outcome_hidden, out_features=1)
            ) 
        
        self.treat_out = nn.Sequential(
            nn.Linear(in_features=self.input_dim+self.n_clusters, out_features=outcome_hidden), nn.ReLU(),
            nn.Linear(in_features=outcome_hidden, out_features=outcome_hidden), nn.ReLU(),
            nn.Linear(in_features=outcome_hidden, out_features=1)
            )
        if self.input_dim == 286:
            self.control_out = nn.Sequential(
                self.control_out,
                nn.Sigmoid()
            )
            self.treat_out = nn.Sequential(
                self.treat_out,
                nn.Sigmoid()
            )
        
        self.propensity = nn.Sequential(
            nn.Linear(in_features=self.input_dim+self.n_clusters, out_features=1), nn.Sigmoid()
            )
    
    def get_features(self, x, lengths=None): 
        n_samples = x.size(0)
        x = self.embedding(x)
        if len(x.shape) == 2:
            x = x.unsqueeze(1) #in: (batch, seq=1, feature)
        features = self.transformer_encoder(x)
        if lengths is not None:
            features = features[torch.arange(n_samples), lengths-1]
        else:
            features = features[:,-1]
        y1_pred_init = self.init_treat_out(features)
        y0_pred_init = self.init_control_out(features)
        return features, y0_pred_init, y1_pred_init

    def subgrouping(self, x):
        return self.assignment(x)
    
    def predict(self, x, t, y, lengths=None):
        features, y0_pred_init, y1_pred_init = self.get_features(x, lengths)
        #te = torch.cat((y0_pred_init, y1_pred_init), 1)
        te = y1_pred_init - y0_pred_init
        clusters = self.subgrouping(te)
        features = torch.cat((features, clusters), 1)
        
        y0_pred = self.control_out(features)
        y1_pred = self.treat_out(features)
        t_pred = self.propensity(features)
        loss = self.criterion(y, t, t_pred, y0_pred, y1_pred, y0_pred_init, y1_pred_init)

        return loss, y0_pred, y1_pred, t_pred, clusters
    
    def update_centers(self, x, x_lengths): 
        _, y0_pred_init, y1_pred_init = self.get_features(x, x_lengths)
        #te = torch.cat((y0_pred_init, y1_pred_init), 1)
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
                    

################################################################################   
################### SubgroupTEE_X ##################################################
################################################################################
                    
class SubgroupTEE_X(nn.Module):
    def __init__(self, config):
        super(SubgroupTEE_X, self).__init__()
        
        self.input_dim = config['input_dim']
        shared_hidden = config['shared_hidden']
        outcome_hidden = config['outcome_hidden']
        n_layer = config['n_layers']
        self.n_clusters = config['n_clusters']
        nhead = 5
        if self.input_dim > 100:
            nhead = 2
        alpha, beta = 1.0, 1.0
        self.w = 1
        self.criterion = partial(SubgroupTEE_loss, alpha=alpha, beta=beta)
        
        self.embedding = nn.Sequential(
            nn.Linear(self.input_dim, self.input_dim), nn.ReLU()
            )
            
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.input_dim, nhead=nhead, batch_first=True, dim_feedforward=shared_hidden) 
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=n_layer)
        self.init_treat_out = nn.Linear(self.input_dim, 1)
        self.init_control_out = nn.Linear(self.input_dim, 1)
                
        self.control_out = nn.Sequential(
            nn.Linear(in_features=self.input_dim, out_features=outcome_hidden), nn.ReLU(),
            nn.Linear(in_features=outcome_hidden, out_features=outcome_hidden), nn.ReLU(),
            nn.Linear(in_features=outcome_hidden, out_features=1)
            )
        
        self.treat_out = nn.Sequential(
            nn.Linear(in_features=self.input_dim, out_features=outcome_hidden), nn.ReLU(),
            nn.Linear(in_features=outcome_hidden, out_features=outcome_hidden), nn.ReLU(),
            nn.Linear(in_features=outcome_hidden, out_features=1)
            )
        if self.input_dim == 286:
            self.control_out = nn.Sequential(
                self.control_out,
                nn.Sigmoid()
            )
            self.treat_out = nn.Sequential(
                self.treat_out,
                nn.Sigmoid()
            )
        
        self.propensity = nn.Sequential(
            nn.Linear(in_features=self.input_dim, out_features=1), nn.Sigmoid()
            )
    
    def get_features(self, x, lengths=None): 
        n_samples = x.size(0)
        x = self.embedding(x)
        if len(x.shape) == 2:
            x = x.unsqueeze(1) #in: (batch, seq=1, feature)
        features = self.transformer_encoder(x)
        if lengths is not None:
            features = features[torch.arange(n_samples), lengths-1]
        else:
            features = features[:,-1]
        y1_pred_init = self.init_treat_out(features)
        y0_pred_init = self.init_control_out(features)
        return features, y0_pred_init, y1_pred_init
    
    def predict(self, x, t, y, lengths=None):
        features, y0_pred_init, y1_pred_init = self.get_features(x, lengths)
        #te = torch.cat((y0_pred_init, y1_pred_init), 1)
        y0_pred = self.control_out(features)
        y1_pred = self.treat_out(features)
        t_pred = self.propensity(features)
        loss = self.criterion(y, t, t_pred, y0_pred, y1_pred, y0_pred_init, y1_pred_init)

        return loss, y0_pred, y1_pred, t_pred
    
    def update_centers(self, x, x_lengths): 
        _, y0_pred_init, y1_pred_init = self.get_features(x, x_lengths)
        #te = torch.cat((y0_pred_init, y1_pred_init), 1)
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
                    

################################################################################   
################### DragonNet ##################################################
################################################################################

class DragonNet(nn.Module):
    def __init__(self, config, alpha=1.0, beta=1.0):
        super(DragonNet, self).__init__()
        
        self.input_dim = config['input_dim']
        shared_hidden = config["shared_hidden"]
        outcome_hidden = config["outcome_hidden"]
        
        self.criterion = partial(dragonnet_loss, alpha=alpha, beta=beta)
        self.shared_layers = nn.Sequential(
            nn.Linear(in_features=self.input_dim, out_features=shared_hidden), nn.ReLU(),
            nn.Linear(in_features=shared_hidden, out_features=shared_hidden), nn.ReLU(),
            nn.Linear(in_features=shared_hidden, out_features=shared_hidden), nn.ReLU()
            )
        self.propensity = nn.Sequential(
            nn.Linear(in_features=shared_hidden, out_features=1), nn.Sigmoid()
            )
        self.control_out = nn.Sequential(
            nn.Linear(in_features=shared_hidden, out_features=outcome_hidden), nn.ReLU(),
            nn.Linear(in_features=outcome_hidden, out_features=outcome_hidden), nn.ReLU(),
            nn.Linear(in_features=outcome_hidden, out_features=1)
            )
        self.treat_out = nn.Sequential(
            nn.Linear(in_features=shared_hidden, out_features=outcome_hidden), nn.ReLU(),
            nn.Linear(in_features=outcome_hidden, out_features=outcome_hidden), nn.ReLU(),
            nn.Linear(in_features=outcome_hidden, out_features=1)
            )

        self.epsilon = nn.Linear(in_features=1, out_features=1)
        torch.nn.init.xavier_normal_(self.epsilon.weight)

    def forward(self, x):
        z = self.shared_layers(x) # x: covariates (n_samples * n_dim)
        t_pred = self.propensity(z)  # predicted treatment
        y0_pred = self.control_out(z)   # outcome under control
        y1_pred = self.treat_out(z) # outcome under treatment
        eps = self.epsilon(torch.ones_like(t_pred)[:, 0:1])
        return y0_pred, y1_pred, t_pred, eps
        
    def predict(self, x, t, y):
        y0_pred, y1_pred, t_pred, eps = self.forward(x)
        loss = self.criterion(y, t, t_pred, y0_pred, y1_pred, eps)
        return loss, y0_pred, y1_pred
        
    def weights_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

################################################################################   
################### VCNet ######################################################
################################################################################
                    
class VCNet(nn.Module):
    def __init__(self, config):
        super(VCNet, self).__init__()
        
        shared_hidden = config['shared_hidden'] 
        outcome_hidden = config['outcome_hidden']
        self.input_dim = config['input_dim']
        
        # cfg/cfg_density = [(ind1, outd1, isbias1, activation),....]
        self.num_grid = 10
        self.cfg_density = [(self.input_dim,shared_hidden,1), (shared_hidden,shared_hidden,1)]
        self.cfg = [(shared_hidden,outcome_hidden,1,'relu'), (outcome_hidden,1,1,'id')]
        self.degree = 2
        self.knots = [0.33, 0.66]
        self.criterion = partial(vcnet_loss, alpha=0.5, epsilon=1e-6)

        # construct the density estimator
        density_blocks = []
        for layer_idx, layer_cfg in enumerate(self.cfg_density):
            # fc layer
            if layer_idx == 0:
                # weight connected to feature
                self.feature_weight = nn.Linear(in_features=layer_cfg[0], out_features=layer_cfg[1], bias=layer_cfg[2])
                density_blocks.append(self.feature_weight)
            else:
                density_blocks.append(nn.Linear(in_features=layer_cfg[0], out_features=layer_cfg[1], bias=layer_cfg[2]))
            density_hidden_dim = layer_cfg[1]
            density_blocks.append(nn.ReLU(inplace=True))
        self.hidden_features = nn.Sequential(*density_blocks)

        self.density_estimator_head = Density_Block(self.num_grid, density_hidden_dim, isbias=1)

        # construct the dynamics network
        blocks = []
        for layer_idx, layer_cfg in enumerate(self.cfg):
            if layer_idx == len(self.cfg)-1: # last layer
                last_layer = Dynamic_FC(layer_cfg[0], layer_cfg[1], self.degree, self.knots, act=layer_cfg[3], isbias=layer_cfg[2], islastlayer=1)
            else:
                blocks.append(
                    Dynamic_FC(layer_cfg[0], layer_cfg[1], self.degree, self.knots, act=layer_cfg[3], isbias=layer_cfg[2], islastlayer=0))
        blocks.append(last_layer)
        self.Q = nn.Sequential(*blocks)

    def forward(self, x, t):
        hidden = self.hidden_features(x)
        t_hidden = torch.cat((t, hidden), 1)
        g = self.density_estimator_head(t, hidden)
        Q = self.Q(t_hidden)
        return g, Q
    
    def predict(self, x, t, y):
        n_samples = len(x)
        out = self.forward(x, t)
        loss = self.criterion(out, y)
        _, y1_pred= self.forward(x, torch.ones(n_samples, 1).to(x.device))
        _, y0_pred= self.forward(x, torch.zeros(n_samples, 1).to(x.device))
        return loss, y0_pred, y1_pred
    
    def weights_init(self):
        for m in self.modules():
            if isinstance(m, Dynamic_FC):
                torch.nn.init.normal_(m.weight, 0, 1.)
                if m.isbias:
                    m.bias.data.zero_()
            elif isinstance(m, Density_Block):
                torch.nn.init.normal_(m.weight, 0, 0.01)
                if m.isbias:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
                    

################################################################################   
################### TransTEE ###################################################
################################################################################
                    
class TransTEE(nn.Module):
    def __init__(self, params, cov_dim=100, att_layers=1, dropout=0.0, init_range_f=0.1, init_range_t=0.1, num_heads=2):
        super(TransTEE, self).__init__()
        """
        cfg_density: cfg for the density estimator; [(ind1, outd1, isbias1), 'act', ....]; the cfg for density estimator head is not included
        num_grid: how many grid used for the density estimator head
        """

        # cfg/cfg_density = [(ind1, outd1, isbias1, activation),....]
        self.input_dim = params['input_dim']
        shared_hidden = params['shared_hidden']
        outcome_hidden = params['outcome_hidden']
        num_heads = num_heads
        self.criterion = nn.MSELoss()
        self.linear1 = nn.Linear(self.input_dim, 100)

        self.feature_weight = Embeddings(shared_hidden, initrange=init_range_f)
        self.treat_emb = Embeddings(shared_hidden, act='id', initrange=init_range_t)
        self.linear2 = MLP(
            dim_input=shared_hidden,
            dim_hidden=shared_hidden,
            dim_output=shared_hidden,
        )

        encoder_layers = TransformerEncoderLayer(shared_hidden, nhead=num_heads, dim_feedforward=shared_hidden, 
                                                 dropout=dropout, num_cov=cov_dim)
        self.encoder = TransformerEncoder(encoder_layers, att_layers)

        decoder_layers = TransformerDecoderLayer(shared_hidden, nhead=num_heads, dim_feedforward=shared_hidden, 
                                                 dropout=dropout,num_t=1)
        self.decoder = TransformerDecoder(decoder_layers, att_layers)

        self.Q = MLP(
            dim_input=shared_hidden,
            dim_hidden=outcome_hidden,
            dim_output=1,
            is_output_activation=False,
        )

    def forward(self, x, t):
        hidden = self.feature_weight(self.linear1(x))
        memory = self.encoder(hidden)
        tgt = self.treat_emb(t)
        tgt = self.linear2(tgt)
        if len(tgt.shape) < 3:
            tgt = tgt.unsqueeze(1)
        out = self.decoder(tgt.permute(1, 0, 2), memory.permute(1, 0, 2))
        if out.shape[0] != 1:
            out = torch.mean(out, dim=1)
        Q = self.Q(out.squeeze(0))
        return torch.mean(hidden, dim=1).squeeze(), Q
    
    def predict(self, x, t, y):
        n_samples = len(x)
        out = self.forward(x, t)
        loss = self.criterion(out[1], y) #.squeeze()
        _, y1_pred= self.forward(x, torch.ones(n_samples, 1).to(x.device))
        _, y0_pred= self.forward(x, torch.zeros(n_samples, 1).to(x.device))
        return loss, y0_pred, y1_pred
                    
    def weights_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
                    
                    
                    
                    