import numpy as np
import random
import torch
import torch.nn as nn
from torch.nn import Parameter
from typing import Optional
from torch.autograd import Variable
import torch.nn.functional as F

SEED = 1111
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class ClusterAssignment(nn.Module):
    def __init__(
        self,
        n_clusters,
        w,
        initialization = 'kmeans++'
    ):
        super(ClusterAssignment, self).__init__()
        self.n_clusters = n_clusters
        self.w = w
        self.initialization = initialization
        self.cluster_centers = None
        self.eps = 1e-5
    def _random_init(self, X):
        indices = torch.randperm(X.shape[0])[:self.n_clusters]
        for idx in range(self.n_clusters):
            self.cluster_centers.data[idx] = X[indices[idx]]

    def _kmeans_plusplus_init(self, X):
        #self.cluster_centers.data[0] = X[torch.randint(0, X.shape[0], (1,))]
        self.cluster_centers.data[0] = X[0]
        for idx in range(1, self.n_clusters):
            distances = torch.cdist(X, self.cluster_centers.data[:idx]).min(dim=1).values 
            probabilities = distances / torch.sum(distances)
            centroid_index = torch.multinomial(probabilities, 1)
            self.cluster_centers.data[idx] = X[centroid_index]
            
    def centroid_init(self, X):
        self.cluster_centers = Parameter(torch.zeros(
                   self.n_clusters, X.shape[1])).to(X.device)
        if self.initialization == 'kmeans++':
            self._kmeans_plusplus_init(X)  
        else:
            self._random_init(X) 
            
    def kernel_density_estimate(self, input, centroid, bandwidth=0.5):
        kernel = torch.exp(-(input - centroid).pow(2).sum(dim=1) / (2 * bandwidth ** 2))
        kernel /= kernel.sum()
        return (kernel.unsqueeze(1) * (input - centroid)).sum(dim=0)

    def _adjust_centers(self, input): 
        if self.cluster_centers is None:
            self.centroid_init(input)
        kde = torch.stack(
            [self.kernel_density_estimate(input, centroid) for centroid in self.cluster_centers.data])
        self.cluster_centers.data += kde 
        
    def _update_centers(self, input, cluster_out):
        for idx in range(self.n_clusters):
            filtered = input[cluster_out==idx]
            if len(filtered) > 0:
                self.cluster_centers.data[idx] = (1-self.w) * self.cluster_centers.data[idx] + \
                                                                    self.w * torch.mean(filtered, 0)  
            
    def forward(self, input):
        if self.cluster_centers is None:
            return torch.zeros(len(input), self.n_clusters).to(input.device)
        clusters = torch.cdist(input, self.cluster_centers)
        clusters = nn.functional.softmin(clusters, dim=1)
        return clusters

    

    
   