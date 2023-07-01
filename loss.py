import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def SubgroupTEE_loss(y, t, t_pred, y0_pred, y1_pred, y0_pred_init, y1_pred_init, alpha=1.0, beta=1.0, gamma=1.0):
    loss_t = F.binary_cross_entropy(t_pred, t)

    loss_y_init = torch.sum((1. - t) * torch.square(y - y0_pred_init)) + torch.sum(t * torch.square(y - y1_pred_init))
    loss_y = torch.sum((1. - t) * torch.square(y - y0_pred)) + torch.sum(t * torch.square(y - y1_pred))

    loss = alpha*loss_t + beta*loss_y_init + gamma*loss_y
    return loss


def target_distribution(batch: torch.Tensor) -> torch.Tensor:
    weight = (batch ** 2) / torch.sum(batch, 0)
    return (weight.t() / torch.sum(weight, 1)).t()
    
    
def dragonnet_init_loss(y_true, t_true, t_pred, y0_pred, y1_pred, d_true, d_pred, alpha=1.0, gamma=1.0):
    #t_pred = (t_pred + 0.01) / 1.02

    loss_t = torch.sum(F.binary_cross_entropy(t_pred, t_true))

    loss0 = torch.sum((1. - t_true) * torch.square(y_true - y0_pred))
    loss1 = torch.sum(t_true * torch.square(y_true - y1_pred))
    loss_y = loss0 + loss1

    loss = loss_y + alpha * loss_t
    if d_true is not None:
        loss_d = torch.sum(F.cross_entropy(d_pred, d_true))
        loss += gamma * loss_d
    return loss


def dragonnet_loss(y_true, t_true, t_pred, y0_pred, y1_pred, eps, d_true=None, d_pred=None, alpha=1.0, beta=1.0):
    vanilla_loss = dragonnet_init_loss(y_true, t_true, t_pred, y0_pred, y1_pred, d_true, d_pred, alpha)
    t_pred = (t_pred + 0.01) / 1.02

    y_pred = t_true * y1_pred + (1 - t_true) * y0_pred

    h = (t_true / t_pred) - ((1 - t_true) / (1 - t_pred))

    y_pert = y_pred + eps * h
    targeted_regularization = torch.sum((y_true - y_pert)**2)

    # final
    loss = vanilla_loss + beta * targeted_regularization
    return loss

def vcnet_loss(out, y, alpha=0.5, epsilon=1e-6):
    return ((out[1].squeeze() - y.squeeze())**2).mean() - alpha * torch.log(out[0] + epsilon).mean()

