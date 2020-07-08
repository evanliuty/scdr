#   -*- coding: utf-8 -*-
#
#   eval.py
#   
#   Developed by Tianyi Liu on 2020-05-26 as tianyi
#   Copyright (c) 2020. All Rights Reserved.

"""

"""


import torch
import torch.nn as nn
from sklearn.metrics import adjusted_rand_score, silhouette_score


class SAELoss(nn.Module):
    def __init__(self, wse_factor=8):
        super(SAELoss, self).__init__()
        self.wse_factor = wse_factor

    def forward(self, x, y):
        loss_wse = self._loss_weighted_se(x, y)
        loss_total = loss_wse * 1
        return loss_total, loss_wse

    def _loss_weighted_se(self, x, y):
        diff = y - x
        weighted_diff = torch.where(diff > 0, diff ** 2, self.wse_factor * diff ** 2)
        loss_wse = torch.sum(weighted_diff) / (y.size(0) * y.size(1))
        return loss_wse

    @staticmethod
    def _loss_kl(x, mu, logvar):
        return - 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)


def cal_ari(pred, truth):
    var_list = map(cast_tensor, [pred, truth])
    return adjusted_rand_score(*var_list)


def cal_silhouette(data, pred):
    var_list = map(cast_tensor, [data, pred])
    return silhouette_score(*var_list)


def cast_tensor(x):
    return x.detach().cpu().numpy() if type(x) == torch.Tensor else x

