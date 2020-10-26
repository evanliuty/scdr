#   -*- coding: utf-8 -*-
#
#   model.py
#   
#   Developed by Tianyi Liu on 2020-05-26 as tianyi
#   Copyright (c) 2020. All Rights Reserved.

"""

"""

import tqdm
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


from cfg import *
from eval import SAELoss


SEED = TORCH_RAND_SEED


class SAE(nn.Module):
    def __init__(self, dim_list, device, activation="tanh"):
        super(SAE, self).__init__()
        self.dim_list = dim_list
        self.sub_aes = []
        self.device = device
        self.enc, self.dec = nn.Sequential(), nn.Sequential()
        self.activation = AE.act_fun(activation)
        for idx in range(len(dim_list) - 1):
            self.sub_aes.append(AE([dim_list[idx], dim_list[idx + 1]], activation=activation).to(device))

    def forward(self, x):
        embedding = self.enc(x)
        y = self.dec(embedding)
        return embedding, y

    def train_sub_ae(self, x_loader, init_lr, epoch, batch_size=128):
        loss_sub_ae = []
        for i, ae in enumerate(self.sub_aes):
            print(">>> Training {}/{} sub-auto-encoder".format(i + 1, len(self.sub_aes)))
            optimizer = torch.optim.Adam(ae.parameters(), lr=init_lr)
            criterion = SAELoss()
            loss, embedding = SAE.fit(ae, x_loader, optimizer, criterion, epoch)
            loss_sub_ae.append(loss)
            x_loader = DataLoader(TensorDataset(embedding), batch_size=batch_size, shuffle=False)
        return loss_sub_ae

    def stack(self):
        print(">>> Stacking sub-auto-encoders")
        for i in range(len(self.sub_aes)):
            for name, layer in self.sub_aes[i].named_modules():
                if name.split('.')[0] == "enc":
                    if isinstance(layer, nn.Linear):
                        self.enc.add_module("{}".format(i + 1), layer)
                        self.enc.add_module("{}_act".format(i + 1), self.activation())
                    elif isinstance(layer, nn.BatchNorm1d) and i == 0:
                        self.enc.add_module("{}_bn".format(i + 1), layer)
                    elif isinstance(layer, nn.Dropout):
                        self.enc.add_module("{}_d".format(i + 1), layer)

            for name, layer in self.sub_aes[len(self.sub_aes) - 1 - i].named_modules():
                if name.split('.')[0] == "dec" and isinstance(layer, nn.Linear):
                    self.dec.add_module("{}".format(i + 1), layer)
                    self.dec.add_module("{}_act".format(i + 1), self.activation())

    @staticmethod
    def get_embedding(model, x_loader, batch_size=128):
        model.eval()
        if not isinstance(x_loader, DataLoader):
            x_loader = DataLoader(x_loader, batch_size=batch_size, shuffle=False)
        results = torch.Tensor().to(model.device)
        with torch.no_grad():
            for step, data_batch in enumerate(x_loader):
                try:
                    (data, label) = data_batch
                except ValueError:
                    (data) = data_batch[0]
                embedding, _ = model(data)
                results = torch.cat((results, embedding))
        return results

    @staticmethod
    def fit(model, x_loader, optimizer, criterion, train_epoch):
        model.train()
        losses = []
        for epoch in range(train_epoch):
            tic = time.time()
            # LR decay
            if epoch % LR_DECAY_EPOCH == 0 and epoch != 0:
                _ = learning_rate_decay(optimizer)

            print("Epoch: {}/{}\t\t\tLR: {}".format(epoch + 1, train_epoch, optimizer.param_groups[0]['lr']))
            loss_epoch = 0
            for step, data_batch in enumerate(tqdm.tqdm(x_loader)):
                try:
                    (data, label) = data_batch
                except ValueError:
                    (data) = data_batch[0]
                embedding, y = model(data)
                loss, loss_wse = criterion(data, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_epoch += loss * len(data)

            loss_epoch /= len(x_loader.dataset)
            losses.append(loss_epoch)
            toc = time.time()
            print(">>> Averaged epoch loss: {:.4f}\t\tTime {:.4f} s\n".format(loss_epoch, toc - tic))
        return losses, SAE.get_embedding(model, x_loader)

    def __str__(self):
        print("SAE Model Summary")
        for (name, layer) in list(self.named_modules()):
            if isinstance(layer,
                          (nn.Linear, nn.Dropout, nn.Tanh, nn.Sigmoid, nn.ReLU, nn.ELU, nn.Identity, nn.BatchNorm1d)):
                print("Layer: {}\t\t{}".format(name, layer))
        return ""


class AE(nn.Module):
    def __init__(self, dim_list, device="cuda", activation="Tanh"):
        super(AE, self).__init__()
        self.dim_list = dim_list
        self.activation = activation
        self.device = device

        def _bla(fc_dim1, fc_dim2, act=self.activation):
            return _b(fc_dim1) + _la(fc_dim1, fc_dim2, act)

        def _bdla(fc_dim1, fc_dim2, drop_prob=DROPOUT_PROB, act=self.activation):
            return _b(fc_dim1) + _d(drop_prob) + _la(fc_dim1, fc_dim2, act)

        def _dla(fc_dim1, fc_dim2, drop_prob=DROPOUT_PROB, act=self.activation):
            return _d(drop_prob) + _la(fc_dim1, fc_dim2, act)
        
        def _la(fc_dim1, fc_dim2, act=self.activation):
            return [nn.Linear(fc_dim1, fc_dim2),
                    AE.act_fun(act)()]
        
        def _b(dim):
            return [nn.BatchNorm1d(dim)]

        def _d(drop_prob=DROPOUT_PROB):
            return [nn.Dropout(drop_prob)]

        def _l(fc_dim1, fc_dim2):
            return [nn.Linear(fc_dim1, fc_dim2)]

        enc_layer, dec_layer = [], []
        if len(dim_list) > 2:
            for dim in range(len(dim_list) - 2):
                if dim == 0:
                    enc_layer += _bdla(dim_list[dim], dim_list[dim + 1])
                    dec_layer += _bdla(dim_list[len(dim_list) - dim - 1], dim_list[len(dim_list) - dim - 2])
                else:
                    enc_layer += _dla(dim_list[dim], dim_list[dim + 1])
                    dec_layer += _dla(dim_list[len(dim_list) - dim - 1], dim_list[len(dim_list) - dim - 2])
            enc_layer += _l(dim_list[-2], dim_list[-1])
            dec_layer += _l(dim_list[1], dim_list[0])
        else:
            enc_layer += _bla(dim_list[0], dim_list[1])
            dec_layer += _la(dim_list[1], dim_list[0])

        self.enc = nn.Sequential(*enc_layer)
        self.dec = nn.Sequential(*dec_layer)


    def forward(self, x):
        embedding = self.enc(x)
        y = self.dec(embedding)
        return embedding, y


    def __str__(self):
        print("Model Summary")
        for (name, layer) in list(self.named_modules()):
            if isinstance(layer,
                          (nn.Linear, nn.Dropout, nn.Tanh, nn.Sigmoid, nn.ReLU, nn.ELU, nn.Identity, nn.LeakyReLU,  nn.BatchNorm1d)):
                print("Layer: {}\t\t{}".format(name, layer))
        return ""


    def fit(model, x_loader, optimizer, criterion, train_epoch):
        return SAE.fit(model, x_loader, optimizer, criterion, train_epoch)


    def get_embedding(model, x_loader, batch_size=128):
        return SAE.get_embedding(model, x_loader, batch_size)


    @staticmethod
    def act_fun(activation):
        if activation.lower() == "tanh":
            return nn.Tanh
        elif activation.lower() == "sigmoid":
            return nn.Sigmoid
        elif activation.lower() == "relu":
            return nn.ReLU
        elif activation.lower() == "elu":
            return nn.ELU
        elif activation.lower() == "identity":
            return nn.Identity
        elif activation.lower() == "leakyrelu":
            return nn.LeakyReLU
        else:
            raise Exception("!!! Invalid activation provided.")


def learning_rate_decay(optimizer):
    if optimizer.param_groups[0]['lr'] <= LR_DECAY_MIN:
        return optimizer.param_groups[0]['lr']

    print("\n>>> Learning rate decays from {} to {}".format(optimizer.param_groups[0]['lr'],
                                                          optimizer.param_groups[0]['lr'] * LR_DECAY_GAMMA))
    for param_group in optimizer.param_groups:
        param_group['lr'] *= LR_DECAY_GAMMA

    return optimizer.param_groups[0]['lr']

