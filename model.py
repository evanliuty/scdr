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
        for idx in range(len(dim_list) - 1):
            self.sub_aes.append(AE(dim_list[idx], dim_list[idx + 1], device).to(device))
        self.activation = AE.act_fun(activation)

    def forward(self, x):
        embedding = self.enc(x)
        y = self.dec(embedding)
        return embedding, y

    def train_sub_ae(self, x_loader, init_lr, epoch, batch_size=128):
        loss_sub_ae = []
        for i, ae in enumerate(self.sub_aes):
            print(">>> Training {}/{} sub-auto-encoder".format(i + 1, len(self.sub_aes)))
            optimizer = torch.optim.Adam(ae.parameters(), lr=init_lr)
            criterion = SAELoss(1)
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
                        self.enc.add_module("enc_{}".format(i + 1), layer)
                        self.enc.add_module("enc_{}_act".format(i + 1), self.activation())
                    elif isinstance(layer, nn.BatchNorm1d) and i == 0:
                        self.enc.add_module("enc_{}_bn".format(i + 1), layer)
            for name, layer in self.sub_aes[len(self.sub_aes) - 1 - i].named_modules():
                if name.split('.')[0] == "dec" and isinstance(layer, nn.Linear):
                    self.dec.add_module("dec_{}".format(i + 1), layer)
                    self.dec.add_module("dec_{}_act".format(i + 1), self.activation())

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
    def __init__(self, input_dim, hidden_dim, device, dropout_rate=DROPOUT_PROB, activation="tanh"):
        super(AE, self).__init__()
        self.activation = AE.act_fun(activation)
        self.enc = nn.Sequential(nn.Dropout(dropout_rate), nn.BatchNorm1d(input_dim), nn.Linear(input_dim, hidden_dim), self.activation())
        self.dec = nn.Sequential(nn.Dropout(dropout_rate), nn.Linear(hidden_dim, input_dim), self.activation())
        self.device = device

    def forward(self, x):
        embedding = self.enc(x)
        y = self.dec(embedding)
        return embedding, y

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
        else:
            raise Exception("!!! Invalid activation provided.")


class _AESC(nn.Module):
    dim = [8000, 1024, 512, 256, 64, 16, 4, 2]
    device = "cuda"

    def __init__(self):
        super(_AESC, self).__init__()

        def _fc_bn_act(fc_dim1, fc_dim2):
            return nn.Sequential(nn.Linear(fc_dim1, fc_dim2),
                                 nn.BatchNorm1d(fc_dim2),
                                 nn.LeakyReLU())

        def _fc_bn_d_act(fc_dim1, fc_dim2, drop_prob=DROPOUT_PROB):
            return nn.Sequential(nn.Linear(fc_dim1, fc_dim2),
                                 nn.BatchNorm1d(fc_dim2),
                                 nn.Dropout(drop_prob),
                                 nn.LeakyReLU())

        def _fc_d_act(fc_dim1, fc_dim2, drop_prob=DROPOUT_PROB):
            return nn.Sequential(nn.Linear(fc_dim1, fc_dim2),
                                 nn.Dropout(drop_prob),
                                 nn.LeakyReLU())

        def _fc_act(fc_dim1, fc_dim2):
            return nn.Sequential(nn.Linear(fc_dim1, fc_dim2),
                                 nn.LeakyReLU())

        def _fc(fc_dim1, fc_dim2):
            return nn.Linear(fc_dim1, fc_dim2)

        self.enc_1 = _fc_act(self.dim[0], self.dim[1])
        self.enc_2 = _fc_act(self.dim[1], self.dim[2])
        self.enc_3 = _fc_act(self.dim[2], self.dim[3])
        self.enc_4 = _fc_act(self.dim[3], self.dim[4])
        self.enc_5 = _fc_act(self.dim[4], self.dim[5])
        self.enc_6 = _fc(self.dim[5], self.dim[6])
        # self.enc_7 = _fc(self.dim[6], self.dim[7])

        # self.dec_7 = _fc_act(self.dim[7], self.dim[6])
        self.dec_6 = _fc_act(self.dim[6], self.dim[5])
        self.dec_5 = _fc_act(self.dim[5], self.dim[4])
        self.dec_4 = _fc_act(self.dim[4], self.dim[3])
        self.dec_3 = _fc_act(self.dim[3], self.dim[2])
        self.dec_2 = _fc_act(self.dim[2], self.dim[1])
        self.dec_1 = _fc_act(self.dim[1], self.dim[0])

    def forward(self, x):
        h1 = self.enc_1(x)
        h2 = self.enc_2(h1)
        h3 = self.enc_3(h2)
        h4 = self.enc_4(h3)
        h5 = self.enc_5(h4)
        h6 = self.enc_6(h5)
        # h7 = self.enc_7(h6)

        # y6 = self.dec_7(h7)
        y5 = self.dec_6(h6)
        y4 = self.dec_5(y5)
        y3 = self.dec_4(y4)
        y2 = self.dec_3(y3)
        y1 = self.dec_2(y2)
        y = self.dec_1(y1)

        return y, h6, h1, y1

    @classmethod
    def adjust_model_par(cls, dim, device):
        cls.dim[0] = dim
        cls.device = device


def learning_rate_decay(optimizer):
    if optimizer.param_groups[0]['lr'] <= LR_DECAY_MIN:
        return optimizer.param_groups[0]['lr']

    print("\n>>> Learning rate decays from {} to {}".format(optimizer.param_groups[0]['lr'],
                                                          optimizer.param_groups[0]['lr'] * LR_DECAY_GAMMA))
    for param_group in optimizer.param_groups:
        param_group['lr'] *= LR_DECAY_GAMMA

    return optimizer.param_groups[0]['lr']

