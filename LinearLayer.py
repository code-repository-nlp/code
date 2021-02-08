#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch, config
import torch.nn as nn
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
conf = config.config()

class LinearLayer(nn.Module):
    def __init__(self, nb_class):
        super().__init__()
        #self.embed = nn.Embedding(pos_size, conf.pos_dim)
        self.fc1 = nn.Linear(conf.hidden_dim*2, nb_class)
        #self.fc2 = nn.Linear(conf.linear_dim, nb_class)
        self.params_init(self.fc1.named_parameters())
        #self.params_init(self.fc2.named_parameters())


    def params_init(self, params):
        for name, param in params:
            if len(param.data.shape)==2:
                nn.init.kaiming_normal_(param, a=1, mode='fan_in')
            if len(param.data.shape)==1:
                nn.init.normal_(param)

    def forward(self, x):
        x = x.to(device)
        logits = self.fc1(x)
        #x = F.relu(x)
        #logits = self.fc2(x)
        #y_hat = logits.argmax(-1)
        return logits
    