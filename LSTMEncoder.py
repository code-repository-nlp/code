#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LSTMEncoder(nn.Module):
    def __init__(self, emb_dim, pos_dim, dep_dim, hidden_dim, num_layers,\
                 pos_size, dep_size, batch_first = True, bidirectional = True):
        super().__init__()
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1
        self.embed_pos = nn.Embedding(pos_size, pos_dim)
        self.embed_dep = nn.Embedding(dep_size, dep_dim)
        self.lstm = nn.LSTM(input_size = emb_dim + pos_dim + dep_dim, 
                           hidden_size = hidden_dim, 
                           num_layers = num_layers, 
                           batch_first = batch_first, 
                           bidirectional = bidirectional)
        self.params_init(self.lstm.named_parameters())
        self.device = device
   
    def params_init(self, params):
        for name, param in params:
            if len(param.data.shape)==2:
                nn.init.kaiming_normal_(param, a=1, mode='fan_in')
            if len(param.data.shape)==1:
                nn.init.normal_(param)
   
    def initHidden(self, batch_size):
        h = torch.zeros(self.num_layers*self.num_directions, batch_size, self.hidden_dim, device=device)
        c = torch.zeros(self.num_layers*self.num_directions, batch_size, self.hidden_dim, device=device)
        return (h, c)
    
    def forward(self, input_x_embs, x_pos_id, x_dep_id, mask, hidden):
        input_x_embs.to(device)
        x_pos_id.to(device)
        x_dep_id.to(device)
        x_pos_embs = self.embed_pos(x_pos_id)
        x_dep_embs = self.embed_dep(x_dep_id)
        #sort the original seqs
        input_x_embs = torch.cat([input_x_embs, x_pos_embs, x_dep_embs], -1)
        input_l = mask.sum(1)
        idx_sort = np.argsort(input_l)[::-1].tolist()
        idx_unsort = np.argsort(idx_sort).tolist()
        #packed the sorted seqs and avoid the impact of PAD on RNN model
        packed = torch.nn.utils.rnn.pack_padded_sequence(input_x_embs[idx_sort], \
                                                         list(input_l[idx_sort]), batch_first=True)
        _, (hn, cn) = self.lstm(packed, hidden)
        #hn: [num_layers * num_directions, batch, hidden_size]
        #hx, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        #hx = hx[idx_unsort]
        hn = hn[:,idx_unsort,:]
        hn = torch.cat([hn[0], hn[1]], dim=-1)
        #cn = cn[:,idx_unsort,:]
        return hn#, (hn, cn) #torch.mean(hx, dim=1)
