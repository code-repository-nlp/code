#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch, config
import torch.nn as nn
from pytorch_pretrained_bert import BertModel
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
conf = config.config()

class BERTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-chinese')
        #self.fc = nn.Linear(conf.emb_dim, vocab_size)
        self.device = device

    def forward(self, x):
        x = x.to(device)
        encoded_layers, _ = self.bert(x)
        enc = encoded_layers[-1]
        #logits = self.fc(enc)
        #y_hat = logits.argmax(-1)
        return enc

class BERTModel_Zh(nn.Module):
    """
    bert-base-chinese
﻿    hfl/chinese-bert-wwm
    hfl/chinese-bert-wwm-ext
    hfl/chinese-roberta-wwm-ext
    """
    def __init__(self):
        super().__init__()
        from transformers import BertModel
        self.bert = BertModel.from_pretrained('bert-base-chinese')
        #self.fc = nn.Linear(conf.emb_dim, vocab_size)
        self.device = device

    def forward(self, x):
        x = x.to(device)
        output = self.bert(x)
        enc = output.last_hidden_state
        pooler_out = output.pooler_output

        #logits = self.fc(enc)
        #y_hat = logits.argmax(-1)
        return enc, pooler_out
   
