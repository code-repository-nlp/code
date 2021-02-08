#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from scipy.stats import pearsonr
from scipy.stats import kendalltau
from scipy.stats import spearmanr
import collections as col
import torch, nltk, pickle, json, utils, msr_data_error_analysis, jieba, stanza
import pandas as pd
import numpy as np
import torch.nn as nn
from torch.utils import data
import torch.optim as optim
from pytorch_pretrained_bert import BertTokenizer
import torch, BERTModel, spacy, config, LSTMEncoder, LinearLayer

nlp = spacy.load("en_core_web_md")
conf = config.config()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USE_CUDA = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor

pretrained = 'bert-base-chinese'
"""
bert-base-chinese
hfl/chinese-bert-wwm
hfl/chinese-bert-wwm-ext
hfl/chinese-roberta-wwm-ext
hfl/chinese-roberta-wwm-ext-large
"""
tokenizer = BertTokenizer.from_pretrained(pretrained, do_lower_case=True)
#%%
def read_multi(name):
    data=[]
    temp=[]
    with open(name, 'r') as f:
        for line in f.readlines():
            if line == '\n':
                data.append(temp)
                temp=[]
            else:
                temp.append(line.split())
    return data

def write(name, data):
    with open(name, 'w') as f:
        for s in data:
            f.writelines(s)    
            f.writelines('\n')
#%%
class PosDataset(data.Dataset):
    def __init__(self, data_inputs):
        sents, upos_tag_list, dep_rel_list, y_score_list = [], [], [], [] # list of lists
        for sent in data_inputs:       
            words     = [e[0] for e in sent[0]]
            upos_tags = [e[1] for e in sent[0]]
            dep_rels  = [e[2] for e in sent[0]]
       
            sents.append(["[CLS]"] + words + ["[SEP]"])
            upos_tag_list.append(["<pad>"] + upos_tags + ["<pad>"])
            dep_rel_list.append(["<pad>"] + dep_rels + ["<pad>"])
            y_score_list.append(sent[-1])
            
        self.sents = sents 
        self.upos_tag_list = upos_tag_list
        self.dep_rel_list  = dep_rel_list
        self.y_score_list  = y_score_list
           
    def __len__(self):
        return len(self.sents)

    def __getitem__(self, idx):
        words, upos_tags, dep_rels, y_labels = self.sents[idx], self.upos_tag_list[idx], \
            self.dep_rel_list[idx], self.y_score_list[idx]# words, tags: string list
            
        x_id, x_tokenized, x_upos, x_dep = [], [], [], []# list of ids
        is_heads = [] # list. 1: the token is the first piece of a word
        for w, upos, dep in zip(words, upos_tags, dep_rels):
            tokens = tokenizer.tokenize(w) if w not in ("[CLS]", "[SEP]") else [w]
            x_tokenized.extend(tokens)
            xx = tokenizer.convert_tokens_to_ids(tokens)
            is_head = [1] + [0]*(len(tokens) - 1)
            x_id.extend(xx)
            is_heads.extend(is_head)
            
            x_upos.append(upos2idx[upos])
            x_dep.append(dep2idx[dep])
        try:
            assert len(x_id)==len(is_heads), "len(x)={},\
                len(is_heads)={}".format(len(x_id), len(is_heads))
        except:
            print("###########")
            print(x_id)
            print(x_upos)
            print(is_heads)
            print("###########")

        # seqlen
        seqlen = len(x_tokenized)

        # to string
        x_tokenized = " ".join(x_tokenized)
        words = " ".join(words)
        
        return x_tokenized, words, x_id, is_heads, upos_tags, dep_rels, x_upos, x_dep, y_labels, seqlen


def pad(batch):
    '''Used as the function of collate_fn in DataLoader: Pads to the longest sample'''
    f = lambda x: [sample[x] for sample in batch]
    x_tokenized = f(0)
    words = f(1)
    #x_id = f(2)
    is_heads = f(3)
    upos_tags = f(4)
    dep_rels = f(5)
    #x_upos = f(6)
    #x_xpos = f(7)
    labels = f(8)
    seqlens = f(9)
    maxlen = np.array(seqlens).max()
    
    f = lambda x, seqlen: [sample[x] + [0] * (seqlen - len(sample[x])) for sample in batch] # 0: <pad>
    x = f(2, maxlen)
    x_upos = f(6, maxlen)
    x_dep = f(7, maxlen)
    return x_tokenized, words, LongTensor(x), is_heads, upos_tags, dep_rels, LongTensor(x_upos), LongTensor(x_dep), labels, seqlens 

class Tokenize():
    def __init__(self):
        self.zh_nlp = stanza.Pipeline('zh', processors='tokenize,pos', \
                      tokenize_with_jieba=True,  use_gpu=False)    
        jieba.set_dictionary('dict.txt.big.txt')

    def jieba_tokenize(self, s):
        res = []
        doc_s = self.zh_nlp(s)
        for sent in doc_s.sentences:
            for w in sent.words:
                res.append((w.text, w.upos, w.xpos))
  
        return res
T = Tokenize()
#%%

with open('data_1003.json', 'r') as f:
    data_1003 = json.load(f)

comp_1003 = ["".join([w[2] for w in s if w[-1]=="1"]) for s in [e["sentence"] for e in data_1003]]
flue_1003 = [np.round(np.mean([int(w) for w in s]),1) for s in [e["fluency"] for e in data_1003]]

score_class = set([np.round(e,1) for e in flue_1003])
score2idx = {s:idx for idx, s in enumerate(score_class)}
idx2score = {idx:s for idx, s in enumerate(score_class)}



data_tokenized = []
upos = set(["<pad>"])
deps = set(["<pad>"])
for e in data_1003:
    s_tokenized = e["sentence"]
    score = np.round(np.mean([int(v) for v in e["fluency"]]), 1)
    temp=[]
    for w in s_tokenized:
        upos.add(w[3])
        deps.add(w[5])
        if w[-1]=="1":
            temp.append([w[2], w[3], w[5]])
    data_tokenized.append([temp, score])
    

#%%

aug_train_data = utils.random_delete(score2idx, quntity=100)
train_data, test_data = aug_train_data, data_tokenized

#aug_dd = read_multi("aug_dd.txt")
sample_data = [[[tuple(w) for w in e]]+[3.0] for e in aug_dd]
print(len(train_data), len(test_data))
#%%   
for e in train_data:
    for w in e[0]:
        upos.add(w[1])
        deps.add(w[2])

upos2idx = {u:idx for idx, u in enumerate(upos)}
idx2upos = {idx:u for idx, u in enumerate(upos)}
dep2idx = {x:idx for idx, x in enumerate(deps)}
idx2dep = {idx:x for idx, x in enumerate(deps)}

#%%
tokenizer = BertTokenizer.from_pretrained(pretrained, do_lower_case=True)

aug_train_dataset = PosDataset(aug_train_data)
train_dataset = PosDataset(train_data)
eval_dataset = PosDataset(test_data)
sample_dataset = PosDataset(sample_data)


aug_train_iter = data.DataLoader(dataset=aug_train_dataset,
                             batch_size=conf.batch_size,
                             shuffle=True,
                             num_workers=0,
                             collate_fn=pad)

train_iter = data.DataLoader(dataset=train_dataset,
                             batch_size=conf.batch_size,
                             shuffle=True,
                             num_workers=0,
                             collate_fn=pad)


sample_iter = data.DataLoader(dataset=sample_dataset,
                             batch_size=conf.batch_size,
                             shuffle=True,
                             num_workers=0,
                             collate_fn=pad)

eval_iter = data.DataLoader(dataset=eval_dataset,
                             batch_size=32,
                             shuffle=False,
                             num_workers=0,
                             collate_fn=pad)

#%%

model = BERTModel.BERTModel_Zh()
model_lstm = LSTMEncoder.LSTMEncoder(emb_dim = conf.emb_dim, 
                                     pos_dim = conf.pos_dim, 
                                     dep_dim = conf.dep_dim, 
                                     hidden_dim = conf.hidden_dim, 
                                     num_layers = conf.num_layers, 
                                     pos_size = len(upos2idx), 
                                     dep_size = len(dep2idx), 
                                     batch_first = True, 
                                     bidirectional = True)
model_linear =  LinearLayer.LinearLayer(len(score2idx))

model.to(device)
model_lstm.to(device)
model_linear.to(device)

model_optimizer = torch.optim.Adam(model.parameters(), lr = conf.lr)
model_lstm_optimizer = torch.optim.Adam(model_lstm.parameters(), lr = conf.lr)
model_linear_optimizer = torch.optim.Adam(model_linear.parameters(), lr = conf.lr)
criterion = nn.CrossEntropyLoss()

#%%
def evaluate(eval_iter, model, model_lstm, model_linear):
    model.eval()
    model_lstm.eval()
    model_linear.eval()
    with torch.no_grad():
        N=0
        corr=0
        y_hat_list=[]
        y_label_list=[]
        for epoch, batch in enumerate(eval_iter):
            x_tokenized, words, x_id, is_heads, upos_tags, dep_rels, \
                x_upos, x_dep, y_label, seqlens = batch
            enc, pooled_out = model(x_id) # logits: (N, T, VOCAB), y: (N, T)
            #Average the subwords [1, 0, 0, 1] -> [ave(1, 0, 0), 1] -> [1, 1]
            actual_mask = []
            actual_max_l = max([sum(is_head) for is_head in is_heads])
            enc_bert = []
            
            x_upos_id = []
            x_dep_id = []
            
            for index_i, (l, is_head, enc_s, upos, dep) in enumerate(zip(seqlens, is_heads, enc, upos_tags, dep_rels)):
                actual_l = sum(is_head)
                l = len(is_head)
                actual_mask.extend([[1]*actual_l+[0]*(actual_max_l-actual_l)])
                x_upos_id.append([upos2idx[e] for e in [upos + \
                                        ['<pad>']*(actual_max_l-actual_l)][0]])
                    
                x_dep_id.append([dep2idx[e] for e in [dep + \
                                        ['<pad>']*(actual_max_l-actual_l)][0]])
     
                m_merge = np.zeros((actual_l, l))#[14, 20]
                head_index = [index for index, e in enumerate(is_head) if e==1]
                head_index.append(l)
                for i in range(len(m_merge)):
                    for j in range(head_index[i], head_index[i+1]):
                        m_merge[i][j] = 1.0
                    ave = sum(m_merge[i])
                    
                    if float(ave)==0:
                        print(words[index_i])
                        print("m_merge[i]", m_merge[i], actual_l, l)
                        print("is_head", is_head)
                    
                    for j in range(len(m_merge[i])):
                        m_merge[i][j]/=float(ave)
                       
                m_merge = FloatTensor(m_merge)
                try:
                    enc_s = torch.mm(m_merge, enc_s[:l])
                except:
                    print(m_merge.shape, m_merge)
                    print(enc_s[:l].shape, enc_s[:l])
                temp_tensor = torch.zeros(actual_max_l-actual_l, enc_s.shape[-1]).to(device) if USE_CUDA \
                    else torch.zeros(actual_max_l-actual_l, enc_s.shape[-1]).to(device)
                enc_s = torch.cat([enc_s, temp_tensor], 0)
                enc_bert.append(enc_s)
           
            actual_mask = np.array(actual_mask, dtype='float64')
            x_upos_id = LongTensor(np.array(x_upos_id))
            x_dep_id = LongTensor(np.array(x_dep_id))
            enc_bert = torch.stack(enc_bert, dim=0)
           
            h0_lstm = model_lstm.initHidden(actual_mask.shape[0])
            hn = model_lstm(enc_bert, x_upos_id, x_dep_id, actual_mask, h0_lstm)
   
            logits = model_linear(hn)
            y_hat = logits.argmax(-1).data.cpu().numpy()
            for y_h, y_t in zip(y_hat, y_label):
                if y_h == y_t:
                    corr+=1
                N+=1
            y_hat_list.extend(list(y_hat))
            y_label_list.extend(list(y_label))
     
        
        #y_label_list = [idx2score[e] for e in y_label_list]
        y_hat_list   = [idx2score[e] for e in y_hat_list]
        print("pred", len(y_hat_list), y_hat_list[:15])
        print("true", len(y_label_list), y_label_list[:15])
        df = pd.DataFrame({'a':y_label_list, 'b':y_hat_list})
        print('pearson', np.round(df.corr('pearson')['a'][1], 3))
        print('kendall', np.round(df.corr('kendall')['a'][1], 3))
        print('spearman', np.round(df.corr('spearman')['a'][1], 3))
            
        return corr/float(N), np.round(df.corr('pearson')['a'][1], 3), y_hat_list
                        
    
def sample(test_data, y_hat_list):
    test_data = ["".join([w[0] for w in e[0]]) for e in test_data]
    res=[]
    for i in range(0, len(y_hat_list), 5):
        print(i)
        print(test_data[i:i+5])
        k = np.argmax(y_hat_list[i:i+5])
        res.append(test_data[i:i+5][k])
    return res
    
#%%
if __name__=='__main__':
    epoch =40
    pearson_max=0
    acc_max=0
    for epoch_i in range(epoch):
        model.train()
        model_lstm.train()
        model_linear.train()
        for iter_i, batch in enumerate(train_iter):
            x_tokenized, words, x_id, is_heads, upos_tags, dep_rels, \
                x_upos, x_dep, y_label, seqlens = batch

            model_optimizer.zero_grad()
            model_lstm_optimizer.zero_grad()
            model_linear_optimizer.zero_grad()
        
            enc, pooled_out = model(x_id) # logits: (N, T, VOCAB), y: (N, T)
            #Average the subwords [1, 0, 0, 1] -> [ave(1, 0, 0), 1] -> [1, 1]
            actual_mask = []
            actual_max_l = max([sum(is_head) for is_head in is_heads])
            enc_bert = []
        
            x_upos_id = []
            x_dep_id = []
            mark=1
            for index_i, (l, is_head, enc_s, upos, dep) in enumerate(zip(seqlens, is_heads, enc, upos_tags, dep_rels)):
                actual_l = sum(is_head)
                l = len(is_head)
                actual_mask.extend([[1]*actual_l+[0]*(actual_max_l-actual_l)])
                x_upos_id.append([upos2idx[e] for e in [upos + \
                                    ['<pad>']*(actual_max_l-actual_l)][0]])
                
                x_dep_id.append([dep2idx[e] for e in [dep + \
                                    ['<pad>']*(actual_max_l-actual_l)][0]])
 
                m_merge = np.zeros((actual_l, l))#[14, 20]
                head_index = [index for index, e in enumerate(is_head) if e==1]
                head_index.append(l)
                for i in range(len(m_merge)):
                    for j in range(head_index[i], head_index[i+1]):
                        m_merge[i][j] = 1.0
                    ave = sum(m_merge[i])
                
                    if float(ave)==0:
                        print(words[index_i])
                        print("m_merge[i]", m_merge[i], actual_l, l)
                        print("is_head", is_head)
                
                    for j in range(len(m_merge[i])):
                        m_merge[i][j]/=float(ave)
                  
                m_merge = FloatTensor(m_merge)
                try:
                    enc_s = torch.mm(m_merge, enc_s[:l])
                except:
                    mark=0
                    break
                temp_tensor = torch.zeros(actual_max_l-actual_l, enc_s.shape[-1]).to(device) if USE_CUDA \
                    else torch.zeros(actual_max_l-actual_l, enc_s.shape[-1])
                enc_s = torch.cat([enc_s, temp_tensor], 0)
                enc_bert.append(enc_s)
            
            if mark==0:
                continue
            actual_mask = np.array(actual_mask, dtype='float64')
            x_upos_id = LongTensor(np.array(x_upos_id))
            x_dep_id = LongTensor(np.array(x_dep_id))
            enc_bert = torch.stack(enc_bert, dim=0)
            y_label = LongTensor(np.array(y_label))
                     
            h0_lstm = model_lstm.initHidden(actual_mask.shape[0])
            hn = model_lstm(enc_bert, x_upos_id, x_dep_id, actual_mask, h0_lstm)
            logits = model_linear(hn)
     
            loss = criterion(logits, y_label)
            loss.backward()
        
            model_optimizer.step()
            model_lstm_optimizer.step()
            model_linear_optimizer.step()

            loss = loss.item()
            print("epoch: {}, iter: {}, loss: {}".format(epoch_i, iter_i, loss))
        
        acc, pearson, y_hat_list = evaluate(eval_iter, model, model_lstm, model_linear)

