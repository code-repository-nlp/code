#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import spacy, scipy, random
nlp = spacy.load("en_core_web_md")


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


def random_delete(score2idx, quntity=10):
    scores = [1.0, 1.3, 1.7, 2.0, 2.3, 2.7, 3.0]
    N = len(scores)
    cr_min = 0.15
    print("cr_min:", cr_min)
    INTERVAL = (1-cr_min)/float(N-1)
    pre_cr = [cr_min+i*INTERVAL for i in range(N)]
    #pre_cr = [0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.0]
    #idx_class = [0, 618, 1204, 1664, 1953, 2161, 2359, 2500, 2617, 2712, 2777, 2918]
    nb_class = len(pre_cr)  

    data = read_multi("data_s_parsed.txt")
    np.random.shuffle(data)
    data = [[[w[0], w[1], w[2], w[2], w[5], w[3], w[4]] for w in e] for e in data]
    data_sent=[]
    for e in data:
        if e!=[['parsing','wrongly']]:
            data_sent.append([(w[2].lower(), w[5], w[4]) for w in e])

    data_aug =[]
    for i in range(nb_class):
        #data_i = data_sent[idx_class[i]:idx_class[i+1]]
        data_i = data_sent[quntity*i:quntity*(i+1)]
        for s in data_i:
            k = int(pre_cr[i]*len(s))
            indexs = random.sample(range(0, len(s)), k)
            indexs = sorted(indexs)
            temp_aug = []
            for idx in indexs:
                temp_aug.append(s[idx])
            
            data_aug.append([temp_aug, score2idx[scores[i]]])
    return data_aug

def tree_depth(father):
    depth_list = []
    for i in range(1, len(father)+1):
        depth=0
        curr=i
        while (curr!=0):
            depth+=1
            try:
                curr = father[curr]
            except:
                return False, depth_list
            if depth>=200:
                return False, depth_list
        
        depth_list.append(depth)    
    if len(depth_list)==len(father):
        return True, depth_list
    else:
        return False, depth_list
        

def parsing(data):
    res = []
    for i, e in enumerate(data):
        if i%10000==0:
            print("%d done!"%i)
        doc = nlp(" ".join([w[2] for w in e]))
        res.append([(token.idx, token.head.idx, token.text,token.text, token.dep_, token.pos_, token.tag_) for token in doc])
        print("\n")
