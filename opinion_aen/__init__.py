# -*- coding: utf-8 -*-
# file: __init__.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.

import numpy as np
import re
import torch
import torch.nn.functional as F
from .models.aen import CrossEntropyLoss_LSR, AEN_BERT
from pytorch_transformers import BertModel
from .data_utils import Tokenizer4Bert, ABSADataset
from torch.utils.data import DataLoader


class Option(object): 
    def __init__(self):
        self.model_name = 'aen_bert'
        self.model_class = AEN_BERT
        self.max_seq_len = 512
        self.pretrained_bert_name='bert-base-chinese' #'bert-base-uncased'
        self.polarities_dim = 3
        self.dropout = 0.1
        self.bert_dim = 768
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.hidden_dim = 300
        self.batch_size = 30


class model(Option):
    def __init__(self, model_path):
        self.opt = Option()
        self.opt.state_dict_path = model_path

        self.tokenizer = Tokenizer4Bert(self.opt.max_seq_len, self.opt.pretrained_bert_name)
        bert = BertModel.from_pretrained(self.opt.pretrained_bert_name)
        self.model = self.opt.model_class(bert, self.opt).to(self.opt.device)

        print('loading model {0} ...'.format(self.opt.model_name))
        self.model.load_state_dict(torch.load(self.opt.state_dict_path))
        self.model.eval()
        torch.autograd.set_grad_enabled(False)

    def predict(self, inputs):
        model_inputs = ABSADataset(inputs, self.tokenizer)
        data_loader = DataLoader(dataset=model_inputs, batch_size=self.opt.batch_size, shuffle=True)
        batch_probs = []
        batch_polar = []
        for t_batch, t_sample_batched in enumerate(data_loader):
            t_inputs = [t_sample_batched[col].to(self.opt.device) for col in ['text_raw_bert_indices', 'aspect_bert_indices']]
            t_outputs = self.model(t_inputs)
            t_probs = F.softmax(t_outputs, dim=-1).cpu().numpy()
            batch_probs.append(t_probs)
            t_polar = [str(i.argmax(axis=-1)) for i in t_probs]
            batch_polar.append(t_polar)

        return batch_probs, batch_polar


# input
class Input(object): 
    def __init__(self, data):
        self.batch_data = data
        self.data = self.process()
     
    def process(self):
        data = []
        for pair in self.batch_data:
            context = pair['context']
            target = pair['target']
            idxs = context.rfind(target)
            data.append([context[:idxs], target, context[idxs+len(target):]])
        return data
