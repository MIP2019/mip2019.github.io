# -*- coding: utf-8 -*-
"""
Created on Wed 5 1 22:37:51 2024

@author: FeiMa
Email:mafei0603@163.com
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from gensim.models import Word2Vec
import jieba
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn import metrics
import torch.nn.functional as F  
.
.
.

# 定义TextCNN模型
class MWNet(nn.Module):
    def __init__(self, embedding_dim, num_classes):
        super(MWNet, self).__init__()
#        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.conv1 = nn.Conv1d(max_length, 128*2, kernel_size=2)
        self.conv2 = nn.Conv1d(max_length, 128*2, kernel_size=3)
        self.conv3 = nn.Conv1d(max_length, 128*2, kernel_size=4)
        self.fc = nn.Linear(128 * 6, 128 * 6)
        self.fc2 = nn.Linear(128 * 6, num_classes)
        self.dropout = nn.Dropout(0.5)
        self.sigm=nn.Sigmoid()

    def forward(self, x):
        # print('  x=',x.shape) 
        x1=self.conv1(x)
        conv1 = torch.relu(x1)
        # print('  x=',x.shape,'  x1=',x1.shape) 
        conv1=conv1+x1*self.sigm(x1)
        # print(conv1.shape)
        pool1 = torch.max_pool1d(conv1, conv1.size(2)).squeeze(2)
        x2=self.conv2(x)
        conv2 = torch.relu(x2)
        conv2=conv2+x2*self.sigm(x2)
        pool2 = torch.max_pool1d(conv2, conv2.size(2)).squeeze(2)
        x3=self.conv3(x)
        conv3 = torch.relu(x3)
        conv3=conv3+x3*self.sigm(x3)
        pool3 = torch.max_pool1d(conv3, conv3.size(2)).squeeze(2)
        # print(pool3.shape)
        x = torch.cat((pool1, pool2, pool3), 1)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.fc2(x)
        x = F.softmax(x, dim=1)  
        return x
    
 .
.
.
