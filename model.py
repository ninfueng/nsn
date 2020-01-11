#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class NetworkWithSub1(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = nn.Parameter(torch.empty(784, 10, requires_grad=True))
        self.b1 = nn.Parameter(torch.empty(10, requires_grad=True))
        nn.init.xavier_uniform_(self.w1)
        nn.init.constant_(self.b1, 1e-2)

    def forward(self, x):
        x = x.view(-1, self.num_flat_features(x))
        z1 = x.mm(self.w1) + self.b1
        y = F.log_softmax(z1, dim=-1)
        return y

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class NetworkWithSub2(nn.Module):
    def __init__(self, neuron):
        super().__init__()
        self.w1 = nn.Parameter(torch.empty(784, neuron, requires_grad=True))
        self.w2 = nn.Parameter(torch.empty(neuron, 10, requires_grad=True))
        self.b1 = nn.Parameter(torch.empty(neuron, requires_grad=True))
        self.b2 = nn.Parameter(torch.empty(10, requires_grad=True))
        self.drop1 = nn.Dropout(p=0.2)

        nn.init.xavier_uniform_(self.w1)
        nn.init.xavier_uniform_(self.w2)
        nn.init.constant_(self.b1, 1e-2)
        nn.init.constant_(self.b2, 1e-2)

    def forward(self, x):
        x = x.view(-1, self.num_flat_features(x))
        z1 = x.mm(self.w1) + self.b1
        a1 = F.relu(z1)
        a1 = self.drop1(a1)
        z2 = a1.mm(self.w2) + self.b2
        y = F.log_softmax(z2, dim=-1)
        return y

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class NetworkWithSub3(nn.Module):
    def __init__(self, neuron):
        super().__init__()
        self.w1 = nn.Parameter(torch.empty(784, neuron, requires_grad=True))
        self.w2 = nn.Parameter(torch.empty(neuron, neuron, requires_grad=True))
        self.w3 = nn.Parameter(torch.empty(neuron, 10, requires_grad=True))

        self.b1 = nn.Parameter(torch.empty(neuron, requires_grad=True))
        self.b2 = nn.Parameter(torch.empty(neuron, requires_grad=True))
        self.b3 = nn.Parameter(torch.empty(10, requires_grad=True))

        self.drop1 = nn.Dropout(p=0.2)
        self.drop2 = nn.Dropout(p=0.5)

        nn.init.xavier_uniform_(self.w1)
        nn.init.xavier_uniform_(self.w2)
        nn.init.xavier_uniform_(self.w3)
        nn.init.constant_(self.b1, 1e-2)
        nn.init.constant_(self.b2, 1e-2)
        nn.init.constant_(self.b3, 1e-2)

    def forward(self, x):
        x = x.view(-1, self.num_flat_features(x))
        z1 = x.mm(self.w1) + self.b1
        a1 = F.relu(z1)
        a1 = self.drop1(a1)
        z2 = a1.mm(self.w2) + self.b2
        a2 = F.relu(z2)
        a2 = self.drop2(a2)
        z3 = a2.mm(self.w3) + self.b3
        y = F.log_softmax(z3, dim=-1)
        return y

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
