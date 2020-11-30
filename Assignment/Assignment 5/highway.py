#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""
import torch
import torch.nn as nn

### YOUR CODE HERE for part 1h
class Highway(nn.Module):
    def __init__(self, word_embed_size):
        super().__init__()
        self.proj = nn.Linear(in_features=word_embed_size, out_features=word_embed_size)
        self.gate = nn.Linear(in_features=word_embed_size, out_features=word_embed_size)
    def forward(self, x_conv_out):
        x_proj = torch.relu(self.proj(x_conv_out))
        x_gate = torch.sigmoid(self.gate(x_conv_out))
        x_highway = x_gate * x_proj + (torch.ones_like(x_gate) - x_gate) * x_conv_out
        return x_highway

### END YOUR CODE 

