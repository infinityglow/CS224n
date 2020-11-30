#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""
import torch
import torch.nn as nn

### YOUR CODE HERE for part 1i
class CNN(nn.Module):
    def __init__(self, in_channels, out_channels, k_size=5):
        super().__init__()
        self.conv1d = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=k_size)
    def forward(self, x_reshaped):
        x_conv = self.conv1d(x_reshaped)
        x_conv_out = torch.max(torch.relu(x_conv), dim=2)[0]
        return x_conv_out

### END YOUR CODE

