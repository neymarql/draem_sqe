#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
  Filename: sqe
  Author: Long Qian
  Date: 2025-02-16
  Email: neymarql0614@gmail.com
"""

import torch
import torch.nn as nn
import torchvision.models as models

class SynthesisQualityEstimator(nn.Module):
    def __init__(self):
        super(SynthesisQualityEstimator, self).__init__()
        backbone = models.wide_resnet50_2(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        self.fc = nn.Linear(2048, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        feats = self.feature_extractor(x)
        feats = feats.view(feats.size(0), -1)
        score = self.fc(feats)
        quality = self.sigmoid(score)
        return quality
    
model = SynthesisQualityEstimator()
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"总参数量: {total_params}")
print(f"可训练参数量: {trainable_params}")
