'''

eye_model - Gated residual network for emotion classification
Copyright (C) 2025 - Chinry Zhu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )
        self.transform = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GLU(),
            nn.LayerNorm(dim)
        )

    def forward(self, x):
        gate = self.gate(x)
        transformed = self.transform(x)
        return x * gate + transformed * (1 - gate)


class AttentionPooling(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Linear(dim, 1)

    def forward(self, x):
        # x: [batch_size, seq_len, dim]
        weights = F.softmax(self.attn(x), dim=1)
        return torch.sum(x * weights, dim=1)


class DeepFFNN(nn.Module):
    def __init__(self, input_dim=31):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.GELU()
        )
        self.res_blocks = nn.Sequential(*[ResidualBlock(128) for _ in range(3)])
        self.attn_pool = AttentionPooling(128)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.embed(x)
        x = self.res_blocks(x)
        x = self.attn_pool(x.unsqueeze(1)).squeeze(1)  # (batch, 128)
        return self.dropout(x)
