# -*- coding: utf-8 -*-
"""Transformer encoder for topic classification, built from scratch.
No nn.Transformer / nn.MultiheadAttention / nn.TransformerEncoder are used."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    """Attention(Q,K,V) = softmax(Q K^T / sqrt(d_k)) V, with optional key-padding mask.
    Returns (output, attention_weights)."""

    def __init__(self, dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        # q,k,v: (B, T, d) ; mask: (B, T) bool, True = real token
        d_k = q.size(-1)
        scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)              # (B, T, T)
        if mask is not None:
            scores = scores.masked_fill(~mask[:, None, :], float("-inf"))
        attn = F.softmax(scores, dim=-1)
        return self.dropout(attn) @ v, attn


class MultiHeadSelfAttention(nn.Module):
    """h heads, each with its OWN W_q, W_k, W_v (d_model -> d_k), shared output projection W_o."""

    def __init__(self, d_model=128, n_heads=4, d_k=32, d_v=32, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.W_q = nn.ModuleList([nn.Linear(d_model, d_k) for _ in range(n_heads)])
        self.W_k = nn.ModuleList([nn.Linear(d_model, d_k) for _ in range(n_heads)])
        self.W_v = nn.ModuleList([nn.Linear(d_model, d_v) for _ in range(n_heads)])
        self.attention = ScaledDotProductAttention(dropout)
        self.W_o = nn.Linear(n_heads * d_v, d_model)

    def forward(self, x, mask=None):
        outs, weights = [], []
        for h in range(self.n_heads):
            o, a = self.attention(self.W_q[h](x), self.W_k[h](x), self.W_v[h](x), mask)
            outs.append(o)
            weights.append(a)
        return self.W_o(torch.cat(outs, dim=-1)), torch.stack(weights, dim=1)   # (B,h,T,T)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model=128, d_ff=512, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(F.relu(self.fc1(x))))


class SinusoidalPositionalEncoding(nn.Module):
    """PE(pos,2i)=sin(pos/10000^(2i/d)), PE(pos,2i+1)=cos(...). Fixed buffer, not learned."""

    def __init__(self, d_model=128, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div = torch.pow(10000.0, torch.arange(0, d_model, 2, dtype=torch.float) / d_model)
        pe[:, 0::2] = torch.sin(pos / div)
        pe[:, 1::2] = torch.cos(pos / div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=True)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class EncoderBlock(nn.Module):
    """Pre-LN block:  x <- x + Drop(MHA(LN(x)));  x <- x + Drop(FFN(LN(x)))"""

    def __init__(self, d_model=128, n_heads=4, d_k=32, d_ff=512, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.mha = MultiHeadSelfAttention(d_model, n_heads, d_k, d_k, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        a, w = self.mha(self.ln1(x), mask)
        x = x + self.drop1(a)
        x = x + self.drop2(self.ffn(self.ln2(x)))
        return x, w


class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, n_classes=5, d_model=128, n_heads=4, d_k=32, d_ff=512,
                 n_layers=4, dropout=0.1, max_len=512, pad_idx=0):
        super().__init__()
        self.pad_idx = pad_idx
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        nn.init.normal_(self.emb.weight, 0.0, d_model ** -0.5)   # x*sqrt(d) then has unit variance
        with torch.no_grad():
            self.emb.weight[pad_idx].zero_()
        self.cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)   # learned [CLS]
        self.pos = SinusoidalPositionalEncoding(d_model, max_len)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([EncoderBlock(d_model, n_heads, d_k, d_ff, dropout) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, n_classes))
        self.d_model = d_model

    def forward(self, ids, return_attn=False):
        mask = ids != self.pad_idx
        x = self.emb(ids) * math.sqrt(self.d_model)
        x = torch.cat([self.cls.expand(ids.size(0), -1, -1), x], dim=1)   # prepend [CLS]
        mask = torch.cat([torch.ones(ids.size(0), 1, dtype=torch.bool, device=ids.device), mask], dim=1)
        x = self.drop(self.pos(x))
        attns = []
        for blk in self.blocks:
            x, w = blk(x, mask)
            attns.append(w)
        logits = self.head(self.ln_f(x[:, 0]))
        return (logits, attns) if return_attn else logits
