# -*- coding: utf-8 -*-
"""BiLSTM sequence labeller with optional CRF output layer (all from scratch)."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class CRF(nn.Module):
    """Linear-chain CRF with a learnable tag-transition matrix.
    transitions[i, j] = score of moving from tag i to tag j."""

    def __init__(self, num_tags):
        super().__init__()
        self.num_tags = num_tags
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags).uniform_(-0.1, 0.1))
        self.start = nn.Parameter(torch.empty(num_tags).uniform_(-0.1, 0.1))
        self.end = nn.Parameter(torch.empty(num_tags).uniform_(-0.1, 0.1))

    def _score(self, em, tags, mask):
        # em (B,T,C), tags (B,T), mask (B,T) bool ; first position always valid
        B, T, _ = em.shape
        score = self.start[tags[:, 0]] + em[:, 0].gather(1, tags[:, :1]).squeeze(1)
        for t in range(1, T):
            m = mask[:, t].float()
            trans = self.transitions[tags[:, t - 1], tags[:, t]]
            emit = em[:, t].gather(1, tags[:, t:t + 1]).squeeze(1)
            score = score + (trans + emit) * m
        lengths = mask.long().sum(1)
        last = tags.gather(1, (lengths - 1).unsqueeze(1)).squeeze(1)
        return score + self.end[last]

    def _log_partition(self, em, mask):
        B, T, C = em.shape
        alpha = self.start.unsqueeze(0) + em[:, 0]                       # (B,C)
        for t in range(1, T):
            nxt = torch.logsumexp(alpha.unsqueeze(2) + self.transitions.unsqueeze(0), dim=1) + em[:, t]
            alpha = torch.where(mask[:, t].unsqueeze(1), nxt, alpha)     # padding keeps alpha
        return torch.logsumexp(alpha + self.end.unsqueeze(0), dim=1)

    def nll(self, em, tags, mask):
        """Mean negative log-likelihood; padding positions do not contribute."""
        return (self._log_partition(em, mask) - self._score(em, tags, mask)).mean()

    @torch.no_grad()
    def viterbi(self, em, mask):
        B, T, C = em.shape
        score = self.start.unsqueeze(0) + em[:, 0]
        history = []
        for t in range(1, T):
            cand = score.unsqueeze(2) + self.transitions.unsqueeze(0)    # (B,C_prev,C)
            best, idx = cand.max(1)
            nxt = best + em[:, t]
            m = mask[:, t].unsqueeze(1)
            score = torch.where(m, nxt, score)
            # for padded steps, back-pointer = identity so the path is unchanged
            ident = torch.arange(C, device=em.device).unsqueeze(0).expand(B, C)
            history.append(torch.where(m, idx, ident))
        score = score + self.end.unsqueeze(0)
        best_last = score.argmax(1)
        paths = [best_last]
        for idx in reversed(history):
            best_last = idx.gather(1, best_last.unsqueeze(1)).squeeze(1)
            paths.append(best_last)
        paths.reverse()
        out = torch.stack(paths, 1)                                      # (B,T)
        lengths = mask.long().sum(1).tolist()
        return [out[b, :lengths[b]].tolist() for b in range(B)]


class BiLSTMTagger(nn.Module):
    def __init__(self, vocab_size, num_tags, emb_dim=100, hidden=128, layers=2,
                 bidirectional=True, dropout=0.5, pretrained=None, freeze=False, use_crf=False):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        if pretrained is not None:
            self.emb.weight.data.copy_(torch.as_tensor(pretrained, dtype=torch.float))
            self.emb.weight.data[0].zero_()
        self.emb.weight.requires_grad = not freeze
        self.lstm = nn.LSTM(emb_dim, hidden, num_layers=layers, bidirectional=bidirectional,
                            batch_first=True, dropout=dropout if layers > 1 else 0.0)
        out_dim = hidden * (2 if bidirectional else 1)   # h = [h_fwd || h_bwd]
        self.proj = nn.Linear(out_dim, num_tags)
        self.crf = CRF(num_tags) if use_crf else None

    def emissions(self, x, lengths):
        e = self.emb(x)
        packed = pack_padded_sequence(e, lengths.cpu(), batch_first=True, enforce_sorted=False)
        h, _ = self.lstm(packed)
        h, _ = pad_packed_sequence(h, batch_first=True, total_length=x.size(1))
        return self.proj(h)

    def loss(self, x, lengths, y, mask):
        em = self.emissions(x, lengths)
        if self.crf is not None:
            return self.crf.nll(em, y, mask)
        # padding positions are labelled -100 and ignored
        return F.cross_entropy(em.reshape(-1, em.size(-1)), y.masked_fill(~mask, -100).reshape(-1),
                               ignore_index=-100)

    @torch.no_grad()
    def predict(self, x, lengths, mask):
        em = self.emissions(x, lengths)
        if self.crf is not None:
            return self.crf.viterbi(em, mask)
        pred = em.argmax(-1)
        return [pred[b, :int(lengths[b])].tolist() for b in range(x.size(0))]
