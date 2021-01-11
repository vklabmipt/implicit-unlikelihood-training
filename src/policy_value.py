import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.distributions.categorical import Categorical

import copy
import numpy as np
from collections import defaultdict, Counter
from transformers import *


class PolicyValueModel(nn.Module):
    def __init__(self, config, gpt2_model):
        super().__init__()
        self.config = config
        self.gpt2_model = gpt2_model

        self.value_head1 = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.n_embd, 3 * config.n_embd),
                nn.ReLU(),
                nn.Linear(3 * config.n_embd, config.n_embd // 4),
                nn.ReLU(),
                nn.Linear(config.n_embd // 4, 1)
            ) for _ in range(self.config.n_layer)])
        self.value_head2 = nn.Linear(self.config.n_layer, 1)

        self.hidden_states_weight = nn.Parameter(
            torch.ones((self.config.n_layer)).float())

    def value(self, hidden_states):
        hidden_states = hidden_states.data
        hidden_states.requires_grad = False
        value = self.value_head2(
            torch.cat([
                self.value_head1[i](hidden_states[..., i]) for i in range(self.config.n_layer)
            ], -1)
        ).squeeze(-1)
        return value

    def forward(self, input_ids: torch.LongTensor,
                past: torch.FloatTensor = None,
                return_value: bool = False):
        output = self.gpt2_model(input_ids=input_ids, past=past)
        lm_logits, presents, hidden_states = output[:3]

        pi = None
        if return_value:
            value = self.value(torch.stack(hidden_states[1:], -1))
            return (lm_logits, presents, hidden_states, value)
        else:
            return (lm_logits, presents, hidden_states)
