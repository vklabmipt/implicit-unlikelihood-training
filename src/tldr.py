import torch
from torch import nn
import torch.nn.functional as F
import gc

import argparse
import logging
import json
import os
import re
import random
from nltk import ngrams
import pickle
from pathlib import Path

import numpy as np
from torch.utils.data import TensorDataset, SequentialSampler, DataLoader, RandomSampler

from fairseq.custom.metrics import TrainingMetrics, Metrics
from fairseq.custom.baseline_cross_entropy import CrossEntropyCriterionWCustomMetrics
from fairseq.custom.sequence_penalty_loss import SequencePenaltyCriterion
from fairseq.custom.evaluate_utils import batch_input_sequence_by_prefix_length

from collections import defaultdict, Counter
from tqdm import tqdm, trange
from pprint import pprint


def tldr_loss(model, batch, args):
    longer_sample = batch[0].to(args.gpu)
    inp = longer_sample[:, :args.train_batch_size]
    model_output = model(input_ids=inp)
    target = longer_sample[:, 1:args.train_batch_size + 1]
    logits = model_output[0]

    lprobs = F.log_softmax(logits, dim=-1)
    assert lprobs.size(0) == 1, 'We work on flat sequences'
    nll_loss = F.nll_loss(lprobs[0], target[0], reduction='sum')
    arange = np.arange(args.train_batch_size)
    lprobs_y = lprobs[:, arange, target]
    print(torch.sum(torch.cos(np.pi * lprobs_y.exp()) + 1 < 0.5))
    loss = ((torch.cos(np.pi * lprobs_y.exp()) + 1)
            ** args.focal_gamma * (-lprobs_y)).sum()
    true_token_logits = -F.nll_loss(logits[0], target[0], reduction='none')
    ntokens = inp.numel()

    logging_output = TrainingMetrics.ranking_metrics(
        logits[0].float(), true_token_logits, None, ntokens, target[0])
    logging_output['loss'] = nll_loss.item()
    logging_output['tldr_loss'] = loss.item()
    logging_output['normalizer'] = ntokens
    logging_output['sample_size'] = ntokens
    logging_output['ntokens'] = ntokens

    loss = loss / ntokens

    return loss, logging_output
