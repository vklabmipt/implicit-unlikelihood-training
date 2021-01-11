import torch
import torch.nn.functional as F
import numpy as np

from fairseq.custom.metrics import TrainingMetrics, Metrics, ngram_metrics
from fairseq.custom.baseline_cross_entropy import CrossEntropyCriterionWCustomMetrics
from fairseq.custom.sequence_penalty_loss import SequencePenaltyCriterion
from fairseq.custom.evaluate_utils import batch_input_sequence_by_prefix_length

from collections import defaultdict, Counter
from tqdm import tqdm, trange
from pprint import pprint

from utils import *


def ul_loss(completions, continuation_logits, prefix_length, sequence_ngram_n):
    pred_toks = completions[:, prefix_length:].contiguous()

    mask = ngram_repeat_mask(
        pred_toks,
        sequence_ngram_n).type_as(continuation_logits)

    lprobs = F.log_softmax(continuation_logits, dim=-1)
    pred_lprobs = lprobs.view(-1, lprobs.size(2)
                              ).gather(1, pred_toks.view(-1, 1))
    one_minus_probs = torch.clamp(
        (1.0 - pred_lprobs.exp()), min=1e-20).view(pred_toks.size(0), pred_toks.size(1))
    loss = -torch.log(one_minus_probs) * mask
    loss = loss.sum()

    ntokens = pred_toks.numel()  # number of output tokens (tokens in completions)

    loss = loss / ntokens
    return loss


def ul_seq(model, batch, args):
    input_sequence = batch[0].to(args.device)
    batch = batch_input_sequence_by_prefix_length(
        input_sequence, args.prefix_length)
    batch = batch[:args.mini_batch_size]
    completions, _, continuation_logits, _ = sample_sequence(model,
                                                             batch,
                                                             args.prefix_length,
                                                             args.continuation_length,
                                                             num_samples=1,
                                                             top_k=args.top_k,
                                                             top_p=args.top_p)
    pred_toks = completions[:, args.prefix_length:].contiguous()

    mask = ngram_repeat_mask(
        pred_toks,
        args.sequence_ngram_n).type_as(continuation_logits)

    lprobs = F.log_softmax(continuation_logits, dim=-1)
    pred_lprobs = lprobs.view(-1, lprobs.size(2)
                              ).gather(1, pred_toks.view(-1, 1))
    one_minus_probs = torch.clamp(
        (1.0 - pred_lprobs.exp()), min=1e-20).view(pred_toks.size(0), pred_toks.size(1))
    loss = -torch.log(one_minus_probs) * mask
    loss = loss.sum()

    ntokens = pred_toks.numel()  # number of output tokens (tokens in completions)

    logging_output = {
        'seq_loss': loss.item(),
        'seq_sample_size': ntokens,
        'seq_ntokens': ntokens,
        'seq_nsentences': batch.size(0),
        'seq_repeat_mask': mask.sum().item(),
    }

    # Sum each statistic, which will be normalized by the number of sentences
    # in `aggregate_logging_outputs`.
    stats = defaultdict(float)
    for tok_list in pred_toks.cpu().tolist():
        ms = ngram_metrics(tok_list)
        for k, v in ms.items():
            stats[k] += v
    for k, v in stats.items():
        logging_output[k] = v

    loss = loss / ntokens
    return loss, logging_output
