import torch
from torch import nn
import torch.nn.functional as F
import json
import re
from nltk import ngrams
import numpy as np

from fairseq.custom.metrics import TrainingMetrics
from entmax import entmax_bisect


RETOK = re.compile(r'\w+|[^\w\s]|\n', re.UNICODE)


def jensen_shannon_divergence(probs, target):
    target_dist = F.one_hot(target, num_classes=probs.size(-1)).float()
    mixture = (probs + target_dist) / 2
    js1 = 1 / 2 * probs * torch.log(probs / mixture)
    js1[probs == 0.] = 0.
    js1 = js1.sum(-1)
    js2 = 1 / 2 * target_dist * torch.log(target_dist / mixture)
    js2[target_dist == 0.] = 0.
    js2 = js2.sum(-1)

    js = (js1 + js2)

    return js


def alpha_entropy(probs, alpha):
    if alpha == 1.:
        ent = -(probs * torch.log(probs)).sum(-1)
    else:
        ent = (probs - probs ** alpha).sum(-1) / (alpha * (alpha - 1))
    return ent


def alpha_entmax_loss(model, batch, args):
    longer_sample = batch[0].to(args.gpu)
    inp = longer_sample[:, :args.train_batch_size]
    model_output = model(input_ids=inp)
    target = longer_sample[:, 1:args.train_batch_size + 1]
    logits = model_output[0]
    alpha = torch.tensor([args.alpha], requires_grad=True,
                         device=torch.device(args.gpu))
    probs = entmax_bisect(logits, alpha)
    loss = ((probs - F.one_hot(target, num_classes=probs.size(-1))) * logits).sum(-1)
    loss += alpha_entropy(probs, args.alpha)
    loss = loss.sum()

    true_token_logits = -F.nll_loss(logits[0], target[0], reduction='none')
    ntokens = inp.numel()

    arange = np.arange(probs.size(1))
    next_token_probs = probs[:, arange, target.squeeze().tolist()]
    voc_sizes = probs.size(-1)
    smoothed_nll = -torch.mean(torch.log(
        (next_token_probs + args.laplas_eps) / (1 + args.laplas_eps * voc_sizes)
    ))

    logging_output = TrainingMetrics.ranking_metrics(
        logits[0].float(), true_token_logits, None, ntokens, target[0])
    logging_output['loss'] = loss.item()
    logging_output['smoothed_nll_loss'] = smoothed_nll.item()
    logging_output['normalizer'] = ntokens
    logging_output['sample_size'] = ntokens
    logging_output['ntokens'] = ntokens
    logging_output['js_div'] = jensen_shannon_divergence(
        probs, target).mean().item()
    print(logging_output['js_div'])

    loss = loss / ntokens

    return loss, logging_output
