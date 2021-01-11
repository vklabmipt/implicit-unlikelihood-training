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

from entmax import entmax_bisect

from alpha_entmax_training import alpha_entropy, jensen_shannon_divergence


RETOK = re.compile(r'\w+|[^\w\s]|\n', re.UNICODE)


def top_k_top_p_filtering(logits_, n=1, top_k=0, top_p=0.0, temperature=1.0,
                          filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (batch size x vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    logits = logits_.clone()
    top_k = min(top_k, logits.size(-1))  # Safety check
    if temperature != 1.0:
        logits = logits / temperature
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the
        # top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[
            0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(
            F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the
        # threshold
        sorted_indices_to_remove[...,
                                 1:] = sorted_indices_to_remove[...,
                                                                :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=1, index=sorted_indices, src=sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits


def ngram_metrics(token_list, pad=1, n=None):
    if pad in token_list:
        # remove possible padding
        token_list = token_list[:token_list.index(pad)]
    stats = defaultdict(float)
    if n is None:
        for n in range(1, 5):
            ngs = [ng for ng in ngrams(token_list, n)]
            counter = Counter([ng for ng in ngrams(token_list, n)])
            try:
                stats[f'pct_repeat_{n}grams'] = 1.0 - len(counter) / len(ngs)
            except BaseException:
                stats[f'pct_repeat_{n}grams'] = 1.0
                print('exception')
    else:
        ngs = [ng for ng in ngrams(token_list, n)]
        counter = Counter([ng for ng in ngrams(token_list, n)])
        try:
            stats['pct_repeat_ngrams'] = 1.0 - len(counter) / len(ngs)
        except BaseException:
            stats['pct_repeat_ngrams'] = 1.0
            print('exception')

    return stats


def get_datasets(dataset_paths, max_len=1536):
    """Args:
        dataset_paths: {'train': str, 'valid': str, 'test': str}
    """
    datasets = {}

    for split, fname in dataset_paths.items():
        tensor = torch.load(fname)
        right_bound = (tensor.size(0) // (max_len + 1)) * (max_len + 1)
        dataset = TensorDataset(tensor[:right_bound].view(-1, (max_len + 1)))
        datasets[split] = dataset

    return datasets


def sample_sequence(model,
                    prefix_batch,
                    prefix_length,
                    continuation_length,
                    num_samples=1,
                    top_k=0,
                    top_p=0.0,
                    temperature=1.0,
                    alpha_entmax=False,
                    output_prefix_hidden=False,
                    repetition_penalty=1.0, **kwargs):
    continuation_logits = []
    context = prefix_batch
    context = torch.cat([context] * num_samples, 0)
    assert context.size(1) == prefix_length

    prev = context
    output = context
    past = None

    log_probs = torch.zeros(
        (num_samples *
         prefix_batch.size(0),
         continuation_length))

    policy_pis = []

    for i in range(continuation_length):
        logits, past = model(input_ids=prev, past=past)[:2]
        if i == 0 and output_prefix_hidden:
            prefix_hidden = out[2]

        logits = logits[:, -1, :]
        logits = logits / temperature

        if repetition_penalty != 1.0:
            for ex_id, pert_logits in enumerate(logits):
                for token_idx in set(output[ex_id].tolist()):
                    if pert_logits[token_idx] < 0:
                        pert_logits[token_idx] *= repetition_penalty
                    else:
                        pert_logits[token_idx] /= repetition_penalty
        if alpha_entmax is False:
            if top_k == 1 and top_p == 0:
                filtered_logits = logits
                prev = logits.float().argmax(dim=1, keepdim=True)
            else:
                filtered_logits = top_k_top_p_filtering(
                    logits, top_k=top_k, top_p=top_p)
                prev = F.softmax(
                    filtered_logits,
                    dim=-
                    1).multinomial(
                    num_samples=1)

            #log_prob = F.log_softmax(filtered_logits, dim=-1)
            log_prob = F.log_softmax(logits, dim=-1)
        else:
            alpha = kwargs.get('alpha', 1.0)
            prob = entmax_bisect(
                logits,
                torch.tensor(
                    [alpha],
                    requires_grad=True,
                    device=logits.device).float())
            log_prob = torch.log(prob)
            prev = prob.multinomial(num_samples=1)
            filtered_logits = logits

        continuation_logits.append(logits)
        output = torch.cat((output, prev), dim=1)

        arange = np.arange(filtered_logits.size(0))
        next_token_logit = filtered_logits[arange,
                                           prev.squeeze().tolist()].squeeze()

        next_token_log_prob = log_prob[arange,
                                       prev.squeeze().tolist()].squeeze()
        log_probs[:, i] = next_token_log_prob
        policy_pis.append(log_prob.squeeze())

    policy_pis = torch.stack(policy_pis, 1)

    continuation_logits = torch.stack(continuation_logits, 1)
    if output_prefix_hidden:
        result = (
            output,
            log_probs,
            continuation_logits,
            policy_pis,
            prefix_hidden)
    else:
        result = (output, log_probs, continuation_logits, policy_pis)
    return result


def mle_loss(model, batch, args):
    longer_sample = batch[0].to(args.gpu)
    inp = longer_sample[:, :args.train_batch_size]
    model_output = model(input_ids=inp)
    target = longer_sample[:, 1:args.train_batch_size + 1]
    logits = model_output[0]

    lprobs = F.log_softmax(logits, dim=-1)
    assert lprobs.size(0) == 1, 'We work on flat sequences'
    loss = F.nll_loss(lprobs[0], target[0], reduction='sum')
    true_token_logits = -F.nll_loss(logits[0], target[0], reduction='none')
    ntokens = inp.numel()

    logging_output = TrainingMetrics.ranking_metrics(
        logits[0].float(), true_token_logits, None, ntokens, target[0])
    logging_output['loss'] = loss.item()
    logging_output['normalizer'] = ntokens
    logging_output['sample_size'] = ntokens
    logging_output['ntokens'] = ntokens

    loss = loss / ntokens

    return loss, logging_output


def ngram_count(token_list, pad=1):
    if pad in token_list:
        # remove possible padding
        token_list = token_list[:token_list.index(pad)]
    counters = []
    for n in range(1, 5):
        ngs = [ng for ng in ngrams(token_list, n)]
        counter = Counter([ng for ng in ngrams(token_list, n)])
        counters.append(counter)
    return counters


def ngram_repeat_mask(xs, n):
    mask = torch.zeros_like(xs)
    for i, x in enumerate(xs):
        seen = set()
        xl = x.tolist()
        for j in range(len(x) - n):
            ng = tuple(xl[j:j + n])
            if ng in seen:
                mask[i, j:j + n] = 1
            seen.add(ng)
    return mask


def tokenize(text):
    # ref:
    # https://github.com/facebookresearch/ParlAI/blob/4da3ec0bdcf1db2c3a5bd5723d1275c32a891192/parlai/core/dict.py#L451
    return RETOK.findall(text)


def get_text_continuation(bpe_completion, tokenizer, args):
    completion = tokenizer.decode(bpe_completion)
    bpe_prefix, bpe_continuation = bpe_completion[:
                                                  args.prefix_length], bpe_completion[args.prefix_length:]
    prefix = tokenizer.decode(bpe_prefix)

    if prefix in completion:
        continuation = completion.replace(prefix, '')
    else:
        prefix_ = ' '.join(prefix.split(' ')[:-2])
        continuation = completion.replace(prefix_, '')

    continuation_tokens = tokenize(continuation)
    return continuation_tokens


def save_completion_metrics(
        bpe_metrics, word_metrics, text_completions, config, args, add=None):
    add = '' if add is None else add
    if add != '':
        add = f'_{add}' if add[0] != '_' else add

    if args.num_beams == 1:
        file_name = 'completion__{model}__spl_{split}__topk_{topk}__topp_{topp}__pfl_{pfl}__cnl_{cnl}'.format(
            model=args.model_name,
            split=args.eval_split,
            topk=args.top_k,
            topp=args.top_p,
            pfl=args.prefix_length,
            cnl=args.continuation_length
        )
    else:
        file_name = 'completion__{model}__spl_{split}__beam_{beam}__pfl_{pfl}__cnl_{cnl}'.format(
            model=args.model_name,
            split=args.eval_split,
            beam=args.num_beams,
            pfl=args.prefix_length,
            cnl=args.continuation_length
        )
    outfile = os.path.join(args.output_dir,
                           file_name)

    json.dump({'bpe_metrics': bpe_metrics,
               'word_metrics': word_metrics,
               'config': config,
               'completions': text_completions},
              open(outfile + add + '.json',
                   'w'))
    print("%s metrics written to %s" % (args.mode, outfile + add + '.json'))


def save_singletoken_metrics(
        metrics, config, args, best=False, train_iter=None):
    output_dir = args.output_dir if not best else os.path.join(
        args.output_dir, 'best')
    outfile = os.path.join(
        output_dir,
        'singletoken__{model}__spl_{split}__bsz_{bsz}{iter}.json'.format(
            model=args.model_name,
            split=args.eval_split,
            bsz=args.batch_size_singletoken,
            iter='_%d' %
            train_iter if train_iter is not None else '',
        ))

    json.dump({'metrics': metrics,
               'config': config}, open(outfile, 'w'))
    print("%s metrics written to %s" % (args.mode, outfile))


def save_singletoken_sampling_metrics(
        metrics,
        config,
        args,
        top_k=1,
        top_p=0.0,
        best=False,
        train_iter=None):
    output_dir = args.output_dir if not best else os.path.join(
        args.output_dir, 'best')
    outfile = os.path.join(
        output_dir,
        'singletoken_{model}__topk_{topk}_topp_{topp}_spl_{split}__bsz_{bsz}{iter}.json'.format(
            model=args.model_name,
            topk=top_k,
            topp=top_p,
            split=args.eval_split,
            bsz=args.batch_size_singletoken,
            iter='_%d' %
            train_iter if train_iter is not None else '',
        ))

    json.dump({'metrics': metrics,
               'config': config}, open(outfile, 'w'))
    print("%s metrics written to %s" % (args.mode, outfile))


def save_acc_metrics(
        metrics, config, args, best=False):
    output_dir = args.output_dir if not best else os.path.join(
        args.output_dir, 'best')
    outfile = os.path.join(output_dir,
                           'acc_spl_{split}.json'.format(
                               split=args.eval_split
                           ))

    json.dump({'metrics': metrics,
               'config': config}, open(outfile, 'w'))
    print("%s metrics written to %s" % (args.mode, outfile))


def eval_singletoken_argmax(model, args, dataset_paths, config,
                            train_iter=None, batch_size=None):
    batch_size = batch_size if batch_size is not None else args.batch_size_singletoken
    datasets = get_datasets(dataset_paths, max_len=batch_size)
    eval_sampler = SequentialSampler(datasets[args.eval_split])
    eval_dataloader = DataLoader(
        datasets[args.eval_split], sampler=eval_sampler, batch_size=1)

    model.eval()

    logging_outputs = []
    predicted_tokens = []
    target_tokens = []
    with torch.no_grad():
        for i, batch in tqdm(enumerate(eval_dataloader),
                             desc="Evaluating", total=len(eval_dataloader)):
            longer_sample = batch[0].to(args.gpu)
            inp = longer_sample[:, :args.batch_size_singletoken]
            model_output = model(input_ids=inp)
            target = longer_sample[:, 1:]
            logits = model_output[0]
            lprobs = F.log_softmax(logits, dim=-1)
            assert lprobs.size(0) == 1, 'We work on flat sequences'
            loss = F.nll_loss(lprobs[0], target[0], reduction='sum')
            true_token_logits = - \
                F.nll_loss(logits[0], target[0], reduction='none')

            pred = lprobs.argmax(dim=-1).view(-1).tolist()
            predicted_tokens.extend(pred)
            ntokens = inp.numel()

            logging_output = TrainingMetrics.ranking_metrics(
                logits[0].float(), true_token_logits, None, ntokens, target[0])
            logging_output['loss'] = loss.item()
            logging_output['normalizer'] = ntokens
            logging_output['sample_size'] = ntokens
            logging_output['ntokens'] = ntokens
            logging_outputs.append(logging_output)

            # for human uniq
            target_tokens.extend(target.view(-1).tolist())

    logging_average = CrossEntropyCriterionWCustomMetrics.aggregate_logging_outputs(
        logging_outputs)
    logging_average['ppl'] = 2 ** logging_average['loss']
    logging_average['uniq'] = len(set(predicted_tokens))
    logging_average['human_uniq'] = len(set(target_tokens))

    save_singletoken_metrics(
        logging_average,
        config.to_dict(),
        args,
        train_iter=train_iter)
    return logging_average


def eval_acc(model, args, dataset_paths, config, batch_size=None):
    batch_size = batch_size if batch_size is not None else args.batch_size_singletoken
    datasets = get_datasets(dataset_paths, max_len=batch_size)
    eval_sampler = SequentialSampler(datasets[args.eval_split])
    eval_dataloader = DataLoader(
        datasets[args.eval_split], sampler=eval_sampler, batch_size=1)
    model.eval()

    logging_outputs = []
    predicted_tokens = []
    target_tokens = []
    with torch.no_grad():
        for i, batch in tqdm(enumerate(eval_dataloader),
                             desc="Evaluating", total=len(eval_dataloader)):
            longer_sample = batch[0].to(args.gpu)
            inp = longer_sample[:, :args.batch_size_singletoken]
            model_output = model(input_ids=inp)
            target = longer_sample[:, 1:]
            logits = model_output[0]
            lprobs = F.log_softmax(logits, dim=-1)
            assert lprobs.size(0) == 1, 'We work on flat sequences'
            loss = F.nll_loss(lprobs[0], target[0], reduction='sum')
            true_token_logits = - \
                F.nll_loss(logits[0], target[0], reduction='none')

            pred = lprobs.argmax(dim=-1).view(-1).tolist()
            predicted_tokens.extend(pred)
            ntokens = inp.numel()

            logging_output = TrainingMetrics.ranking_metrics(
                logits[0].float(), true_token_logits, None, ntokens, target[0])
            logging_output['loss'] = loss.item()
            logging_output['normalizer'] = ntokens
            logging_output['sample_size'] = ntokens
            logging_output['ntokens'] = ntokens
            logging_outputs.append(logging_output)

            # for human uniq
            target_tokens.extend(target.view(-1).tolist())

    logging_average = CrossEntropyCriterionWCustomMetrics.aggregate_logging_outputs(
        logging_outputs)

    save_acc_metrics(
        logging_average,
        config.to_dict(),
        args)
    return logging_average


def eval_singletoken(model,
                     args,
                     dataset_paths,
                     config,
                     top_k=1,
                     top_p=0.0,
                     t=1.0,
                     train_iter=None,
                     batch_size=None):
    alpha_entmax = args.alpha_entmax

    batch_size = batch_size if batch_size is not None else args.batch_size_singletoken
    datasets = get_datasets(dataset_paths, max_len=batch_size)
    eval_sampler = SequentialSampler(datasets[args.eval_split])
    eval_dataloader = DataLoader(
        datasets[args.eval_split], sampler=eval_sampler, batch_size=1)

    model.eval()

    logging_outputs = []
    predicted_tokens = []
    target_tokens = []
    with torch.no_grad():
        for i, batch in tqdm(enumerate(eval_dataloader),
                             desc="Evaluating", total=len(eval_dataloader)):
            longer_sample = batch[0].to(args.gpu)
            inp = longer_sample[:, :args.batch_size_singletoken]
            model_output = model(input_ids=inp)
            target = longer_sample[:, 1:]
            logits = model_output[0]
            log_softmax_probs = F.log_softmax(logits, dim=-1)
            nll = F.nll_loss(log_softmax_probs[0], target[0], reduction='sum')
            true_token_logits = - \
                F.nll_loss(logits[0], target[0], reduction='none')

            if alpha_entmax is False:
                filtered_logits = top_k_top_p_filtering(
                    logits.squeeze(0), top_k=args.top_k, top_p=args.top_p).unsqueeze(0)
                prev = F.softmax(
                    filtered_logits.view(filtered_logits.shape[1:]),
                    dim=-1).multinomial(num_samples=1).unsqueeze(0).squeeze(-1)
                probs = F.softmax(filtered_logits, dim=-1)
            else:
                probs = entmax_bisect(logits, torch.tensor(
                    [args.alpha], requires_grad=True, device=torch.device(args.gpu)).float())
            arange = np.arange(logits.size(1))

            next_token_probs = probs[:, arange, target.squeeze().tolist()]
            voc_sizes = probs.size(-1)
            smoothed_nll = -torch.mean(torch.log(
                (next_token_probs + args.laplas_eps) / (1 + args.laplas_eps * voc_sizes)
            ))

            pred = probs.view(-1, probs.size(-1)
                              ).multinomial(num_samples=1).view(probs.shape[:-1])
            predicted_tokens.extend(pred.view(-1).tolist())
            ntokens = inp.numel()

            rep_logits = torch.zeros_like(logits)
            rep_logits[:, arange, pred.squeeze().tolist()] = 1
            logging_output = TrainingMetrics.ranking_metrics(
                rep_logits[0].float(), true_token_logits, None, ntokens, target[0])
            logging_output['loss'] = nll.item()
            logging_output['smoothed_nll_loss'] = smoothed_nll.item()
            logging_output['normalizer'] = ntokens
            logging_output['sample_size'] = ntokens
            logging_output['ntokens'] = ntokens
            logging_output['js_div'] = jensen_shannon_divergence(
                probs, target).mean().item()
            if args.token_loss == 'alpha_entmax':
                loss = ((probs - F.one_hot(target,
                                           num_classes=probs.size(-1))) * logits).sum(-1)
                loss += alpha_entropy(probs, args.alpha)
                logging_output['alpha_entmax_loss'] = loss.mean().item()
            logging_outputs.append(logging_output)

            # for human uniq
            target_tokens.extend(target.view(-1).tolist())

    logging_average = CrossEntropyCriterionWCustomMetrics.aggregate_logging_outputs(
        logging_outputs)
    logging_average['e_ppl'] = np.exp(
        np.mean([x['smoothed_nll_loss'] for x in logging_outputs]))
    # aggregate_logging_outputs does division by log(2) of loss
    logging_average['ppl'] = 2**logging_average['loss']
    logging_average['human_uniq'] = len(set(target_tokens))
    logging_average['uniq'] = len(set(predicted_tokens))
    logging_average['wrep'] = np.mean(
        [v for k, v in logging_average.items() if k.startswith('wrong_repeat')])
    logging_average['rep'] = np.mean(
        [v for k, v in logging_average.items() if k.startswith('repeat')])
    logging_average['js_div'] = np.mean([x['js_div'] for x in logging_outputs])
    if args.token_loss == 'alpha_entmax':
        logging_average['alpha_entmax_loss'] = np.mean(
            [x['alpha_entmax_loss'] for x in logging_outputs])

    save_singletoken_sampling_metrics(
        logging_average,
        config.to_dict(),
        args,
        top_k=top_k,
        top_p=top_p,
        train_iter=train_iter)

    return logging_average


def eval_completion(model, tokenizer, args, dataset_paths, config,
                    train_iter=None):
    eval_datasets = get_datasets(
        dataset_paths, max_len=args.batch_size_completion)

    eval_sampler = SequentialSampler(eval_datasets[args.eval_split])
    eval_dataloader = DataLoader(
        eval_datasets[args.eval_split], sampler=eval_sampler, batch_size=1)

    model.eval()

    logging_outputs = []
    predicted_tokens = []
    target_tokens = []
    with torch.no_grad():
        all_text_completions = []

        bpe_ngram_metrics = Metrics(pad=-1)
        word_ngram_metrics = Metrics(pad=-1)

        for i, batch in tqdm(enumerate(eval_dataloader),
                             desc="Evaluating", total=len(eval_dataloader)):
            if i > args.compl_steps:
                break

            input_sequence = batch[0].cuda(non_blocking=True)
            if input_sequence.size(1) < args.prefix_length:
                continue

            # Predict the completions.
            batch = batch_input_sequence_by_prefix_length(
                input_sequence, args.prefix_length)
            if args.num_beams == 1:
                output = sample_sequence(
                    model,
                    batch,
                    args.prefix_length,
                    args.continuation_length,
                    num_samples=1,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    alpha_entmax=args.alpha_entmax,
                    alpha=args.alpha)
                bpe_completions = output[0].tolist()
            else:
                bpe_completions = model.generate(
                    input_ids=batch,
                    num_beams=args.num_beams,
                    eos_token_id=None,
                    pad_token_id=None,
                    max_length=args.prefix_length +
                    args.continuation_length).tolist()

            # Extract continuations from the predicted
            # completions.
            bpe_continuations = []
            text_continuations = []
            for bpe_completion in bpe_completions:
                bpe_continuations.append(
                    bpe_completion[args.prefix_length:])
                text_continuations.append(
                    get_text_continuation(
                        bpe_completion, tokenizer, args))
                all_text_completions.append(
                    tokenizer.decode(bpe_completion))

            # Only keep continuations with at least one 4-gram
            # (A short continuation may occur due to predicted whitespace, then tokenizing, despite being
            #  normal length in BPE tokens).
            text_continuations = [
                c for c in text_continuations if len(c) > 3]

            # Update metrics with this batch of
            # continuations.
            bpe_ngram_metrics.update(bpe_continuations)
            word_ngram_metrics.update(text_continuations)

            # Save the (possibly intermediate) metrics.
        train_iter = str(train_iter) if train_iter is not None else None
        if args.rank == args.start_rank:
            save_completion_metrics(
                bpe_metrics=bpe_ngram_metrics.report(
                    'bpe_%s' %
                    args.eval_split),
                word_metrics=word_ngram_metrics.report(
                    'word_%s' %
                    args.eval_split),
                text_completions=all_text_completions,
                config=config.to_dict(),
                args=args,
                add=train_iter)
