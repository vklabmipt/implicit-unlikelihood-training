from collections import defaultdict
from fairseq.custom.evaluate_utils import batch_input_sequence_by_prefix_length
from unlikelihood import ul_loss
from utils import *
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
import torch.distributed as dist
import copy

from transformers import *

# for correct work of apex:
import pickle
import math
from transformers import activations


def gelu_new(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi)
                                     * (x + 0.044715 * torch.pow(x, 3.0))))


activations.ACT2FN['gelu_new'] = gelu_new


class OnceRewardTrainer():
    def __init__(self, model, config, tokenizer, optimizer, **kwargs):
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.device = kwargs.get('device', 'cpu')
        self.mini_batch_size = kwargs.get('mini_batch_size', 4)
        self.prefix_length = kwargs.get('prefix_length', 50)
        self.continuation_length = kwargs.get('continuation_length', 100)
        self.top_k = kwargs.get('top_k', 1)
        self.top_p = kwargs.get('top_p', 0.)
        self.temperature = kwargs.get('temperature', 1.)
        self.n_samples = kwargs.get('n_samples', 1)
        self.ngram_reward = kwargs.get('ngram_reward', 4)
        self.max_grad_norm = kwargs.get('max_grad_norm', 10)
        self.psy_type = kwargs.get('psy_type', 'reward')
        self.repetition_penalty = kwargs.get('repetition_penalty', 1.0)

        if self.psy_type not in ['reward', 'advantage']:
            raise ValueError
        elif self.psy_type == 'advantage':
            self.value_model = ValueModel(
                config.n_embd, config.n_layer).to(
                self.device)
            self.value_loss_f = nn.SmoothL1Loss(reduction='sum')

    def sample(self, batch_, args):
        prefix_hidden = None
        input_sequence = batch_[0].to(args.gpu)
        batch = batch_input_sequence_by_prefix_length(
            input_sequence, args.prefix_length)
        batch = batch[:self.mini_batch_size]
        batch_size = batch.size(0)

        output_prefix_hidden = True if self.psy_type == 'advantage' else False
        completions = sample_sequence(
            self.model,
            batch,
            args.prefix_length,
            args.continuation_length,
            num_samples=self.n_samples,
            top_k=self.top_k,
            top_p=self.top_p,
            temperature=self.temperature,
            output_prefix_hidden=output_prefix_hidden,
            repetition_penalty=args.repetition_penalty)[0]
        torch.cuda.empty_cache()

        ntokens = 0
        pred_toks = []
        bpe_completions = completions.tolist()
        bpe_continuations = []
        text_continuations = []
        for bpe_completion in bpe_completions:
            bpe_continuations.append(bpe_completion[self.prefix_length:])
            text_continuations.append(
                get_text_continuation(
                    bpe_completion, self.tokenizer, args))

        rep_rewards = self._calc_rewards(bpe_continuations)
        print('Repetition reward: ', rep_rewards)

        rewards = rep_rewards
        psy, value = self._calc_psy(rewards, prefix_hidden, args)

        text_continuations = [c for c in text_continuations if len(c) > 3]

        pred_toks = completions[:, self.prefix_length:].contiguous()
        ntokens += pred_toks.numel()

        logging_output = {
            'seq_sample_size': ntokens,
            'seq_ntokens': ntokens,
            'seq_nsentences': self.n_samples * batch_size
        }

        stats = defaultdict(float)
        for tok_list in pred_toks.cpu().tolist():
            ms = ngram_metrics(tok_list, pad=-1)
            for k, v in ms.items():
                stats[k] += v
        for k, v in stats.items():
            logging_output[k] = v

        samples = {
            'batch': batch_,
            'completions': completions,
            'rewards': rewards,
            'psy': psy,
            'value': value
        }

        return samples, logging_output, batch_size

    def _calc_rewards(self, bpe_continuations):
        seq_rep_ns = np.array([ngram_metrics(
            x, pad=-1, n=self.ngram_reward)['pct_repeat_ngrams'] for x in bpe_continuations])
        rep = np.mean([1 - y for y in seq_rep_ns])
        rewards = torch.FloatTensor(
            [1 - y for y in seq_rep_ns]).to(self.device)

        return rewards

    def _calc_psy(self, rewards, prefix_hidden, args):
        value = None
        if self.psy_type == 'reward':
            mean = rewards.clone().detach()
            std = rewards.clone().detach() ** 2
            dist.all_reduce(
                mean,
                op=dist.ReduceOp.SUM,
                group=dist.group.WORLD,
                async_op=False)
            dist.all_reduce(
                std,
                op=dist.ReduceOp.SUM,
                group=dist.group.WORLD,
                async_op=False)
            mean = mean.mean() / args.world_size
            std = std.sum() ** 0.5
            psy = rewards - mean
        else:
            value = self.value_model(prefix_hidden)
            psy = rewards - value.data

        return psy, value

    def ppo_step(self, samples, args, make_step=True):
        psy = samples['psy']
        batch = samples['batch']
        rewards = samples['rewards']
        input_ids = samples['completions'].to(args.device)

        if args.ppo_epoch > 1:
            with torch.no_grad():
                vocab_log_pis_old = F.log_softmax(self.model(
                    input_ids)[0][:, self.prefix_length - 1:-1, :], dim=-1)
                log_pis_old = vocab_log_pis_old.reshape(-1, vocab_log_pis_old.size(-1))[
                    np.arange(vocab_log_pis_old.size(0) * vocab_log_pis_old.size(1)), input_ids[:, self.prefix_length:].reshape(-1)
                ].reshape(vocab_log_pis_old.shape[:-1]).data

        for k in range(args.ppo_epoch):
            vocab_log_pis = F.log_softmax(self.model(
                input_ids)[0][:, self.prefix_length - 1:-1, :], dim=-1)
            log_pis = vocab_log_pis.reshape(-1, vocab_log_pis.size(-1))[
                np.arange(vocab_log_pis.size(0) * vocab_log_pis.size(1)), input_ids[:, self.prefix_length:].reshape(-1)
            ].reshape(vocab_log_pis.shape[:-1])

            if args.ppo_epoch == 1:
                ratio = (log_pis.sum(-1))
                j = (ratio * psy).sum()
            else:
                ratio = (log_pis.sum(-1) - log_pis_old.sum(-1)).exp()
                clipped_ratio = torch.clamp(
                    ratio, 1 - args.clip_range, 1 + args.clip_range)
                j = torch.min(clipped_ratio * psy, ratio * psy).sum()

            loss = args.pg_coef * (-j) / (log_pis.size(1)
                                          * args.mini_batch_size * args.world_size)
            if args.add_mle_loss_to_pg is True:
                loss += mle_loss(self.model, batch, args)[0]
            if args.add_kl_loss_to_pg:
                kl = (vocab_log_pis.exp() * (vocab_log_pis - \
                      vocab_log_pis_old)).sum(-1).mean(-1).mean()
                loss += kl
            if args.add_ul_loss_to_pg:
                loss += ul_loss(samples['completions'],
                                samples['continuation_logits'],
                                self.prefix_length,
                                4)
            loss = loss / args.ppo_epoch

            ent = (-vocab_log_pis.exp() * vocab_log_pis).sum(-1).mean()
            print('Entropy: ', ent.item())

            if make_step is True:
                try:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=self.max_grad_norm)
                    self.optimizer.step()
                except BaseException:
                    print('Oopds... backward exception')

        return torch.tensor([loss.item(), rewards.mean().item(), 0, 0])
