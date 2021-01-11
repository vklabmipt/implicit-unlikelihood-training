import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.distributions.categorical import Categorical

import copy
import numpy as np
from collections import defaultdict, Counter
from transformers import *
from policy_value import PolicyValueModel

from utils import *
from fairseq.custom.evaluate_utils import batch_input_sequence_by_prefix_length
from collections import defaultdict


class TimeRewardTrainer():
    def __init__(self, model, optimizer, **kwargs):
        self.gamma = kwargs.get('gamma', 0.99)
        self.lamda = kwargs.get('lamda', 0.85)

        self.psy_type = kwargs.get('psy_type', 'advantage')

        self.n_gather = kwargs.get('n_gather', 1)
        self.epochs = kwargs.get('epochs', 1)
        self.n_samples = kwargs.get('n_samples', 1)
        self.n_of_gram = kwargs.get('n_of_gram', 4)
        self.reward_type = kwargs.get('reward_type', 'reward_for_past')
        self.prefix_length = kwargs.get('prefix_length', 50)
        self.continuation_length = kwargs.get('continuation_length', 100)

        self.device = kwargs.get('device', 'cpu')
        self.max_grad_norm = kwargs.get('max_grad_norm', 1.)

        self.mini_batch_size = kwargs.get('mini_batch_size', 1)

        self.model = model
        self.optimizer = optimizer

        self.value_loss = kwargs.get('value_err_f', 'smooth_l1')
        if self.value_loss == 'smooth_l1':
            self.value_loss = nn.SmoothL1Loss()
        elif self.value_loss == 'l2':
            self.value_loss = lambda x: 0.5 * nn.MSELoss()(x)

        self.value_loss = nn.SmoothL1Loss()

    def sample(self, batch_, filter_method, args):
        input_sequence = batch_[0].to(args.gpu)
        batch = batch_input_sequence_by_prefix_length(
            input_sequence, args.prefix_length)
        batch = batch[:self.mini_batch_size]
        n_sent = batch.size(0)

        continuation_logits = []
        log_pis = []
        context = batch

        out = self.model(input_ids=batch, past=None)
        logits, past, hid = out[:3]
        hid = torch.stack(hid[1:], -1)
        if self.psy_type == 'advantage':
            value = self.model.value(hid[:, -1, ...])
        else:
            value = torch.zeros((self.mini_batch_size), device=logits.device)
        target = context[:, 1:]
        lprobs = F.log_softmax(logits, dim=-1)

        cur_hid = torch.cat([hid[:, -1, ...]] * self.n_samples, 0)
        past = [torch.cat([x] * self.n_samples, 1) for x in past]
        logits = torch.cat([logits] * self.n_samples, 0)
        value = torch.cat([value] * self.n_samples, 0)
        prev = torch.cat([context] * self.n_samples, 0)
        output = torch.cat([context] * self.n_samples, 0)

        log_probs = torch.zeros(
            self.n_samples * n_sent,
            self.continuation_length,
            device=self.device).float()
        values = torch.zeros(
            self.n_samples * n_sent,
            self.continuation_length,
            device=self.device).float()
        hidden = torch.zeros(
            self.n_samples *
            n_sent,
            self.continuation_length,
            self.model.config.n_embd,
            self.model.config.n_layer,
            device=self.device).float()

        for i in range(self.continuation_length):
            values[:, i] = value
            hidden[:, i, ...] = cur_hid
            logits = logits[:, -1, :]
            filtered_logits = filter_method(logits)
            prev = F.softmax(
                filtered_logits,
                dim=-1).multinomial(
                num_samples=1)
            continuation_logits.append(logits)
            output = torch.cat((output, prev), dim=1)

            arange = np.arange(logits.size(0))
#             next_token_log_prob = F.log_softmax(
# filtered_logits, dim=-1)[arange, prev.squeeze().tolist()].squeeze()
            next_token_log_prob = F.log_softmax(
                logits, dim=-1)[arange, prev.squeeze().tolist()].squeeze()
            log_probs[:, i] = next_token_log_prob

            if i != self.continuation_length - 1:
                out = self.model(prev, past=past)
                logits, past, cur_hid = out[:3]
                cur_hid = torch.stack(cur_hid[1:], -1)
                cur_hid = cur_hid[:, -1, ...]
                if self.psy_type == 'advantage':
                    value = self.model.value(cur_hid)
                else:
                    value = torch.zeros(value.size(), device=value.device)

            else:
                out = self.model(prev, past=past)
                _, past, cur_hid = out[:3]
                cur_hid = torch.stack(cur_hid[1:], -1)
                cur_hid = cur_hid[:, -1, ...]
                if self.psy_type == 'advantage':
                    last_value = self.model.value(cur_hid)
                else:
                    last_value = torch.zeros(value.size(), device=value.device)

        continuation_logits = torch.stack(continuation_logits, 1)

        contin = output[:, self.prefix_length:].view(
            n_sent, self.n_samples, self.continuation_length).data
        values = values.view(n_sent, self.n_samples, self.continuation_length)
        hidden = hidden.view(n_sent, self.n_samples,
                             self.continuation_length, -1, 12)
        last_value = last_value.view(n_sent, self.n_samples).data

        rewards = self._calc_rewards(context, contin)
        values = values.resize(n_sent, self.n_samples,
                               self.n_gather, rewards.size(-1))
        values = values.mean(-2)

        advantages = self._calc_advantages(rewards, values, last_value).data
        total_rewards = self._calc_total_rewards(rewards).data

        samples = {
            'obs': output.view(n_sent, self.n_samples, -1),
            'hidden': hidden,
            'actions': contin,
            'log_pis': log_probs.view(n_sent, self.n_samples, self.continuation_length),
            'values': values,
            'last_value': last_value,
            'total_rewards': total_rewards,
            'advantages': advantages
        }

        ntokens = contin.contiguous().numel()

        logging_output = {
            'seq_sample_size': ntokens,
            'seq_ntokens': ntokens,
            'seq_nsentences': n_sent * self.n_samples
        }
        stats = defaultdict(float)
        for x in contin:
            for y in x:
                tok_list = y.contiguous().cpu().tolist()
                ms = ngram_metrics(tok_list, pad=-1)
                for k, v in ms.items():
                    stats[k] += v
        for k, v in stats.items():
            logging_output[k] = v

        return samples, n_sent, logging_output

    def _calc_rewards(self, context: torch.LongTensor,
                      contin: torch.LongTensor) -> torch.FloatTensor:
        continuation_length = self.continuation_length
        n_sent = context.size(0)

        if self.n_gather > 1:
            rewards = torch.zeros(
                n_sent,
                self.n_samples,
                continuation_length //
                self.n_gather,
                device=self.device).float()
        else:
            rewards = torch.zeros(
                n_sent,
                self.n_samples,
                continuation_length,
                device=self.device).float()
            if self.reward_type == 'seq_rep':
                rewards_to_go = torch.zeros(
                    n_sent,
                    self.n_samples,
                    continuation_length,
                    device=self.device).float()

        for i in range(n_sent):
            for j in range(self.n_samples):
                #                 if self.n_gather > 1:
                #                     split = [(context[i].tolist() + contin[i, j, t : t + self.n_gather].tolist())
                #                              for t in range(continuation_length // self.n_gather)]
                #                     for t in range(len(split)):
                #                         rewards[i, j, t] = 1 - ngram_metrics(split[t], pad=-1)['pct_repeat_3grams']
                #                 else:
                for t in range(continuation_length):
                    seq = contin[i, j, t:].tolist()

                    def split_on_ngrams(l, n): return [
                        l[m: m + n] for m in range(len(l) - n)]

                    if self.reward_type == 'reward_for_past':
                        hist = (context[i, :].tolist() +
                                contin[i, j, :t].tolist())
                        cur_ngram = (context[i, :].tolist(
                        ) + contin[i, j, :t + 1].tolist())[-self.n_of_gram:]
                        hist_ngrams = split_on_ngrams(hist, self.n_of_gram)
                        rewards[i, j, t] = 1 if cur_ngram not in hist_ngrams else 0

                    elif self.reward_type == 'reward_for_future':
                        cur_ngram = (context[i, :].tolist(
                        ) + contin[i, j, :t + 1].tolist())[-self.n_of_gram:]
                        future = contin[i, j, t + 1:].tolist()
                        future_ngrams = split_on_ngrams(future, self.n_of_gram)
                        rewards[i, j, t] = 1 if cur_ngram not in future_ngrams else 0
#                         elif self.reward_type == 'window':
#                             seq = (context[i].tolist() + contin[i, j, :t].tolist())
#                             rewards[i, j, t] = 1 - ngram_metrics(seq[-32:], pad=-1)['pct_repeat_1grams']
#                         elif self.reward_type == 'seq_rep':
#                             rewards_to_go[i, j, t] = 1. - ngram_metrics(seq, pad=-1)['pct_repeat_3grams']
#                             if t > 0 and continuation_length - t >= 4:
#                                 rewards[i, j, t-1] = rewards_to_go[i, j, t-1] - rewards_to_go[i, j, t]
#                             elif continuation_length - t < 4:
#                                 rewards[i, j, t-1] = rewards[i, j, t-2]
#                         elif self.reward_type == 'pure_seq_rep':
#                             rewards[i, j, t] = 1. - ngram_metrics(contin[i, j].tolist(), pad=-1)['pct_repeat_4grams']
        return rewards

    def _calc_advantages(self, rewards: torch.FloatTensor,
                         values: torch.FloatTensor,
                         last_value: torch.FloatTensor) -> torch.FloatTensor:
        continuation_length = rewards.size(-1)
        advantages = torch.zeros_like(rewards, device=self.device).float()
        last_advantage = 0
        renorm_coef = 1

        for t in reversed(range(continuation_length)):
            delta = rewards[:, :, t] + self.gamma * \
                last_value - values[:, :, t] * renorm_coef

            last_advantage = delta + self.gamma * self.lamda * last_advantage
            advantages[:, :, t] = last_advantage
            last_value = values[:, :, t] * renorm_coef

            if self.psy_type == 'advantage' and self.reward_type != 'seq_rep':
                renorm_coef = 1 + self.gamma * renorm_coef

        return advantages

    def _calc_total_rewards(self, rewards: torch.FloatTensor
                            ) -> torch.FloatTensor:
        continuation_length = rewards.size(-1)
        prev_step = 0
        total_rewards = torch.zeros_like(rewards, device=self.device).float()

        norm_coef = 1
        for t in reversed(range(continuation_length)):
            total_rewards[:, :, t] = 1. / norm_coef * \
                (rewards[:, :, t] + self.gamma * prev_step)
            prev_step = rewards[:, :, t] + self.gamma * prev_step

            if self.psy_type == 'advantage' and self.reward_type != 'seq_rep':
                norm_coef = 1 + self.gamma * norm_coef

        return total_rewards

    def train(
            self,
            samples,
            filter_method: callable,
            progress: float,
            args,
            make_step: bool = True) -> torch.FloatTensor:
        train_info = []

        for _ in range(self.epochs):
            if self.n_samples > 1:
                for episode_id in range(self.mini_batch_size):
                    indexes = torch.randperm(self.n_samples)
                    for start in range(
                            0, self.n_samples, self.mini_batch_size):
                        end = start + self.mini_batch_size
                        mini_batch_indexes = indexes[start: end]
                        mini_batch = {}
                        for k, v in samples.items():
                            mini_batch[k] = v[episode_id,
                                              mini_batch_indexes, ...]
                        res = self.step(mini_batch,
                                        args.ratio_clip_range,
                                        filter_method,
                                        progress,
                                        make_step)
                        train_info.append(res)
            else:
                mini_batch = {}
                for k, v in samples.items():
                    mini_batch[k] = v[:, 0, ...]
                res = self.step(mini_batch,
                                args.ratio_clip_range,
                                filter_method,
                                progress,
                                make_step)
                train_info.append(res)

        train_info = torch.stack(train_info, dim=0)

        return torch.mean(train_info, dim=0)

    def step(self,
             samples: dict,
             clip_range: float,
             filter_method: callable,
             progress: float,
             make_step: bool = True) -> torch.FloatTensor:

        input_ids = samples['obs']
        hidden = samples['hidden']
        sampled_action = samples['actions']
        sampled_return = samples['values'] + samples['advantages']
        sampled_advantage = self._normalize(
            samples['advantages']) if self.n_samples > 1 else samples['advantages']
        sampled_neg_log_pi = -samples['log_pis']
        sampled_value = samples['values']
        total_rewards = samples['total_rewards']

        if self.psy_type == 'reward':
            psy = (total_rewards - total_rewards.mean(0)) / \
                (total_rewards.std(0) + 1e-8)
        elif self.psy_type == 'advantage':
            psy = sampled_advantage

        if self.epochs > 1:
            sampled_value = sampled_value.data
            sampled_neg_log_pi = sampled_neg_log_pi.data
            pi = []
            cl_pi = []
            value = []
            for t in range(self.continuation_length):
                hid = hidden[:, t, ...]
                logits = self.model.gpt2_model.lm_head(hid[..., -1])
                pi.append(filter_method(logits))
                value.append(self.model.value(hid).squeeze(-1).view(-1))

            value = torch.stack(value, 1)
            value = value.resize(value.size(
                0), self.n_gather, total_rewards.size(-1))
            value = value.mean(-2)

            pi = torch.stack(pi, 1).squeeze(2)
            pi = Categorical(logits=pi)

            log_pi = pi.log_prob(sampled_action)

            ratio: torch.Tensor = torch.exp(sampled_neg_log_pi + log_pi)

            ratio = ratio.resize(ratio.size(
                0), self.n_gather, total_rewards.size(-1))
            ratio = ratio.mean(-2)

            clipped_ratio = ratio.clamp(min=1.0 - clip_range,
                                        max=1.0 + clip_range)
            policy_reward = torch.min(ratio * psy,
                                      clipped_ratio * psy)
        else:
            log_pi = -sampled_neg_log_pi
            log_pi = log_pi.resize(log_pi.size(0), self.n_gather, psy.size(-1))
            log_pi = log_pi.sum(-2)
            policy_reward = log_pi * psy
            value = sampled_value

        policy_reward = policy_reward.mean()

        clipped_value = sampled_value + \
            (value - sampled_value).clamp(min=-clip_range, max=clip_range)

        if make_step is True:
            self.optimizer.zero_grad()

        if self.psy_type == 'advantage':
            vf_loss = self.value_loss(value, total_rewards)
            vf_loss.backward()
            vf_loss = vf_loss.item()
        else:
            vf_loss = 0

        loss: torch.Tensor = - \
            (policy_reward * progress ** 2) / self.epochs

        if self.epochs > 1:
            loss.backward(retain_graph=True)
        else:
            loss.backward()

        torch.nn.utils.clip_grad_norm_([p for p in self.model.parameters(
        ) if p.name != 'value_head'], max_norm=self.max_grad_norm)

        if make_step:
            self.optimizer.step()

        return torch.tensor([loss.item(),
                             policy_reward.item(),
                             vf_loss,
                             value.mean().item()
                             ])

    @staticmethod
    def _normalize(adv: torch.FloatTensor):
        return (adv - adv.mean(dim=-1).mean(dim=-1)) / \
            (adv.std(dim=-1).std(dim=-1) + 1e-8)
