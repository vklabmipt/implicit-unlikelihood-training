import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
import torch.distributed as dist
import torch.multiprocessing as mp

import functools
from functools import partial

import argparse
import logging
import json
import os
import re
import gc
import random
from nltk import ngrams
import pickle

import numpy as np

from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW, get_linear_schedule_with_warmup, WEIGHTS_NAME, CONFIG_NAME, GPT2Config
from torch.utils.data import TensorDataset, SequentialSampler, DataLoader, RandomSampler

from collections import defaultdict, Counter
from tqdm import tqdm, trange
from pprint import pprint

from apex.parallel import DistributedDataParallel as DDP
from apex import amp

from fairseq.custom.metrics import TrainingMetrics, Metrics, ngram_metrics
from fairseq.custom.baseline_cross_entropy import CrossEntropyCriterionWCustomMetrics
from fairseq.custom.sequence_penalty_loss import SequencePenaltyCriterion
from fairseq.custom.evaluate_utils import batch_input_sequence_by_prefix_length

from utils import *

from unlikelihood import ul_seq
from alpha_entmax_training import alpha_entmax_loss
from tldr import tldr_loss

from time_reward_pg import TimeRewardTrainer
from policy_value import PolicyValueModel
from once_reward_pg import OnceRewardTrainer


torch.autograd.set_detect_anomaly(True)

RETOK = re.compile(r'\w+|[^\w\s]|\n', re.UNICODE)

logger = logging.getLogger(__name__)
logger.propagate = False
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO)
logger = logging.getLogger(__name__)


def random_seed(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def cleanup():
    dist.destroy_process_group()


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--world-size', type=int, default=1)
    parser.add_argument(
        '--opt-level',
        type=str,
        default='O0',
        choices=[
            'O0',
            'O1',
            'O2'])
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--start_rank', type=int, default=0)
    parser.add_argument('--adress', type=str, default='tcp://127.0.0.1:80')
    parser.add_argument(
        '--mode',
        choices=[
            'train',
            'eval-singletoken',
            'eval-completion',
            'eval-singletoken-sampling',
            'eval-acc',
            'eval-both',
            'eval-singletoken-argmax'],
        default='eval-singletoken')
    parser.add_argument('--eval-split', choices=['train', 'valid', 'test'])
    parser.add_argument(
        '--model-name',
        choices=[
            'gpt2',
            'gpt2-medium',
            'gpt2-large'],
        default='gpt2')
    parser.add_argument('--model-load-dir', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument(
        '--data-base',
        type=str,
        default='../fairseq/data-bin/wikitext-103-bpe_v0')
    parser.add_argument('--num-train-epochs', type=int, default=1)
    parser.add_argument('--batch-size-singletoken', type=int, default=1024)
    parser.add_argument('--batch-size-completion', type=int, default=1024)
    parser.add_argument(
        '--output-dir',
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.")

    # eval-completion
    parser.add_argument('--prefix-length', type=int, default=50)
    parser.add_argument('--continuation-length', type=int, default=100)
    parser.add_argument('--top-k', type=int, default=1)
    parser.add_argument('--top-p', type=float, default=0.0)

    # custom training
    parser.add_argument('--pg-tune-rate', type=float, default=0.0)
    parser.add_argument('--ul-tune-rate', type=float, default=0.0)
    parser.add_argument('--train-batch-size', type=int, default=300)
    parser.add_argument('--report-metrics-every', type=int, default=1)
    parser.add_argument('--save-every', type=int, default=50)
    parser.add_argument('--sequence-ngram-n', type=int, default=4)
    parser.add_argument('--train-n-steps', type=int, default=5000)
    parser.add_argument('--validate-every', type=int, default=50)
    parser.add_argument('--ul-accum-steps', type=int, default=1)
    parser.add_argument('--pg-accum-steps', type=int, default=1)
    parser.add_argument('--eval-compl-every', type=int, default=50)
    parser.add_argument('--compl-steps', type=int, default=float('inf'))
    parser.add_argument('--seq-max-grad-norm', type=int, default=10.0)
    parser.add_argument('--n-samples', type=int, default=1)
    parser.add_argument('--mini-batch-size', type=int, default=6)
    parser.add_argument('--report-value-loss-every', type=int, default=50)
    parser.add_argument('--load-start-iter', action='store_true')

    # policy gradient training
    parser.add_argument('--policy-top-k', type=int, default=1)
    parser.add_argument('--policy-top-p', type=float, default=0.0)
    parser.add_argument('--policy-temperature', type=float, default=1.0)
    parser.add_argument('--reward-ngram-n', type=int, default=4)
    parser.add_argument('--add_ul_loss_to_pg', action='store_true')
    parser.add_argument('--add_kl_loss_to_pg', action='store_true')
    parser.add_argument('--add_mle_loss_to_pg', action='store_true')
    parser.add_argument(
        '--algorithm',
        type=str,
        choices=[
            'once_reward_pg',
            'time_reward_pg'],
        default='once_reward_pg',
        required=False)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--lamda', type=float, default=0.85)
    parser.add_argument('--ppo-epoch', type=int, default=1)
    parser.add_argument('--reward-type', type=str, default='reward_for_past',
                        choices=['reward_for_past', 'reward_for_future'])
    parser.add_argument(
        '--psy-type',
        type=str,
        default='reward',
        choices=[
            'reward',
            'advantage'])
    parser.add_argument('--ratio-clip-range', type=float, default=0.2)
    parser.add_argument('--pg-coef', type=float, default=3.0)

    # training loop
    parser.add_argument("--adam-epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument('--max-grad-norm', type=int, default=1)
    parser.add_argument("--max-steps", default=-1, type=int,
                        help="If > 0: set total number of training \
                            steps to perform. Override num_train_epochs.")
    parser.add_argument('--gradient-accumulation-steps', type=int, default=1,
                        help="Number of updates steps to accumulate before\
                            performing a backward/update pass.")
    parser.add_argument('--learning-rate', type=float, default=6.25e-5)
    parser.add_argument("--warmup-steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--lr-schedule', type=str, default='warmup_linear')
    parser.add_argument('--weight-decay', type=float, default=0.01)

    parser.add_argument('--laplas-eps', type=float, default=1e-6)
    parser.add_argument('--alpha', type=float, default=1.2)
    parser.add_argument(
        '--token-loss',
        type=str,
        default='mle',
        choices=[
            'mle',
            'alpha_entmax_loss',
            'tldr'])
    parser.add_argument('--alpha-entmax', action='store_true')

    parser.add_argument('--num-beams', type=int, default=1)

    args = parser.parse_args()

    return args


def main(gpu, nprocs, args):
    random_seed(args.seed)
    print(f'GPU: {gpu}')
    args.gpu = gpu
    args.rank = args.start_rank + args.gpu
    print(f'Rank: {args.rank}')
    dist.init_process_group("nccl",
                            rank=args.rank,
                            init_method=args.adress,
                            world_size=args.world_size)
    group = dist.group.WORLD
    args.device = gpu
    n_gpu = torch.cuda.device_count()
    logger.info("gpu {}, n_gpu {}".format(gpu, n_gpu))
    torch.cuda.set_device(gpu)

    start_iter = 0

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    tokenizer = GPT2Tokenizer.from_pretrained(args.model_name)
    config = GPT2Config.from_pretrained(args.model_name)

    dataset_paths = {
        'train': os.path.join(args.data_base, 'train_tokens_bpe_gpt2.pt'),
        'valid': os.path.join(args.data_base, 'valid_tokens_bpe_gpt2.pt'),
        'test': os.path.join(args.data_base, 'test_tokens_bpe_gpt2.pt'),
    }

    gpt2 = GPT2LMHeadModel(config).from_pretrained(
        args.model_name, output_hidden_states=True)
    gpt2.config.n_embd = config.n_embd
    gpt2.config.eos_token_id = None

    if args.psy_type == 'reward':
        model = gpt2
    else:
        model = PolicyValueModel(config, gpt2)

    if args.model_load_dir:
        print('Load checkpoint')
        model.load_state_dict(
            torch.load(
                args.model_load_dir,
                map_location=torch.device(gpu)),
            strict=False)
        if args.load_start_iter:
            dir_path = args.model_load_dir[:-len('pytorch_model.bin')]
            if not os.path.exists(dir_path):
                raise ValueError(f'Path \'{dir_path}\' doesn\'t exists')
            if not os.path.isdir(dir_path):
                raise ValueError(f'Path \'{dir_path}\' is not a directory')

            names = list(
                filter(
                    lambda x: x.startswith('singletoken'),
                    os.listdir(dir_path)))
            start_iter = max(
                list(map(lambda x: int(x.split('_')[-1][:-len('.json')]), names)))
    model.cuda(gpu)

    if args.mode == 'eval-singletoken-argmax':
        eval_singletoken_argmax(model, args, dataset_paths, config)

    if args.mode == 'eval-acc':
        eval_acc(model, args, dataset_paths, config)

    if args.mode == 'eval-singletoken' or args.mode == 'eval-both':
        eval_singletoken(
            model,
            args,
            dataset_paths,
            config,
            top_k=args.top_k,
            top_p=args.top_p)

    if args.mode == 'eval-singletoken-sampling' or args.mode == 'eval-both':
        eval_singletoken(model, args, dataset_paths, config)

    if args.mode == 'eval-completion' or args.mode == 'eval-both':
        print('Eval-completion')
        eval_completion(
            model,
            tokenizer,
            args,
            dataset_paths,
            config,
            train_iter=None)

    if args.mode == 'train':
        if not os.path.exists(os.path.join(args.output_dir, 'best')):
            os.makedirs(os.path.join(args.output_dir, 'best'))

        if args.token_loss == 'mle':
            token_loss = mle_loss
        elif args.token_loss == 'alpha_entmax_loss':
            token_loss = alpha_entmax_loss
        elif args.token_loss == 'tldr':
            token_loss = tldr_loss
        else:
            raise ValueError('token loss is not defined')

        datasets = get_datasets(dataset_paths, max_len=args.train_batch_size)
        if args.world_size > 1:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                datasets['train'],
                num_replicas=args.world_size,
                rank=args.rank
            )
        else:
            train_sampler = RandomSampler(datasets['train'])
        train_seq_dataloader = DataLoader(
            datasets['train'], sampler=train_sampler, batch_size=1)

        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(
                nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in param_optimizer if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=args.learning_rate,
            eps=args.adam_epsilon)
        model, optimizer = amp.initialize(
            model, optimizer, opt_level=args.opt_level)
        model = DDP(model)
        model.config = config
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=args.train_n_steps)
        if args.algorithm == 'time_reward_pg':
            if args.policy_top_k != 0:
                def filter_method(x): return top_k_top_p_filtering(
                    x, n=1, top_k=args.policy_top_k, top_p=0.0)
            elif args.policy_top_p != 0.:
                def filter_method(x): return top_k_top_p_filtering(
                    x, n=1, top_k=0, top_p=args.policy_top_p)
            elif args.policy_temperature != 0.:
                def filter_method(x): return x / args.policy_temperature

            agent_policy = TimeRewardTrainer(
                model,
                optimizer,
                prefix_length=args.prefix_length,
                continuation_length=args.continuation_length,
                device=gpu,
                mini_batch_size=args.mini_batch_size,
                reward_type=args.reward_type,
                psy_type=args.psy_type,
                n_samples=args.n_samples,
                max_grad_norm=args.seq_max_grad_norm)

        if args.algorithm == 'once_reward_pg':
            agent_policy = OnceRewardTrainer(
                model,
                config,
                tokenizer,
                optimizer,
                prefix_length=args.prefix_length,
                continuation_length=args.continuation_length,
                n_samples=args.n_samples,
                device=gpu,
                top_k=args.policy_top_k,
                top_p=args.policy_top_p,
                temperature=args.policy_temperature,
                max_grad_norm=args.seq_max_grad_norm,
                ngram_reward=args.reward_ngram_n,
                mini_batch_size=args.mini_batch_size,
                psy_type=args.psy_type,
                clip_range=args.ratio_clip_range)

        total_steps = start_iter
        best_ppl = 1e20
        vf_loss = []
        avg_value = []
        for _ in trange(args.num_train_epochs, desc="Epoch",
                        initial=start_iter):
            logging_outputs = []
            epoch_loss = 0
            epoch_steps = 0
            tqdm_bar = tqdm(
                train_seq_dataloader,
                desc="Training",
                total=int(
                    args.train_n_steps *
                    (
                        (1 -
                         args.ul_tune_rate -
                         args.pg_tune_rate) +
                        args.ul_tune_rate *
                        args.ul_accum_steps +
                        args.pg_tune_rate *
                        args.pg_accum_steps)))

            ul_cnt = 0
            pg_cnt = 0
            for step, batch in enumerate(tqdm_bar):
                if ul_cnt % args.ul_accum_steps == 0 and pg_cnt % args.pg_accum_steps == 0:
                    model.zero_grad()
                gc.collect()

                # pg
                if torch.rand(1).item(
                ) < args.pg_tune_rate or pg_cnt % args.pg_accum_steps != 0:
                    if pg_cnt == 1:
                        rewards = []
                    if batch[0].size(1) < args.prefix_length:
                        continue
                    pg_cnt = (pg_cnt + 1) % args.pg_accum_steps

                    if args.algorithm == 'time_reward_pg':
                        samples, n_sent, batch_metrics = agent_policy.sample(
                            batch, filter_method, args)
                        logging_outputs.append(batch_metrics)
                        progress = max(0, (total_steps - 0) /
                                       (args.train_n_steps - 0))
                        clip_range = args.clip_range * (1 - progress)
                        train_info = agent_policy.train(
                            samples, filter_method, progress, args, make_step=(
                                pg_cnt %
                                args.pg_accum_steps == 0))
                        del samples
                        vf_loss.append(train_info[2].item())
                        avg_value.append(train_info[3].item())
                        loss = train_info[0]
                        loss = loss.item()

                    if args.algorithm == 'once_reward_pg':
                        samples, batch_metrics, batch_size = agent_policy.sample(
                            batch, args)
                        logging_outputs.append(batch_metrics)
                        train_info = agent_policy.ppo_step(
                            samples, args, make_step=(
                                pg_cnt %
                                args.pg_accum_steps == 0))
                        del samples
                        vf_loss.append(train_info[2].item())
                        avg_value.append(train_info[3].item())
                        loss = train_info[0]
                        loss = loss / args.pg_accum_steps
                        loss = loss.item()

                # ul
                elif torch.rand(
                        1).item() < args.ul_tune_rate or ul_cnt % args.ul_accum_steps != 0:
                    ul_cnt = (ul_cnt + 1) % args.ul_accum_steps
                    loss, batch_metrics = ul_seq(model, batch, args)
                    logging_outputs.append(batch_metrics)
                    loss.backward()
                    loss = loss.item()
                    if ul_cnt % args.ul_accum_steps == 0:
                        optimizer.step()
                        optimizer.zero_grad()

                # Token loss
                else:
                    loss = 0
                    loss, batch_metrics = token_loss(model, batch, args)
                    loss.backward()
                    loss = loss.item()
                    logging_outputs.append(batch_metrics)
                    optimizer.step()
                    optimizer.zero_grad()

                if ul_cnt % args.ul_accum_steps == 0 and pg_cnt % args.pg_accum_steps == 0:
                    scheduler.step()
                    epoch_loss += loss
                    epoch_steps += 1
                    total_steps += 1
                    tqdm_bar.desc = "Training loss: {:.2e} lr: {:.2e}".format(
                        epoch_loss / epoch_steps, scheduler.get_lr()[0])

                    if epoch_steps % args.report_value_loss_every == 0:
                        vf_loss_rep = np.mean(vf_loss)
                        avg_value_rep = np.mean(avg_value)
                        vf_loss = []
                        avg_value
                        with open(os.path.join(args.output_dir, f'vf_loss_{total_steps}.json'), 'w') as f:
                            json.dump({'vf_loss': vf_loss_rep,
                                       'avg_value': avg_value_rep}, f)

                    if epoch_steps % args.report_metrics_every == 0:
                        logging_average = CrossEntropyCriterionWCustomMetrics.aggregate_logging_outputs(
                            logging_outputs)
                        temp = SequencePenaltyCriterion.aggregate_logging_outputs(
                            logging_outputs)
                        for k, v in temp.items():
                            logging_average[k] = v
                        if args.token_loss == 'mle':
                            logging_average['ppl'] = 2 ** logging_average['loss']
                        elif args.token_loss == 'alpha_entmax_loss':
                            logging_average['e_ppl'] = np.exp(
                                np.mean([x['smoothed_nll_loss'] for x in logging_outputs]))
                            logging_average['js_div'] = np.mean(
                                [x['js_div'] for x in logging_outputs])
                        print(logging_average)
                        logging_outputs = []

                    if total_steps == args.train_n_steps:
                        break

                    if epoch_steps % args.save_every == 0 and args.rank == args.start_rank:
                        model_to_save = model.module if hasattr(
                            model, 'module') else model
                        output_model_file = os.path.join(
                            args.output_dir, WEIGHTS_NAME)
                        output_config_file = os.path.join(
                            args.output_dir, CONFIG_NAME)
                        torch.save(
                            model_to_save.state_dict(),
                            output_model_file)
                        model_to_save.config.to_json_file(output_config_file)
                        with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
                            json.dump(vars(args), f)
                        tokenizer.save_vocabulary(args.output_dir)

                    if total_steps % args.validate_every == 0:
                        print("Validating...")
                        validation_outputs = eval_singletoken(
                            model, args, dataset_paths, config, train_iter=total_steps)
                        if args.token_loss == 'mle':
                            metric = 'ppl'
                        elif args.token_loss == 'alpha_entmax_loss':
                            metric = 'js_div'
                        elif args.token_loss == 'tldr':
                            metric = 'ppl'
                        if validation_outputs[metric] < best_ppl and args.rank == args.start_rank:
                            best_ppl = validation_outputs[metric]
                            model_to_save = model.module if hasattr(
                                model, 'module') else model
                            output_model_file = os.path.join(
                                args.output_dir, 'best', WEIGHTS_NAME)
                            output_config_file = os.path.join(
                                args.output_dir, 'best', CONFIG_NAME)
                            torch.save(
                                model_to_save.state_dict(), output_model_file)
                            model_to_save.config.to_json_file(
                                output_config_file)
                            with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
                                json.dump(vars(args), f)
                            tokenizer.save_vocabulary(
                                os.path.join(args.output_dir, 'best'))
                            save_singletoken_metrics(
                                validation_outputs, config.to_dict(), args, train_iter=total_steps, best=True)

                    if total_steps % args.eval_compl_every == 0:
                        print("Eval completion...")
                        eval_completion(model,
                                        tokenizer,
                                        args,
                                        dataset_paths,
                                        config,
                                        train_iter=total_steps)
    cleanup()


if __name__ == '__main__':
    args = parse_args()
    ngpus_per_node = torch.cuda.device_count()
    torch.multiprocessing.spawn(
        main, nprocs=ngpus_per_node, args=(ngpus_per_node, args)
    )
