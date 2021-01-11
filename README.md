# Implicit Unlikelihood Training: Improving Neural Text Generation with Reinforcement Learning

This repository contains the code needed for running the experiments from the paper: 

Implicit Unlikelihood Training: Improving Neural Text Generation with Reinforcement Learning
(under review at COLING)


| Table of Contents                     |
| ------------------------------------- |
| [Setup](#setup)                       |
| [Finetuning GPT-2](#finetuning-gpt-2) |
| [Evaluation](#evaluation)             |

## Setup

#### Building a Docker container

```
cd docker
docker build -t iut .
```

Note that wikitext-103 dataset (around 160Mb) will be automatically downloaded.

#### Running a container

```
docker run --gpus '"device=all"' --runtime=nvidia\
    -v <PATH_TO_SRC>:/app/src --net=host\
    -i -t iut /bin/bash
```

```
cd src
```

A hack is needed: comment out tf.logging usage from fairseq/fairseq/custom/evaluation.py:18 should you get an error.

## Finetuning GPT-2

#### MLE:

```bash
python run_gpt2.py \
    --output-dir ./checkpoint/gpt2/MLE \
    --eval-split valid \
    --mode train \
    --save-every 50 \
    --validate-every 50 \
    --eval-compl-every 500 \
    --device 0 \
    --adress tcp://127.0.0.1:80 \
    --opt-level O0 \
    --seed 42
```

#### Unlikelihood Training:

```bash
python run_gpt2.py \
    --output-dir ./checkpoint/gpt2/UT \
    --eval-split valid \
    --mode train \
    --save-every 50 \
    --validate-every 50 \
    --eval-compl-every 500 \
    --ul-tune-rate 0.5 \
    --device 0 \
    --adress tcp://127.0.0.1:80 \
    --opt-level O0 \
    --seed 42
```


#### Implicit Unlikelihood Training:

```bash
python run_gpt2.py \
    --output-dir ./checkpoint/gpt2/iUT \
    --eval-split valid \
    --mode train \
    --save-every 50 \
    --validate-every 50 \
    --eval-compl-every 500 \
    --pg-tune-rate 0.25 \
    --ul-tune-rate 0.25 \
    --device 0 \
    --adress tcp://127.0.0.1:80 \
    --opt-level O0 \
    --seed 42
```

#### Alpha-entmax training:

```bash
python run_gpt2.py \
    --output-dir ./checkpoint/gpt2/alpha_entmax \
    --eval-split valid \
    --mode train \
    --save-every 50 \
    --validate-every 50 \
    --eval-compl-every 500 \
    --device 0 \
    --adress tcp://127.0.0.1:80 \
    --opt-level O0 \
    --token-loss alpha_entmax_loss \
    --alpha-entmax \
    --seed 42
```


#### Running TLDR:

```bash
python run_gpt2.py \
    --output-dir ./checkpoint/gpt2/TLDR \
    --eval-split valid \
    --mode train \
    --save-every 50 \
    --validate-every 50 \
    --eval-compl-every 500 \
    --device 0 \
    --adress tcp://127.0.0.1:80 \
    --opt-level O0 \
    --token-loss tldr \
    --seed 42
```

## Running evaluation

```bash
python run_evaluation.py \
    --path_to_script run_gpt2_apex.py \
    --checkpoint_folder name_of_checkpoint/best \
    --device 0 \
    --adress tcp://127.0.0.1:80 \
    --eval_mode all \
    --eval_split valid
```

