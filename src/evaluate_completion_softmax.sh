#!/bin/bash
python $1  \
    --output-dir $2 \
    --model-load-dir $2/pytorch_model.bin \
    --eval-split valid \
    --mode eval-completion \
    --batch-size-completion 1024 \
    --prefix-length 50 \
    --continuation-length 100 \
    --device $3 \
    --top-k $4 \
    --top-p $5 \
    --algorithm $6 \
    --adress $7 \
    --model-name $8 \
    --eval-split $9