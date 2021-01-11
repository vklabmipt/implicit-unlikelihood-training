#!/bin/bash
python $1 \
    --output-dir $2 \
    --model-load-dir $2/pytorch_model.bin \
    --mode eval-singletoken \
    --batch-size-singletoken 1024 \
    --prefix-length 50 \
    --continuation-length 100 \
    --device $3 \
    --algorithm $4 \
    --adress $5 \
    --model-name $6 \
    --top-k $7 \
    --top-p $8 \
    --laplas-eps $9 \
    --eval-split ${10}