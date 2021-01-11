#!/bin/bash
python $1 \
    --output-dir $2 \
    --model-load-dir $2/pytorch_model.bin \
    --eval-split valid \
    --mode eval-singletoken \
    --batch-size-singletoken 1024 \
    --prefix-length 50 \
    --continuation-length 100 \
    --device $3 \
    --algorithm $4 \
    --adress $5 \
    --model-name $6 \
    --alpha-entmax \
    --alpha $7 \
    --laplas-eps $8 \
    --eval-split $9