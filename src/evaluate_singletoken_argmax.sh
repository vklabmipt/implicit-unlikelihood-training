#!/bin/bash
python $1 \
    --output-dir $2 \
    --model-load-dir $2/pytorch_model.bin \
    --mode eval-singletoken-argmax \
    --batch-size-singletoken 1024 \
    --device $3 \
    --algorithm $4 \
    --adress $5 \
    --model-name $6 \
    --eval-split ${7}