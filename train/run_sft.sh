#!/bin/bash
export HF_HUB_OFFLINE=1
M=mlx-community/Qwen2.5-0.5B-Instruct-4bit
for arm in clean_verify clean_noverify; do
  .venv/bin/python -m mlx_lm lora --model $M --train --data data/$arm --adapter-path adapters/$arm \
    --iters 500 --batch-size 4 --learning-rate 2e-4 --mask-prompt --max-seq-length 1536 \
    --steps-per-report 50 --steps-per-eval 250 --save-every 500 > results/sft_$arm.log 2>&1
done
echo done > results/sft_done
