#!/bin/bash
export HF_HUB_OFFLINE=1
M=mlx-community/Qwen2.5-1.5B-Instruct-4bit
.venv/bin/python -u -m mlx_lm lora --model $M --train --data data/fault_full --adapter-path adapters/fault_full_15b \
  --iters 400 --batch-size 2 --learning-rate 2e-4 --mask-prompt --max-seq-length 1536 \
  --steps-per-report 50 --steps-per-eval 200 --save-every 400 > results/sft_fault_full_15b.log 2>&1
EVAL_MODEL=$M .venv/bin/python -u eval_llm.py fault_full_15b adapters/fault_full_15b 30 >> results/sft_fault_full_15b.log 2>&1
echo done > results/15b_done
