#!/bin/bash
python inference_wait_input.py \
    --model_local_path ./checkpoints/qwen2vl7B \
    --model_finetune_path ./checkpoints/unitime \
    --output_dir ./results/interactive \
    --nf_short 128

