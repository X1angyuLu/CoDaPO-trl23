#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1

model_list=("GD-ML/Qwen2.5-Math-7B-GPG"
 "/data0/xiangyu/saves/agrpo/Qwen2.5-Math-7B/LIMR/checkpoint-345"
 )
# for step in $(seq 30 30 360); do
#     model_list+=("/data0/xiangyu/saves/codapo/Qwen2.5-Math-7B/X-R1-7500-500/checkpoint-${step}")
# done

task_list=(
    "math"
    "aime24"
    "aime25"
    "amc23"
    "gsm8k"
    "minerva"
    "olympiadbench"
)

for model in "${model_list[@]}"; do
  for task in "${task_list[@]}"; do

    model_name=$(basename "$model")

    echo "Evaluating model: $model_name on task: $task"

    python eval/eval.py \
      --model_name_or_path "$model" \
      --data_name "$task" \
      --prompt_type "qwen-instruct" \
      --temperature 0.6 \
      --start_idx 0 \
      --end_idx -1 \
      --n_sampling 32 \
      --k 1 \
      --split "test" \
      --max_tokens 4096 \
      --seed 3372 \
      --top_p 1 \
      --output_dir "./eval/outputs" \
      --surround_with_messages

  done
done
