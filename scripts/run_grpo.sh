export PYTHONPATH=$(pwd)/src:$PYTHONPATH
export no_proxy=localhost,127.0.0.1
export NO_PROXY=localhost,127.0.0.1

CUDA_VISIBLE_DEVICES=2,4 ACCELERATE_LOG_LEVEL=info accelerate launch \
--main_process_port=29505 \
--config_file recipes/accelerate_configs/zero3.yaml \
--num_processes=2 train/grpo.py \
--config recipes/grpo_config.yaml \