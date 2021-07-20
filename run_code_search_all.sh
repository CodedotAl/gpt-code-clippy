#! /bin/bash

./run_clm_flax.py \
    --output_dir $HOME/gpt-neo-125M-code-clippy-code-search-all \
    --model_name_or_path="EleutherAI/gpt-neo-125M" \
    --dataset_name code_search_net \
    --dataset_config_name="all" \
    --do_train --do_eval \
    --block_size="512" \
    --per_device_train_batch_size="32" \
    --per_device_eval_batch_size="64" \
    --preprocessing_num_workers="8" \
    --learning_rate="1.2e-4" \
    --num_train_epochs 20 \
    --warmup_steps 3000 \
    --adam_beta1="0.9" \
    --adam_beta2="0.95" \
    --weight_decay="0.1" \
    --overwrite_output_dir \
    --logging_steps="25" \
    --eval_steps="500" \
    --push_to_hub="False" \
    --report_to="all" \
    --skip_memory_metrics="True" \
    --save_steps="500" \
    --save_total_limit 10 \
    --report_to="wandb" \
    --run_name="gpt-neo-125M-code-clippy-code-search-all" \
    # --max_eval_samples 2000 \
    # --gradient_accumulation_steps 32 \
