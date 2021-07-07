#! /bin/bash
./run_clm_flax.py \
    --output_dir $HOME/gpt-neo-125M-code-clippy \
    --model_name_or_path="EleutherAI/gpt-neo-125M" \
    --dataset_name="code_search_net" \
    --dataset_config_name="python" \
    --text_column_name="func_code_string" \
    --do_train --do_eval \
    --block_size="1024" \
    --per_device_train_batch_size="16" \
    --per_device_eval_batch_size="16" \
    --preprocessing_num_workers="8" \
    --learning_rate="3e-4" \
    --adafactor \
    --warmup_steps="1000" \
    --adam_beta1="0.9" \
    --adam_beta2="0.98" \
    --weight_decay="0.01" \
    --overwrite_output_dir \
    --num_train_epochs="1" \
    --eval_steps="500" \
    --push_to_hub="False" \
    --dtype="bfloat16" \
    --skip_memory_metrics="False"
    # --max_train_samples="10000" \
    # --max_eval_samples="1000"
    