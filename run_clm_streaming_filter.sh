#! /bin/bash
./run_clm_streaming_filter_flax.py \
    --output_dir $HOME/gpt-code-clippy-lr1e-4-bs1024-f \
    --model_name_or_path EleutherAI/gpt-neo-125M \
    --dataset_name $HOME/gpt-code-clippy/code_clippy_filter.py \
    --data_dir /home/shared/code_clippy_dedup_data \
    --text_column_name="text" \
    --do_train --do_eval \
    --block_size="2048" \
    --per_device_train_batch_size="8" \
    --per_device_eval_batch_size="16" \
    --preprocessing_num_workers="8" \
    --learning_rate="1e-4" \
    --max_steps 100000 \
    --warmup_steps 4000 \
    --decay_steps 50000 \
    --adam_beta1="0.9" \
    --adam_beta2="0.95" \
    --weight_decay="0.1" \
    --overwrite_output_dir \
    --logging_steps="50" \
    --eval_steps="1000" \
    --push_to_hub="True" \
    --report_to="all" \
    --dtype="bfloat16" \
    --skip_memory_metrics="False" \
    --save_steps="1000" \
    --save_total_limit 5 \
    --gradient_accumulation_steps 16 \
    --report_to="wandb" \
    --run_name="testing" \
    --max_eval_samples 2000 \
    --save_optimizer true \
    # --adafactor \
    # --resume_from_checkpoint $HOME/gpt-neo-125M-code-clippy/ \
    # --max_train_samples="10000" \
    
