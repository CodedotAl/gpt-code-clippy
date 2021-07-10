#! /bin/bash
./run_clm_streaming_flax_v2.py \
    --output_dir $HOME/gpt-neo-125M-code-clippy \
    --model_name_or_path="EleutherAI/gpt-neo-125M" \
    --dataset_name $HOME/gpt-code-clippy/code_clippy.py \
    --data_dir /home/shared/code-clippy-dataset/merged-data \
    --text_column_name="text" \
    --do_train --do_eval \
    --block_size="2048" \
    --per_device_train_batch_size="16" \
    --per_device_eval_batch_size="16" \
    --preprocessing_num_workers="8" \
    --learning_rate="3e-4" \
    --adafactor \
    --max_steps 100000 \
    --warmup_steps 3000 \
    --decay_steps 30000 \
    --adam_beta1="0.9" \
    --adam_beta2="0.95" \
    --weight_decay="0.1" \
    --overwrite_output_dir \
    --num_train_epochs="1" \
    --logging_steps="50" \
    --eval_steps="500" \
    --push_to_hub="False" \
    --report_to="all" \
    --dtype="bfloat16" \
    --skip_memory_metrics="False" \
    --save_steps="500" \
    --save_total_limit 2 \
    --gradient_accumulation_steps 4 \
    --report_to="wandb" \
    --max_eval_samples="1000" \
    --save_optimizer = True \
    # --resume_from_checkpoint $HOME/gpt-neo-125M-code-clippy/ckpt_201 \
    # --max_train_samples="10000" \
    
