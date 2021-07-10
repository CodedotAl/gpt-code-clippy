#! /bin/bash
./run_clm_streaming_flax_v2.py \
    --output_dir $HOME/gpt-neo-125M-code-clippy \
    --model_name_or_path="EleutherAI/gpt-neo-125M" \
    --dataset_name $HOME/gpt-code-clippy/code_clippy.py \
    --data_dir /home/shared/code-clippy-dataset/merged-data \
    --text_column_name="text" \
    --do_train --do_eval \
    --block_size="2048" \
    --per_device_train_batch_size="8" \
    --per_device_eval_batch_size="16" \
    --preprocessing_num_workers="8" \
    --learning_rate="6e-4" \
    --adafactor \
    --max_steps 10000 \
    --warmup_steps 3000 \
    --decay_steps 5000 \
    --adam_beta1="0.9" \
    --adam_beta2="0.95" \
    --weight_decay="0.01" \
    --overwrite_output_dir \
    --num_train_epochs="1" \
    --logging_steps="50" \
    --eval_steps="50" \
    --push_to_hub="False" \
    --report_to="all" \
    --dtype="bfloat16" \
    --skip_memory_metrics="False" \
    --save_steps="50" \
    --save_total_limit 2 \
    --gradient_accumulation_steps 1 \
    --report_to="wandb" \
    --run_name="testing" \
    --max_eval_samples 10 \
    --save_optimizer true \
    --resume_from_checkpoint $HOME/gpt-neo-125M-code-clippy/ \
    # --max_train_samples="10000" \
    
