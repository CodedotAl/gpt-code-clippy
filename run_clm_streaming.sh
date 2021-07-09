#! /bin/bash
./run_clm_streaming_flax.py \
    --output_dir $HOME/gpt-neo-125M-code-clippy \
    --model_name_or_path="EleutherAI/gpt-neo-125M" \
    --dataset_name $HOME/gpt-code-clippy/code_clippy.py \
    --data_dir $HOME/data \
    --text_column_name="text" \
    --do_train \
    --block_size="1024" \
    --per_device_train_batch_size="16" \
    --per_device_eval_batch_size="16" \
    --preprocessing_num_workers="8" \
    --learning_rate="3e-4" \
    --adafactor \
    --warmup_steps="50" \
    --adam_beta1="0.9" \
    --adam_beta2="0.98" \
    --weight_decay="0.01" \
    --overwrite_output_dir \
    --max_steps 1000 \
    --num_train_epochs="1" \
    --logging_steps="100" \
    --eval_steps="200" \
    --push_to_hub="False" \
    --report_to="all" \
    --dtype="bfloat16" \
    --skip_memory_metrics="False" \
    --save_steps="200" \
    --save_total_limit 2 \
    --gradient_accumulation_steps 1 \
    --report_to="none" \
    # --resume_from_checkpoint $HOME/gpt-neo-125M-code-clippy/ckpt_201 \
    # --max_train_samples="10000" \
    # --max_eval_samples="1000"
