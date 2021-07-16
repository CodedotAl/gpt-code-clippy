#! /bin/bash
./run_clm_streaming_flax.py \
    --output_dir $HOME/gpt-code-clippy-125M-2048 \
    --model_name_or_path flax-community/gpt-neo-125M-code-clippy \
    --dataset_name $HOME/gpt-code-clippy/code_clippy.py \
    --data_dir /home/shared/code-clippy-dataset/merged-data \
    --text_column_name="text" \
    --do_train --do_eval \
    --block_size="2048" \
    --per_device_train_batch_size="8" \
    --per_device_eval_batch_size="16" \
    --preprocessing_num_workers="8" \
    --learning_rate="1e-4" \
    --max_steps 100000 \
    --warmup_steps 2000 \
    --decay_steps 30000 \
    --adam_beta1="0.9" \
    --adam_beta2="0.95" \
    --weight_decay="0.1" \
    --overwrite_output_dir \
    --logging_steps 25 \
    --eval_steps 500 \
    --push_to_hub="False" \
    --dtype="bfloat16" \
    --skip_memory_metrics="True" \
    --save_steps 500 \
    --save_total_limit 4 \
    --gradient_accumulation_steps 32 \
    --report_to="wandb" \
    --run_name="gpt-code-clippy-125m-1e-4-2048" \
    --max_eval_samples 2000 \
    --save_optimizer true \
    # --adafactor \
    # --resume_from_checkpoint $HOME/gpt-neo-125M-test \
    # --max_train_samples="10000" \
    
