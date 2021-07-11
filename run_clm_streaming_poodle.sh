#! /bin/bash
./run_clm_streaming_flax_v2.py \
    --output_dir $HOME/gpt-neo-125M-code-clippy-test \
    --model_name_or_path="flax-community/gpt-neo-125M-code-clippy" \
    --dataset_name $HOME/gpt-code-clippy/code_clippy.py \
    --data_dir /home/shared/code_clippy_data \
    --text_column_name="text" \
    --do_train --do_eval \
    --block_size="2048" \
    --per_device_train_batch_size="8" \
    --per_device_eval_batch_size="16" \
    --preprocessing_num_workers="8" \
    --learning_rate="1e-4" \
    --max_steps 100000 \
    --warmup_steps 2500 \
    --decay_steps 25000 \
    --adam_beta1="0.9" \
    --adam_beta2="0.95" \
    --weight_decay="0.01" \
    --overwrite_output_dir \
    --logging_steps="100" \
    --eval_steps="500" \
    --push_to_hub="False" \
    --report_to="all" \
    --dtype="bfloat16" \
    --skip_memory_metrics="True" \
    --save_steps="500" \
    --save_total_limit 2 \
    --gradient_accumulation_steps 16 \
    --report_to="wandb" \
    --run_name="poodle-long-run" \
    --max_eval_samples 2000 \
    --save_optimizer true \
    # --adafactor \
    # --resume_from_checkpoint $HOME/gpt-neo-125M-code-clippy/ \
    # --max_train_samples="10000" \
    
