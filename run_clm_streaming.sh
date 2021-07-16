#! /bin/bash
./run_clm_streaming_flax_v2.py \
    --output_dir $HOME/gpt-neo-13B-test \
    --model_name_or_path EleutherAI/gpt-neo-1.3B \
    --dataset_name $HOME/gpt-code-clippy/code_clippy.py \
    --data_dir /home/arto/exdata/merged-data \
    --text_column_name="text" \
    --do_train --do_eval \
    --block_size="1024" \
    --per_device_train_batch_size="1" \
    --per_device_eval_batch_size="2" \
    --preprocessing_num_workers="8" \
    --learning_rate="1e-4" \
    --max_steps 500 \
    --warmup_steps 150 \
    --decay_steps 250 \
    --adam_beta1="0.9" \
    --adam_beta2="0.95" \
    --weight_decay="0.01" \
    --overwrite_output_dir \
    --logging_steps="10" \
    --eval_steps="50" \
    --push_to_hub="True" \
    --report_to="all" \
    --dtype="bfloat16" \
    --skip_memory_metrics="False" \
    --save_steps="50" \
    --save_total_limit 2 \
    --gradient_accumulation_steps 1 \
    --report_to="wandb" \
    --run_name="testing" \
    --max_eval_samples 100 \
    --save_optimizer true \
    # --adafactor \
    # --resume_from_checkpoint $HOME/gpt-neo-125M-code-clippy/ \
    # --max_train_samples="10000" \
    
