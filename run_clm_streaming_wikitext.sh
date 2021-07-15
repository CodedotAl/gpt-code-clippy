#! /bin/bash
./run_clm_streaming_flax_clean.py \
    --output_dir $HOME/tmp/gpt-neo-125M-test \
    --model_name_or_path="EleutherAI/gpt-neo-125M" \
    --dataset_name="wikitext" \
    --dataset_config_name="wikitext-103-raw-v1" \
    --text_column_name="text" \
    --do_train --do_eval \
    --block_size="128" \
    --per_device_train_batch_size="8" \
    --per_device_eval_batch_size="8" \
    --preprocessing_num_workers="8" \
    --learning_rate="6e-4" \
    --max_steps 500 \
    --warmup_steps 150 \
    --decay_steps 250 \
    --adam_beta1="0.9" \
    --adam_beta2="0.95" \
    --weight_decay="0.1" \
    --overwrite_output_dir \
    --logging_steps="10" \
    --eval_steps="50" \
    --push_to_hub="False" \
    --report_to="all" \
    --dtype="bfloat16" \
    --skip_memory_metrics="False" \
    --save_steps="50" \
    --save_total_limit 2 \
    --gradient_accumulation_steps 8 \
    --report_to="none" \
    --run_name="testing-mini" \
    --max_eval_samples 100 \
    --save_optimizer true \
    # --resume_from_checkpoint $HOME/gpt-neo-125M-test/ckpt-800 \
    # --adafactor \
    # --max_train_samples="10000" \
    
