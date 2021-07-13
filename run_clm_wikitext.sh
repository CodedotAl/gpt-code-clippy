#! /bin/bash
./run_clm_flax.py \
    --output_dir $HOME/tmp/gpt-neo-125M-test-2 \
    --model_name_or_path="EleutherAI/gpt-neo-125M" \
    --dataset_name="wikitext" \
    --dataset_config_name="wikitext-2-raw-v1" \
    --text_column_name="text" \
    --do_train --do_eval \
    --block_size="128" \
    --per_device_train_batch_size="8" \
    --per_device_eval_batch_size="16" \
    --preprocessing_num_workers="8" \
    --learning_rate="2e-5" \
    --adafactor \
    --warmup_steps="100" \
    --adam_beta1="0.9" \
    --adam_beta2="0.98" \
    --weight_decay="0.01" \
    --overwrite_output_dir \
    --num_train_epochs="10" \
    --logging_steps="10" \
    --eval_steps="10" \
    --push_to_hub="False" \
    --report_to="none" \
    --run_name="test-non-streaming" \
    --dtype="bfloat16" \
    --skip_memory_metrics="False" \
    --save_steps="200" \
    --save_strategy epoch \
    --save_total_limit 2 \
    --gradient_accumulation_steps 8 \
    --save_optimizer true \
    --resume_from_checkpoint $HOME/tmp/gpt-neo-125M-test-2/ckpt-2591 \
    # --max_train_samples="10000" \
    # --max_eval_samples="1000"
