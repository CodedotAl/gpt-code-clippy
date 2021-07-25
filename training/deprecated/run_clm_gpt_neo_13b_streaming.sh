#! /bin/bash
./run_clm_streaming_flax.py \
    --output_dir $HOME/gpt-neo-1.3B-code-clippy \
    --model_name_or_path="EleutherAI/gpt-neo-1.3B" \
    --dataset_name="code_search_net" \
    --dataset_config_name="python" \
    --text_column_name="func_code_string" \
    --do_train --do_eval \
    --block_size="2048" \
    --per_device_train_batch_size="1" \
    --per_device_eval_batch_size="1" \
    --preprocessing_num_workers="8" \
    --learning_rate="1e-4" \
    --warmup_steps="1000" \
    --adam_beta1="0.9" \
    --adam_beta2="0.98" \
    --weight_decay="0.01" \
    --overwrite_output_dir \
    --num_train_epochs="1" \
    --push_to_hub="False" \
    --dtype="bfloat16" \
    --adafactor \
    --skip_memory_metrics="False" \
    --gradient_accumulation_steps 2 \
    --report_to="none" \
    --run_name="test_13b" \
    --max_train_samples="10000" \
    --max_eval_samples="1000" \
    --save_total_limit 1 \
    # --resume_from_checkpoint="None" \
    