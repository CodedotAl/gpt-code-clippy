#! /bin/bash
./run_clm_apps.py \
    --output_dir /home/shared/models/gpt-code-clippy-1.3B-apps \
    --model_name_or_path EleutherAI/gpt-neo-1.3B \
    --dataset_name ./apps.py \
    --dataset_config_name formatted \
    --do_train --do_eval \
    --block_size="1024" \
    --per_device_train_batch_size="3" \
    --per_device_eval_batch_size="3" \
    --preprocessing_num_workers="16" \
    --learning_rate="2e-5" \
    --warmup_steps="5000" \
    --adam_beta1="0.9" \
    --adam_beta2="0.98" \
    --weight_decay="0.1" \
    --overwrite_output_dir \
    --num_train_epochs="5" \
    --logging_steps="50" \
    --eval_steps="2000" \
    --push_to_hub="False" \
    --report_to="wandb" \
    --dtype="bfloat16" \
    --skip_memory_metrics="False" \
    --save_steps="1000" \
    --save_strategy epoch \
    --save_total_limit 2 \
    --gradient_accumulation_steps 1 \
    --adafactor true \
    --all_data true \
    --seed 842 \
    # --resume_from_checkpoint $HOME/gpt-neo-125M-code-clippy/ckpt_201 \
    # --max_train_samples="10000" \
    # --max_eval_samples="1000"
