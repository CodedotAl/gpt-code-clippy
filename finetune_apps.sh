#! /bin/bash
./run_clm_apps.py \
    --output_dir $HOME/gpt-code-clippy-apps-125m-2048-raw \
    --model_name_or_path $HOME/gpt-code-clippy-125M-bs2048-raw \
    --dataset_name ./apps.py \
    --do_train --do_eval \
    --block_size="1024" \
    --per_device_train_batch_size="16" \
    --per_device_eval_batch_size="16" \
    --preprocessing_num_workers="16" \
    --learning_rate="5e-5" \
    --warmup_steps="800" \
    --adam_beta1="0.9" \
    --adam_beta2="0.98" \
    --weight_decay="0.1" \
    --overwrite_output_dir \
    --num_train_epochs="5" \
    --logging_steps="20" \
    --eval_steps="100" \
    --push_to_hub="False" \
    --report_to="wandb" \
    --dtype="bfloat16" \
    --skip_memory_metrics="False" \
    --save_steps="100" \
    --save_strategy epoch \
    --save_total_limit 5 \
    --gradient_accumulation_steps 2 \
    --adafactor \
    # --resume_from_checkpoint $HOME/gpt-neo-125M-code-clippy/ckpt_201 \
    # --max_train_samples="10000" \
    # --max_eval_samples="1000"
