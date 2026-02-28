BASE=/local/venv/hf_cache/hub/models--google--gemma-2-2b
REV=$(cat $BASE/refs/main)
MODEL_PATH=$BASE/snapshots/$REV
export WANDB_MODE=offline

CUDA_VISIBLE_DEVICES=0 python /local/parthiv.sarkar/mini2_code/lora.py \
  --model_path "$MODEL_PATH" \
  --hf_dataset roneneldan/TinyStories \
  --hf_dataset_split train \
  --text_column text \
  --max_length 1024 \
  --output_dir /local/parthiv.sarkar/mini2_code/saved/lora \
  --epochs 1 \
  --learning_rate 2e-5 \
  --per_device_train_batch_size 1 \
  --wandb_project mini_project_train_inclass \
  --wandb_run_name lora \
  --use_lora \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --lora_target_modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" \
  --test \
  --test_train_samples 200 \
  --test_eval_samples 100
