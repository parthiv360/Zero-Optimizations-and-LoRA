BASE=/local/venv/hf_cache/hub/models--google--gemma-2-2b
REV=$(cat $BASE/refs/main)
MODEL_PATH=$BASE/snapshots/$REV
export WANDB_MODE=offline
CUDA_VISIBLE_DEVICES=0 python /local/parthiv.sarkar/mini2_code/main.py --model_path "$MODEL_PATH" \
  --hf_dataset roneneldan/TinyStories \
  --hf_dataset_split train \
  --text_column text \
  --max_length 1024 \
  --output_dir /local/parthiv.sarkar/mini2_code/saved/baseline \
  --epochs 1 \
  --learning_rate 2e-5 \
  --per_device_train_batch_size 1 \
  --wandb_project mini_project_train_inclass \
  --wandb_run_name baseline \
  --bf16 \
  --save_only_model \
  --test \
  --test_train_samples 200 \
  --test_eval_samples 100
