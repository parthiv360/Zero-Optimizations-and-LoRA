# ================================================================
# Accelerate vs DeepSpeed (how they work together)
#
# - Accelerate is the *launcher / orchestrator*:
#   * Spawns multiple processes (one per GPU by default)
#   * Sets up torch.distributed (rank, world size, master addr/port)
#   * Chooses the distributed backend based on the config file
#     (DDP, FSDP, or DeepSpeed)
#   * Lets the training code stay mostly framework-agnostic
#
# - DeepSpeed is the *runtime engine*:
#   * Activated because the accelerate config sets:
#       distributed_type: DEEPSPEED
#   * Handles memory and performance optimizations:
#       - ZeRO stages (1/2/3)
#       - Optimizer state sharding
#       - Gradient partitioning
#       - Optional CPU/NVMe offload
#
# - Important:
#   * You DO NOT call deepspeed.initialize() in your Python code
#   * You only use Accelerate APIs (Accelerator(), prepare(), etc.)
#   * Accelerate automatically wires DeepSpeed under the hood
#
# - In this command:
#   * --config_file ds_z1.yaml → selects DeepSpeed + ZeRO-1
#   * --num_processes 2        → run on 2 GPUs (2 ranks)
#   * --main_process_port 3000 → port for distributed initialization
#
# DeepSpeed "stages" (ZeRO) are therefore controlled entirely by
# the accelerate config file, not by changes in main.py.
# ================================================================

BASE=/local/venv/hf_cache/hub/models--google--gemma-2-2b
REV=$(cat $BASE/refs/main)
MODEL_PATH=$BASE/snapshots/$REV
export WANDB_MODE=offline
accelerate launch --config_file="/local/parthiv.sarkar/mini2_code/config/ds_z2.yaml" \
    --num_processes 2 \
    --main_process_port 31000 \
    /local/mini2_code/main.py --model_path "$MODEL_PATH" \
    --hf_dataset roneneldan/TinyStories \
    --hf_dataset_split train \
    --text_column text \
    --max_length 2048 \
    --output_dir /local/parthiv.sarkar/mini2_code/saved/zero_2 \
    --epochs 1 \
    --learning_rate 2e-5 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --wandb_project mini_project_train_inclass \
    --wandb_run_name zero_2 \
    --bf16 \
    --save_only_model \
    --gradient_checkpointing \
    --test \
    --test_train_samples 200 \
    --test_eval_samples 100
