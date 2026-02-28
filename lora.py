# main.py
import os
import argparse
import time
from datasets import load_dataset, DatasetDict
import wandb
import math

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    set_seed,
    PreTrainedTokenizer,
    DataCollatorForLanguageModeling,
    TrainerCallback

)

from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
)
class ThroughputCallback(TrainerCallback):
    def __init__(self,max_length):
        self.last_time = None
        self.last_step = None
        self.max_length = max_length

    def on_train_begin(self, args, state, control, **kwargs):
        self.last_time = time.perf_counter()
        self.last_step = state.global_step
    


    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or not state.is_world_process_zero:
            return

        now = time.perf_counter()
        dt = now - self.last_time
        dstep = state.global_step - self.last_step

        if dt > 0 and dstep > 0:
            # samples_per_sec measures training throughput.
            # dstep: number of optimizer (training) steps completed during the time window dt
            # eff_batch: effective global batch size per step, accounting for:
            #   - per-device batch size
            #   - gradient accumulation steps
            #   - number of distributed processes (world_size)
            # Total samples processed = dstep * eff_batch
            # Throughput (samples/sec) = (dstep * eff_batch) / dt
            # ADDITIONAL: add your code to log the metric of tokens_per_sec

            world_size = getattr(state, "num_processes", 1) or 1
            eff_batch = (
                args.per_device_train_batch_size
                * args.gradient_accumulation_steps
                * world_size
            )
            logs["train/samples_per_sec"] = (dstep * eff_batch) / dt
            sequence_length = self.max_length
            if sequence_length:
                logs["train/tokens_per_sec"] = ((dstep * eff_batch) / dt)*sequence_length

        self.last_time = now
        self.last_step = state.global_step

def parse_args():
    p = argparse.ArgumentParser(description="Train LoRA adapters for a causal LM on an HF dataset.")

    # Model / data
    p.add_argument("--model_path", type=str, required=True, help="HF model name or local path")
    p.add_argument("--hf_dataset", type=str, required=True, help="HF dataset name, e.g. roneneldan/TinyStories")
    p.add_argument("--hf_dataset_split", type=str, default="train", help="Dataset split, e.g. train")
    p.add_argument("--text_column", type=str, default="text", help="Which column contains text")
    p.add_argument("--max_length", type=int, default=1024, help="Max sequence length")
    p.add_argument("--output_dir", type=str, required=True, help="Where to save outputs")

    # Train hyperparams
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)

    # Eval hyperparams
    p.add_argument("--per_device_eval_batch_size", type=int, default=1, help="Batch size for evaluation. Keep at 1 to avoid padding issues.")

    # W&B (optional)
    p.add_argument("--wandb_api_key", type=str, default=None, help="Prefer env var WANDB_API_KEY instead")
    p.add_argument("--wandb_project", type=str, default=None)
    p.add_argument("--wandb_run_name", type=str, default=None)

    # LoRA hyperparameters
    p.add_argument("--use_lora", action="store_true", help="Enable LoRA training")
    p.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    p.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    p.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    p.add_argument(
        "--lora_target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated module names to apply LoRA to",
    )
    p.add_argument("--lora_bias", type=str, default="none", choices=["none", "all", "lora_only"])
    p.add_argument("--lora_task_type", type=str, default="CAUSAL_LM", choices=["CAUSAL_LM"])

    # "test mode" like your comment: only use 10 samples
    p.add_argument("--test", action="store_true", help="Use only 10 samples for quick testing")
    p.add_argument("--test_train_samples", type=int, default=10, help="Train samples to use when --test is set")
    p.add_argument("--test_eval_samples", type=int, default=10, help="Eval samples to use when --test is set")
    return p.parse_args()


def load_hf_dataset_only(args, tokenizer: PreTrainedTokenizer) -> DatasetDict:
    """
    Loads ONLY a Hugging Face dataset (load_dataset) and returns DatasetDict with train/eval.
    Always tokenizes raw text into input_ids/attention_mask/labels for Causal LM.
    """

    if args.hf_dataset is None:
        raise ValueError("This version only supports Hugging Face datasets. Provide --hf_dataset.")

    # Load split the user requested (commonly "train")
    train_ds = load_dataset(args.hf_dataset, split=args.hf_dataset_split)

    # Try to find a good eval split from the dataset builder if user loaded "train"
    # NOTE: load_dataset(..., split=...) returns a Dataset, not DatasetDict,
    # so we need a second call if we want validation/test.
    eval_ds = None
    for candidate in ["validation", "eval", "test"]:
        try:
            eval_ds = load_dataset(args.hf_dataset, split=candidate)
            break
        except Exception:
            eval_ds = None

    if eval_ds is None:
        # Create an eval split from train
        split = train_ds.train_test_split(test_size=args.eval_ratio, seed=args.seed)
        ds = DatasetDict({"train": split["train"], "eval": split["test"]})
    else:
        ds = DatasetDict({"train": train_ds, "eval": eval_ds})

    # Determine text column
    train_cols = set(ds["train"].column_names)
    text_col = args.text_column
    if text_col is None:
        candidate_cols = ["text", "content", "prompt", "completion", "instruction", "input", "output"]
        for c in candidate_cols:
            if c in train_cols:
                text_col = c
                break
        if text_col is None:
            # fallback to first column
            text_col = ds["train"].column_names[0]

    if text_col not in train_cols:
        raise ValueError(f"--text_column '{text_col}' not found. Available: {sorted(train_cols)}")

    def tokenize_fn(batch):
        texts = batch[text_col]
        tok = tokenizer(
            texts,
            truncation=True,
            max_length=args.max_length,
            padding=False,  # collator pads dynamically
        )
        tok["labels"] = tok["input_ids"].copy()
        return tok

    # Optional tiny test run
    if args.test:
        ds["train"] = ds["train"].select(range(min(args.test_train_samples, len(ds["train"]))))
        ds["eval"]  = ds["eval"].select(range(min(args.test_eval_samples, len(ds["eval"]))))

    required_cols = {"input_ids", "attention_mask", "labels"}

    # Tokenize and drop other columns
    ds = ds.map(
        tokenize_fn,
        batched=True,
        remove_columns=[c for c in ds["train"].column_names if c not in required_cols],
        desc=f"Tokenizing from column '{text_col}'",
    )

    # Sanity check
    missing = required_cols - set(ds["train"].column_names)
    if missing:
        raise ValueError(f"Tokenization failed; missing columns: {missing}")

    return ds

def main():
    args = parse_args()
    set_seed(args.seed)

    # Load tokenizer + model
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    ds = load_hf_dataset_only(args, tokenizer=tokenizer)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map="auto",
    )

    # Apply LoRA
    if args.use_lora:
        target_modules = [m.strip() for m in args.lora_target_modules.split(",") if m.strip()]
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
            bias=args.lora_bias,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    else:
        print("WARNING: --use_lora not set. This will full-finetune the base model (expensive).")

    # W&B env setup if requested
    if args.wandb_api_key:
        if args.wandb_api_key:
            os.environ["WANDB_API_KEY"] = args.wandb_api_key
        if args.wandb_project:
            os.environ["WANDB_PROJECT"] = args.wandb_project
        os.environ["WANDB_WATCH"] = "all"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        report_to="wandb",
        run_name=args.wandb_run_name,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["eval"],
        tokenizer=tokenizer,
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        callbacks = [
            ThroughputCallback(args.max_length)
        ]
    )

    trainer.train()

    # Save:
    # - If LoRA: saves adapter weights/config into output_dir (recommended)
    # - If full finetune: saves full model
    trainer.save_model(args.output_dir)
    print("Running final evaluation...")
    metrics = trainer.evaluate()  # This runs the loop on ds['eval']
    print(metrics)
    
    # If using WandB, this manually logs the final eval results
    if wandb.run is not None:
        wandb.log(metrics)
        
    try:
        perplexity = math.exp(metrics["eval_loss"])
        print(f"Final Perplexity: {perplexity}")
        if wandb.run is not None:
            wandb.log({"eval/perplexity": perplexity})
    except:
        print("Could not calculate perplexity.")
    print(f"Done. Saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
