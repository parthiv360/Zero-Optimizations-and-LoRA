import os
import math
import argparse
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import wandb

from datasets import  load_dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    set_seed,
    PreTrainedTokenizer,
    DataCollatorForLanguageModeling,
    TrainerCallback,
)

# -----------------------------
# The whole project is focused on the unsupervised training of causal language models.
# We also provide materials for supervised fine tuning, check the https://huggingface.co/docs/trl/en/sft_trainer for more details.
# -----------------------------

# ------------------------------
# Useful tools: TrainerCallback, https://huggingface.co/docs/transformers/en/main_classes/callback
# ------------------------------

# -----------------------------
# Callback: save per-epoch loss + perplexity
# -----------------------------
class EpochMetricsWriterCallback(TrainerCallback):
    def __init__(self, output_dir: str, filename: str = "epoch_metrics.csv"):
        self.output_dir = output_dir
        self.filepath = os.path.join(output_dir, filename)
        self._last_train_loss = None
        self._header_written = os.path.exists(self.filepath) and os.path.getsize(self.filepath) > 0

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self._last_train_loss = float(logs["loss"])

    def on_epoch_end(self, args, state, control, **kwargs):
        control.should_evaluate = True
        return control

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics:
            return
        epoch = float(state.epoch) if state.epoch is not None else None
        eval_loss = metrics.get("eval_loss", None)

        ppl = None
        if eval_loss is not None:
            try:
                ppl = float(math.exp(min(20.0, float(eval_loss))))
            except OverflowError:
                ppl = float("inf")

        if not self._header_written:
            with open(self.filepath, "w", encoding="utf-8") as f:
                f.write("epoch,train_loss,eval_loss,perplexity\n")
            self._header_written = True

        with open(self.filepath, "a", encoding="utf-8") as f:
            f.write(f"{epoch},{self._last_train_loss},{eval_loss},{ppl}\n")

# -----------------------------
# Callback: log training throughput to wandb
# -----------------------------

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

class PerplexityCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            try:
                ppl = math.exp(logs["loss"])
            except (OverflowError, ValueError):
                ppl = float("inf")
            if wandb.run is not None:
                wandb.log({"train/perplexity": ppl}, step=state.global_step)
        
# -----------------------------
# Helpers
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Continue training a CausalLM with CLI args")

    # Model/tokenizer
    p.add_argument("--model_path", type=str, required=True, help="HF model name or local path")
    p.add_argument("--output_dir", type=str, required=True, help="Where to save checkpoints and final model")

    # Dataset options (pick one)
    p.add_argument("--dataset_path", type=str, default=None, help="Path to a dataset saved with datasets.save_to_disk()")
    p.add_argument("--hf_dataset", type=str, default=None, help='HF dataset name (e.g. "tatsu-lab/alpaca")')
    p.add_argument("--hf_dataset_split", type=str, default="train", help='Split name for hf_dataset (default: "train")')
    p.add_argument("--text_column", type=str, default=None, help="If using raw text dataset, column name with text")
    p.add_argument("--max_length", type=int, default=1024, help="Max sequence length for tokenization")

    # Training hyperparams
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--warmup_steps", type=int, default=0)
    p.add_argument("--lr_scheduler_type", type=str, default="cosine")

    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--per_device_eval_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)

    p.add_argument("--seed", type=int, default=42)

    # Precision + memory
    p.add_argument("--bf16", action="store_true", help="Use bf16 (recommended on Ampere+)")
    p.add_argument("--gradient_checkpointing", action="store_true")

    # Logging / saving
    p.add_argument("--logging_strategy", type=str, default="steps")
    p.add_argument("--save_strategy", type=str, default="epoch")
    p.add_argument("--save_only_model", action="store_true", help="Only save the model weights")
    p.add_argument("--resume", action="store_true", help="Resume from checkpoint if present")

    # For testing purposes
    p.add_argument("--test", action="store_true", help="Tiny run: train/eval on small subsets for quick testing")
    p.add_argument("--test_train_samples", type=int, default=10, help="Train samples to use when --test is set")
    p.add_argument("--test_eval_samples", type=int, default=10, help="Eval samples to use when --test is set")

    # Wandb
    # If you don't have Wandb account, register one and get your own api key
    # https://docs.wandb.ai/models/quickstart 
    p.add_argument("--report_to", type=str, default="wandb", help='Use "wandb" or "none"')
    p.add_argument("--wandb_project", type=str, default=None)
    p.add_argument("--wandb_run_name", type=str, default=None)
    p.add_argument("--wandb_api_key", type=str, default=None)

    # Eval split creation if dataset has no eval
    # Use for dataset which doesn't have the split. 
    p.add_argument("--eval_ratio", type=float, default=0.02, help="If no eval split exists, create one with this ratio")

    args = p.parse_args()

    if (args.dataset_path is None) == (args.hf_dataset is None):
        raise ValueError("Provide exactly one of --dataset_path or --hf_dataset")

    return args

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
            padding=True
        )
        tok["labels"] = tok["input_ids"].copy()
        return tok

    # Optional tiny test run in class
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

    # W&B env setup if requested
    if args.report_to == "wandb":
        if args.wandb_api_key:
            os.environ["WANDB_API_KEY"] = args.wandb_api_key
        if args.wandb_project:
            os.environ["WANDB_PROJECT"] = args.wandb_project
        os.environ["WANDB_WATCH"] = "all"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Load dataset (expects tokenized)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset (tokenize if needed)
    ds = load_hf_dataset_only(args, tokenizer=tokenizer)
    train_dataset = ds["train"]
    eval_dataset = ds["eval"]

    model = AutoModelForCausalLM.from_pretrained(args.model_path)
    model.resize_token_embeddings(len(tokenizer))

    # Prepare TrainingArguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        # gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=float(args.learning_rate),
        warmup_steps=args.warmup_steps,
        lr_scheduler_type=args.lr_scheduler_type,
        num_train_epochs=args.epochs,
        seed=args.seed,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        logging_strategy=args.logging_strategy,
        logging_steps=1,
        logging_first_step=True,
        save_strategy=args.save_strategy, # save every epoch or steps
        save_only_model=args.save_only_model,
        load_best_model_at_end=False,
        report_to=([] if args.report_to == "none" else [args.report_to]),
        run_name=args.wandb_run_name,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False), 
        callbacks=[
            EpochMetricsWriterCallback(output_dir=args.output_dir),
            ThroughputCallback(args.max_length),
            # PerplexityCallback()
        ],
    )

    # Resume logic (if checkpoints exist)
    resume_from_checkpoint = False
    if args.resume:
        import glob
        resume_from_checkpoint = len(glob.glob(os.path.join(args.output_dir, "checkpoint-*"))) > 0

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    trainer.save_model(output_dir=args.output_dir)

    print(f"Done. Model saved to: {args.output_dir}")
    print(f"Per-epoch metrics saved to: {os.path.join(args.output_dir, 'epoch_metrics.csv')}")


if __name__ == "__main__":
    main()
