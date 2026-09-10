import math
import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # Must be set before torch is imported.
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"  # Limits fragmentation.

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)


MODEL_NAME = "Qwen/Qwen3-1.7B-Base"  # Use Qwen/Qwen3-0.6B-Base on 8GB machines.
DATA_PATH = "dataset.clean.jsonl"
OUTPUT_DIR = "./Amazon-title-gen"
MAX_LENGTH =  2048 # Lower to 1024 or 512 if receiving OOM error; reduces the dataset a lot.
DISABLE_MPS = False  # Set True to fall back to CPU on macOS.


# Picks the device to train on, plus the number format and batch size to use with it.
def device_settings():
    if torch.cuda.is_available():
        # Ampere (capability 8) and newer do bf16 natively, older ones use fp16.
        major = torch.cuda.get_device_capability()[0]
        dtype = torch.bfloat16 if major >= 8 else torch.float16
        return "cuda", dtype, 2, 8
    # macOS GPU and CPU: full precision, smaller batch, more accumulation.
    device = "mps" if torch.backends.mps.is_available() and not DISABLE_MPS else "cpu"
    return device, torch.float32, 1, 16


device, dtype, batch_size, grad_accum = device_settings()
print(f"Device: {device} | dtype: {dtype} | batch {batch_size} x {grad_accum} accum")

dataset = load_dataset("json", data_files=DATA_PATH, split="train")
dataset = dataset.train_test_split(test_size=0.05, seed=42)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token


# Builds one token sequence per row: the product text, then the title it should produce.
def tokenize_function(examples):
    input_ids_list = []
    labels_list = []

    for product, title in zip(examples["input"], examples["output"]):
        product_text = product + "\n"  # Separates the input text from the title.
        product_tokens = tokenizer(product_text, add_special_tokens=False)["input_ids"]
        title_tokens = tokenizer(title, add_special_tokens=False)["input_ids"]
        title_tokens += [tokenizer.eos_token_id]

        input_ids = product_tokens + title_tokens
        # The -100 entries mask the product text, so only the title is trained on.
        labels = [-100] * len(product_tokens) + title_tokens

        input_ids_list.append(input_ids)
        labels_list.append(labels)

    return {
        "input_ids": input_ids_list,
        "labels": labels_list
    }


# Tokenizes a split and drops rows longer than MAX_LENGTH.
def tokenize_split(split):
    return split.map(
        tokenize_function,
        batched=True,
        remove_columns=split.column_names
    ).filter(lambda x: len(x["input_ids"]) <= MAX_LENGTH)

train_dataset = tokenize_split(dataset["train"]).shuffle(seed=42)
eval_dataset = tokenize_split(dataset["test"])

print(f"Samples: {len(dataset['train'])} → {len(train_dataset)} (after filtering)")
print(f"Samples: {len(dataset['test'])} → {len(eval_dataset)} (after filtering)")

# Optional: Use a smaller subset for a quick test of the flow.
# train_dataset = train_dataset.select(range(100))
# eval_dataset = eval_dataset.select(range(10))

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=dtype)
model.config.pad_token_id = tokenizer.pad_token_id

lora_config = LoraConfig(
    task_type="CAUSAL_LM",
    r=32,  # Adapter size. Raise if underfitting, lower to save memory.
    lora_alpha=64,  # Adapter strength, keep at twice r.
    lora_dropout=0.05,  # Raise toward 0.1 if overfitting.
    bias="none",  # Train the adapters only, leave bias terms frozen.
    # The layers LoRA adapts. Covering attention and MLP beats attention alone.
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ]
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Configuration settings and hyperparameters for fine-tuning.
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    use_cpu=(device == "cpu"),
    bf16=(dtype == torch.bfloat16),
    fp16=(dtype == torch.float16),

    # Training duration.
    num_train_epochs=3,  # Drop to 2 if eval loss turns up before the end.

    # Batch size and memory management.
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    gradient_accumulation_steps=grad_accum,  # Effective batch = batch x accum.
    gradient_checkpointing=True,  # Saves a lot of memory, ~20-30% slower.
    gradient_checkpointing_kwargs={"use_reentrant": False},  # Required with LoRA.

    # Use 'group_by_length' instead when transformers<= 5.0.0
    train_sampling_strategy=True,  # Batches similar lengths together, less padding.

    # Learning rate and regularization.
    learning_rate=2e-4,  # Raise if underfitting, lower if the loss is unstable.
    weight_decay=0.01,  # Raise toward 0.1 if overfitting.
    max_grad_norm=1.0,
    warmup_steps=0.05,  # A float is a ratio; 5-10% is the usual range.
    lr_scheduler_type="cosine",

    # Evaluation and checkpointing.
    eval_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    save_steps=50,
    save_total_limit=3,  # Keeps only the 3 newest checkpoints on disk.
    load_best_model_at_end=True,  # Restores the best checkpoint when done.
    metric_for_best_model="eval_loss",
    greater_is_better=False,

    # Logging.
    logging_steps=25,
    logging_first_step=True,
    # disable_tqdm=True # Avoids the notebook widget Kaggle logs cannot capture.
)

# Pads every row in a batch to the same length, labels included.
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator
)

# Measure the untrained model first, to compare the final numbers against.
baseline = trainer.evaluate()["eval_loss"]
print(f"Baseline Evaluation Loss: {baseline:.4f}")
print(f"Baseline Perplexity: {math.exp(baseline):.2f}")

# Fine-tune the base model.
print("Fine-tuning started...")
trainer.train()

# Save the LoRA adapter only, not a merged model.
model.save_pretrained(f"{OUTPUT_DIR}/final")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/final")

# Evaluate after fine-tuning.
eval_results = trainer.evaluate()["eval_loss"]
print(f"Final Evaluation Loss: {eval_results:.4f}")
print(f"Final Perplexity: {math.exp(eval_results):.2f}")
