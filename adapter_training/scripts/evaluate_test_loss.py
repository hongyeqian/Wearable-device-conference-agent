import os
import torch
import math
from datasets import load_dataset
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from trl import SFTTrainer, SFTConfig
from transformers import TrainingArguments

# ==========================================
# 1. Configuration
# ==========================================
max_seq_length = 2048
dtype = None # Auto-detection
load_in_4bit = True

# Path where your LoRA weights were saved during training
model_path = "/content/drive/MyDrive/adapter_training/preference_extractor_lora"
# Path to our newly sanitized and combined Ground Truth test set
test_data_path = "/content/drive/MyDrive/adapter_training/data/combined_test.jsonl"

print("="*50)
print(f"Phase 1: Evaluating Test Loss")
print(f"Model Path: {model_path}")
print(f"Test Data Path: {test_data_path}")
print("="*50)

# ==========================================
# 2. Load Model & Tokenizer
# ==========================================
print("\n[1/4] Loading model and tokenizer...")
# FastLanguageModel automatically detects the base model from adapter_config.json 
# and merges the LoRA weights.
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_path,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

# For inference/evaluation, we do NOT need gradients
FastLanguageModel.for_inference(model)

# ==========================================
# 3. Data Formatting
# ==========================================
print("\n[2/4] Preparing test dataset...")
tokenizer = get_chat_template(
    tokenizer,
    chat_template = "llama-3",
    mapping = {"role": "from", "content": "value", "user": "human", "assistant": "gpt"}
)

def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
    return { "text" : texts }

dataset = load_dataset("json", data_files=test_data_path, split="train")
print(f"Loaded {len(dataset)} Ground Truth samples.")

test_dataset = dataset.map(formatting_prompts_func, batched = True)

# ==========================================
# 4. Evaluation via Trainer
# ==========================================
print("\n[3/4] Initializing Trainer for Evaluation...")

trainer = SFTTrainer(
    model = model,
    processing_class = tokenizer,
    eval_dataset = test_dataset,
    args = SFTConfig(
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        dataset_num_proc = 2,
        packing = False, # Set to False for cleaner eval metric alignment per sample
        per_device_eval_batch_size = 2,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        output_dir = "/tmp/eval_results", # Just a temporary directory for eval
    ),
)

# ==========================================
# 5. Run Evaluation & Calculate Metrics
# ==========================================
print("\n[4/4] Computing Test Loss...")
eval_results = trainer.evaluate()

test_loss = eval_results.get("eval_loss", None)
if test_loss is not None:
    test_perplexity = math.exp(test_loss)
    print("\n" + "="*50)
    print("🏆 EVALUATION RESULTS (GROUND TRUTH TEST SET) 🏆")
    print(f"Test Loss:       {test_loss:.4f}")
    print(f"Test Perplexity: {test_perplexity:.4f}")
    print("="*50)
    print("\nAnalysis Tip:")
    print("- If Test Loss is roughly equal to or slightly higher than your final Train Loss (~0.5 - 0.8), your model has excellent generalization!")
    print("- If Test Loss is extremely high (> 2.0), your model has overfitted and memorized the training set.")
else:
    print("Error: Could not calculate eval_loss.")
