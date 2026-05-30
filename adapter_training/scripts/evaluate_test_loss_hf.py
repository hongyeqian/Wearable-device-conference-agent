"""
Phase 1 Evaluation — Response-Only Test Loss (Native HuggingFace)

Key improvement: labels for system/user tokens are set to -100,
so CrossEntropyLoss only measures the assistant response tokens.

Usage (Colab):
    !python evaluate_test_loss_hf.py
"""

import torch
import math
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import PeftModel

# ==========================================
# 1. Configuration
# ==========================================
base_model_name = "unsloth/llama-3-8b-Instruct-bnb-4bit"
model_path = "/content/drive/MyDrive/adapter_training/preference_extractor_lora"
test_data_path = "/content/drive/MyDrive/adapter_training/data/combined_test.jsonl"
max_seq_length = 2048

print("=" * 60)
print("Phase 1: Response-Only Test Loss (Native HuggingFace)")
print(f"Base Model:  {base_model_name}")
print(f"Adapter:     {model_path}")
print(f"Test Data:   {test_data_path}")
print("=" * 60)

# ==========================================
# 2. Load Model & Tokenizer via HF
# ==========================================
print("\n[1/4] Loading tokenizer and 4-bit base model...")
tokenizer = AutoTokenizer.from_pretrained(model_path)

# LLaMA-3 has no pad_token by default; reuse eos_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 4-bit NF4 quantization (same config as training)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=(
        torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    ),
)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=bnb_config,
    device_map="auto",
)

print("Applying LoRA adapter...")
model = PeftModel.from_pretrained(base_model, model_path)
model.eval()

# ==========================================
# 3. Data Formatting — Response-Only Masking
# ==========================================
print("\n[2/4] Preparing test dataset with response-only masking...")

# Masking statistics (populated during map)
_stats = {"total": 0, "masked": 0, "response": 0}


def prepare_masked_dataset(examples):
    """
    For each sample:
      1. apply_chat_template(full conversation)     → full token sequence
      2. apply_chat_template(without assistant msg)  → prompt-only tokens
      3. prompt tokens get label=-100 (ignored by loss)
      4. assistant response tokens keep real label (evaluated)
    """
    all_input_ids = []
    all_attention_mask = []
    all_labels = []

    for convo in examples["conversations"]:
        # ShareGPT → HF chat format
        hf_msgs = []
        for msg in convo:
            role = (
                "user" if msg["from"] == "human"
                else "assistant" if msg["from"] == "gpt"
                else "system"
            )
            hf_msgs.append({"role": role, "content": msg["value"]})

        # Full text: system + user + assistant response
        full_text = tokenizer.apply_chat_template(
            hf_msgs, tokenize=False, add_generation_prompt=False
        )
        # Prompt only: system + user + generation-prompt header
        prompt_text = tokenizer.apply_chat_template(
            hf_msgs[:-1], tokenize=False, add_generation_prompt=True
        )

        # Tokenize both
        full_enc = tokenizer(
            full_text, truncation=True,
            max_length=max_seq_length, padding=False,
        )
        prompt_enc = tokenizer(
            prompt_text, truncation=True,
            max_length=max_seq_length, padding=False,
        )

        input_ids = full_enc["input_ids"]
        attn_mask = full_enc["attention_mask"]
        prompt_len = len(prompt_enc["input_ids"])

        # Labels: -100 for prompt, real IDs for response
        labels = [-100] * prompt_len + input_ids[prompt_len:]

        # Accumulate stats
        _stats["total"] += len(input_ids)
        _stats["masked"] += prompt_len
        _stats["response"] += len(input_ids) - prompt_len

        all_input_ids.append(input_ids)
        all_attention_mask.append(attn_mask)
        all_labels.append(labels)

    return {
        "input_ids": all_input_ids,
        "attention_mask": all_attention_mask,
        "labels": all_labels,
    }


dataset = load_dataset("json", data_files=test_data_path, split="train")
print(f"Loaded {len(dataset)} Ground Truth samples.")

eval_dataset = dataset.map(
    prepare_masked_dataset,
    batched=True,
    remove_columns=dataset.column_names,
)

# Print masking stats
if _stats["total"] > 0:
    pct_m = _stats["masked"] / _stats["total"] * 100
    pct_r = _stats["response"] / _stats["total"] * 100
    avg_r = _stats["response"] / len(dataset)
    print(f"\n  Masking Statistics:")
    print(f"    Total tokens:       {_stats['total']}")
    print(f"    Masked (prompt):    {_stats['masked']} ({pct_m:.1f}%)")
    print(f"    Evaluated (resp):   {_stats['response']} ({pct_r:.1f}%)")
    print(f"    Avg response len:   {avg_r:.0f} tokens/sample")

# ==========================================
# 4. Evaluation via Trainer
# ==========================================
print("\n[3/4] Initializing Trainer...")

# DataCollatorForSeq2Seq pads input_ids with pad_token_id
# and pads labels with -100 (so padding never contributes to loss)
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    padding=True,
    label_pad_token_id=-100,
)

trainer = Trainer(
    model=model,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    args=TrainingArguments(
        per_device_eval_batch_size=2,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        output_dir="/tmp/eval_results",
        dataloader_num_workers=2,
    ),
)

# ==========================================
# 5. Run Evaluation & Report
# ==========================================
print("\n[4/4] Computing Response-Only Test Loss...")
eval_results = trainer.evaluate()

test_loss = eval_results.get("eval_loss", None)
if test_loss is not None:
    test_perplexity = math.exp(test_loss)
    print("\n" + "=" * 60)
    print("  RESPONSE-ONLY EVALUATION RESULTS")
    print(f"  Test Loss (response-only):       {test_loss:.4f}")
    print(f"  Test Perplexity (response-only):  {test_perplexity:.4f}")
    print("=" * 60)
    print(f"\n  Debug Stats:")
    print(f"  - Total Masked Tokens (ignored): {_stats['masked']}")
    print(f"  - Total Evaluated Tokens (loss): {_stats['response']}")
    print(f"\n  Loss is computed ONLY on assistant response tokens.")
    print(f"  Expect this to be HIGHER than the old full-sequence loss.")
else:
    print("Error: Could not calculate eval_loss.")
