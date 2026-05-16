"""
Phase 2 Prediction Script — Generate Adapter Predictions on Test Set

Designed to run on Google Colab (GPU required).
Uses native HuggingFace (AutoModelForCausalLM + PeftModel) for stability.
Loads the fine-tuned LoRA adapter and generates predictions for each
sample in the Golden Test set. Outputs predictions.jsonl for local evaluation.

Usage (Colab):
    !python predict_test_set.py
"""

import json
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# ==========================================
# 1. Configuration
# ==========================================
base_model_name = "unsloth/llama-3-8b-Instruct-bnb-4bit"
model_path = "/content/drive/MyDrive/adapter_training/preference_extractor_lora"
test_data_path = "/content/drive/MyDrive/adapter_training/data/combined_test.jsonl"
output_path = "/content/drive/MyDrive/adapter_training/data/predictions.jsonl"
max_seq_length = 2048

print("=" * 60)
print("Phase 2: Generate Predictions on Golden Test Set")
print(f"Base Model:  {base_model_name}")
print(f"Adapter:     {model_path}")
print(f"Test Data:   {test_data_path}")
print(f"Output:      {output_path}")
print("=" * 60)

# ==========================================
# 2. Load Model & Tokenizer via Native HF
# ==========================================
print("\n[1/3] Loading tokenizer and 4-bit base model...")
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Configure 4-bit quantization natively
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
# 3. Load Test Data
# ==========================================
print("\n[2/3] Loading test data...")
with open(test_data_path, "r", encoding="utf-8") as f:
    test_samples = [json.loads(line) for line in f if line.strip()]

print(f"Loaded {len(test_samples)} test samples.")

# ==========================================
# 4. Generate Predictions
# ==========================================
print("\n[3/3] Generating predictions...")
results = []

for i, sample in enumerate(test_samples):
    conversations = sample["conversations"]

    # Extract turns: system (0), human (1), gpt ground truth (2)
    system_msg = conversations[0]
    human_msg = conversations[1]
    ground_truth_msg = conversations[2]

    # Parse current_preferences and new_evidence from human turn
    human_text = human_msg["value"]
    parts = human_text.split("\n\n[New Evidence]\n", 1)
    current_prefs_str = parts[0].replace("[Current Preferences]\n", "")
    evidence_str = parts[1] if len(parts) > 1 else ""

    # Build chat messages in standard HF format (role/content)
    hf_messages = [
        {"role": "system", "content": system_msg["value"]},
        {"role": "user", "content": human_msg["value"]},
    ]

    # Apply LLaMA-3 chat template with generation prompt
    input_text = tokenizer.apply_chat_template(
        hf_messages, tokenize=False, add_generation_prompt=True
    )

    # Tokenize and move to device
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    # Greedy generation for reproducibility
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            temperature=1.0,
            use_cache=True,
        )

    # Decode only the newly generated tokens
    predicted = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    ).strip()

    # Assemble result with all fields needed for local evaluation
    result = {
        "id": sample.get("id", f"sample_{i}"),
        "metadata": sample.get("metadata", {}),
        "current_preferences": current_prefs_str,
        "new_evidence": evidence_str,
        "ground_truth": ground_truth_msg["value"],
        "predicted": predicted,
    }
    results.append(result)

    print(
        f"  [{i+1}/{len(test_samples)}] {result['id']}: "
        f"{len(predicted)} chars generated"
    )

# ==========================================
# 5. Save Predictions
# ==========================================
with open(output_path, "w", encoding="utf-8") as f:
    for result in results:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")

print(f"\n{'=' * 60}")
print(f"Done! Saved {len(results)} predictions to {output_path}")
print("Next step: Download predictions.jsonl and run evaluate_phase2.py locally.")
print("=" * 60)
