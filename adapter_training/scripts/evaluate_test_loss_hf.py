import torch
import math
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from peft import PeftModel
from transformers import Trainer, DataCollatorForLanguageModeling

# ==========================================
# 1. Configuration
# ==========================================
base_model_name = "unsloth/llama-3-8b-Instruct-bnb-4bit"
model_path = "/content/drive/MyDrive/adapter_training/preference_extractor_lora"
test_data_path = "/content/drive/MyDrive/adapter_training/data/combined_test.jsonl"
max_seq_length = 2048

print("="*50)
print(f"Phase 1: Evaluating Test Loss (Native HuggingFace)")
print(f"Model Path: {model_path}")
print(f"Test Data Path: {test_data_path}")
print("="*50)

# ==========================================
# 2. Load Model & Tokenizer via HF
# ==========================================
print("\n[1/4] Loading tokenizer and 4-bit base model...")
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Configure 4-bit quantization natively
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=bnb_config,
    device_map="auto"
)

print("Applying LoRA adapter...")
model = PeftModel.from_pretrained(base_model, model_path)
model.eval()

# ==========================================
# 3. Data Formatting
# ==========================================
print("\n[2/4] Preparing test dataset...")
def formatting_prompts_func(examples):
    convos = examples["conversations"]
    formatted_texts = []
    
    for convo in convos:
        # Convert our custom schema ('from', 'value') to standard HF chat template schema ('role', 'content')
        hf_convo = []
        for msg in convo:
            role = "user" if msg["from"] == "human" else "assistant" if msg["from"] == "gpt" else "system"
            hf_convo.append({"role": role, "content": msg["value"]})
            
        # Apply standard LLaMA-3 Chat Template
        text = tokenizer.apply_chat_template(hf_convo, tokenize=False, add_generation_prompt=False)
        formatted_texts.append(text)
        
    return { "text" : formatted_texts }

dataset = load_dataset("json", data_files=test_data_path, split="train")
print(f"Loaded {len(dataset)} Ground Truth samples.")

test_dataset = dataset.map(formatting_prompts_func, batched = True)

# ==========================================
# 4. Evaluation via Trainer
# ==========================================
# Standard HF Tokenization
def tokenize_func(examples):
    return tokenizer(examples["text"], truncation=True, max_length=max_seq_length, padding=False)

tokenized_test_dataset = test_dataset.map(tokenize_func, batched=True)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model = model,
    eval_dataset = tokenized_test_dataset,
    data_collator = data_collator,
    args = TrainingArguments(
        per_device_eval_batch_size = 2,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        output_dir = "/tmp/eval_results", 
        dataloader_num_workers = 2,
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
