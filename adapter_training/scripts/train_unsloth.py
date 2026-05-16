"""
Unsloth Fine-Tuning Script for LLaMA-3-8B-Instruct (Preference Extraction)
Designed to run on Google Colab (T4/A100) or local GPU.

Instructions for Google Colab:
1. Install Unsloth:
   !pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
   !pip install --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes
2. Upload your `combined_train.jsonl` to Colab.
3. Run this script!
"""

import os
import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from trl import SFTTrainer, SFTConfig
from transformers import TrainingArguments

# ==========================================
# 1. Configuration
# ==========================================
max_seq_length = 2048 # Good enough for preference extraction contexts
dtype = None # None for auto-detection (Float16/Bfloat16)
load_in_4bit = True # Use 4bit quantization to save memory

model_name = "unsloth/llama-3-8b-Instruct-bnb-4bit" # Pre-quantized for speed
data_path = "/content/drive/MyDrive/adapter_training/data/combined_train.jsonl"
output_dir = "/content/drive/MyDrive/adapter_training/preference_extractor_lora"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created directory: {output_dir}")
# ==========================================
# 2. Load Model & Tokenizer
# ==========================================
print("Loading model and tokenizer...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

# Apply LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 32,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

# ==========================================
# 3. Data Formatting (ShareGPT / LLaMA-3)
# ==========================================
print("Preparing dataset...")
# LLaMA-3 uses a specific chat template. Unsloth simplifies this map.
tokenizer = get_chat_template(
    tokenizer,
    chat_template = "llama-3",
    mapping = {"role": "from", "content": "value", "user": "human", "assistant": "gpt"}
)

def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
    return { "text" : texts }

dataset = load_dataset("json", data_files=data_path, split="train")
dataset = dataset.map(formatting_prompts_func, batched = True)

# Split into 90% train and 10% eval
dataset = dataset.train_test_split(test_size=0.1)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

# ==========================================
# 4. Training
# ==========================================
print("Starting training...")
trainer = SFTTrainer(
    model = model,
    processing_class = tokenizer,
    train_dataset = train_dataset,
    eval_dataset = eval_dataset,
    args = SFTConfig(
        dataset_text_field = "text",
        max_seq_length = None,
        dataset_num_proc = 2,
        packing = True, 
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 150, 
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 5,
        eval_strategy = "steps",
        eval_steps = 10,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = output_dir, # Use the predefined Google Drive folder
        save_strategy = "steps", # Save checkpoints based on steps
        save_steps = 50,         # Save every 50 steps
        save_total_limit = 2,    # Keep only the latest 2 checkpoints to save Drive space
    ),
)

# Auto-detect last checkpoint to resume if disconnected
from transformers.trainer_utils import get_last_checkpoint
last_checkpoint = get_last_checkpoint(output_dir) if os.path.exists(output_dir) else None

if last_checkpoint:
    print(f"Resuming training from checkpoint: {last_checkpoint}")
    trainer_stats = trainer.train(resume_from_checkpoint=last_checkpoint)
else:
    trainer_stats = trainer.train()

# ==========================================
# 5. Visualization (Loss Curve)
# ==========================================
print("Generating loss curve...")
import matplotlib.pyplot as plt

history = trainer.state.log_history

train_steps = [x['step'] for x in history if 'loss' in x]
train_loss = [x['loss'] for x in history if 'loss' in x]

eval_steps = [x['step'] for x in history if 'eval_loss' in x]
eval_loss = [x['eval_loss'] for x in history if 'eval_loss' in x]

plt.figure(figsize=(10, 6))
plt.plot(train_steps, train_loss, label='Train Loss', color='blue')

# Plot evaluation loss if it exists
if len(eval_loss) > 0:
    plt.plot(eval_steps, eval_loss, label='Eval Loss', color='red', marker='o', linestyle='--')

plt.title('Training and Evaluation Loss Curve (Preference Extraction)')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, "loss_curve.png"))
print(f"Loss curve saved to {os.path.join(output_dir, 'loss_curve.png')}")

# ==========================================
# 6. Save the Adapter (LoRA)
# ==========================================
print(f"Saving LoRA adapter to {output_dir}...")
model.save_pretrained(output_dir) # Local saving
tokenizer.save_pretrained(output_dir)

print("Done! You can now use this LoRA adapter with vLLM or Ollama.")
