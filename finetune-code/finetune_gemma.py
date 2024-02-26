from dataclasses import dataclass, field
from typing import Optional
import torch
import time
from transformers import AutoTokenizer, HfArgumentParser, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTTrainer




@dataclass
class ScriptArguments:
    """
    These arguments vary depending on how many GPUs you have, what their capacity and features are, and what size model you want to train.
    """
    per_device_train_batch_size: Optional[int] = field(default=4)
    per_device_eval_batch_size: Optional[int] = field(default=1)
    gradient_accumulation_steps: Optional[int] = field(default=4)
    learning_rate: Optional[float] = field(default=2e-4)
    max_grad_norm: Optional[float] = field(default=0.3)
    weight_decay: Optional[int] = field(default=0.001)
    lora_alpha: Optional[int] = field(default=16)
    lora_dropout: Optional[float] = field(default=0.1)
    lora_r: Optional[int] = field(default=8)
    max_seq_length: Optional[int] = field(default=512)
    model_name: Optional[str] = field(
        default="gemma-7b-it",
        metadata={
            "help": "The model that you want to train from the Hugging Face hub. E.g. gpt2, gpt2-xl, bert, etc."
        }
    )
    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The preference dataset to use."},
    )
    fp16: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables fp16 training."},
    )
    bf16: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables bf16 training."},
    )
    packing: Optional[bool] = field(
        default=True,
        metadata={"help": "Use packing dataset creating."},
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True,
        metadata={"help": "Enables gradient checkpointing."},
    )
    use_flash_attention_2: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables Flash Attention 2."},
    )
    optim: Optional[str] = field(
        default="paged_adamw_32bit",
        metadata={"help": "The optimizer to use."},
    )
    lr_scheduler_type: str = field(
        default="constant",
        metadata={"help": "Learning rate schedule. Constant a bit better than cosine, and has advantage for analysis"},
    )
    max_steps: int = field(default=300, metadata={"help": "How many optimizer update steps to take"})
    warmup_ratio: float = field(default=0.03, metadata={"help": "Fraction of steps to do a warmup for"})
    save_steps: int = field(default=30, metadata={"help": "Save checkpoint every X updates steps."})
    logging_steps: int = field(default=10, metadata={"help": "Log every X updates steps."})
    output_dir: str = field(
        default="./results",
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )

def formatting_func(example):
    text = f"### USER: {example['data'][0]}\n### ASSISTANT: {example['data'][1]}"
    return text

# Load the GG model - this is the local one, update it to the one on the Hub
model_id = "./gemma-7b-it"

quantization_config = BitsAndBytesConfig(
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4"
)

# Load model
print("loading model...")
t_start = time.time_ns() / 1000000
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    quantization_config=quantization_config, 
    torch_dtype=torch.float32,
    attn_implementation="sdpa" if not ScriptArguments.use_flash_attention_2 else "flash_attention_2"
)
t_end = time.time_ns() / 1000000
print("took {ms}ms to load gemma".format(ms=(t_end-t_start)))

# Load tokenizer
print("loading tokenizer...")
t_start = time.time_ns() / 1000000
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token_id = tokenizer.eos_token_id
t_end = time.time_ns() / 1000000
print("took {ms}ms to load tokenizer".format(ms=(t_end-t_start)))

lora_config = LoraConfig(
    r=ScriptArguments.lora_r,
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    bias="none",
    task_type="CAUSAL_LM",
    lora_alpha=ScriptArguments.lora_alpha,
    lora_dropout=ScriptArguments.lora_dropout
)

print("loading data")
train_dataset = load_dataset("json",  data_files="./data/book_data_sft.txt", split="train")

print("training data size is {size}".format(size=len(train_dataset)))

output_dir = f"finetune_output/gemma-book"

training_arguments = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=ScriptArguments.per_device_train_batch_size,
    gradient_accumulation_steps=ScriptArguments.gradient_accumulation_steps,
    optim=ScriptArguments.optim,
    save_steps=ScriptArguments.save_steps,
    logging_steps=ScriptArguments.logging_steps,
    learning_rate=ScriptArguments.learning_rate,
    max_grad_norm=ScriptArguments.max_grad_norm,
    max_steps=ScriptArguments.max_steps,
    warmup_ratio=ScriptArguments.warmup_ratio,
    lr_scheduler_type=ScriptArguments.lr_scheduler_type,
    gradient_checkpointing=ScriptArguments.gradient_checkpointing,
    fp16=ScriptArguments.fp16,
    bf16=ScriptArguments.bf16,
)

trainer = SFTTrainer(
    model=model,
    args=training_arguments,
    train_dataset=train_dataset,
    peft_config=lora_config,
    packing=ScriptArguments.packing,
    dataset_text_field="id",
    tokenizer=tokenizer,
    max_seq_length=ScriptArguments.max_seq_length,
    formatting_func=formatting_func,
)
print("start training...")
t_start = time.time_ns() / 1000000
trainer.train()
t_end = time.time_ns() / 1000000
print("took {ms}ms to finetune".format(ms=(t_end-t_start)))




