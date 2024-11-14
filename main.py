import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import gc
from huggingface_hub import login
import pandas as pd
from contextlib import nullcontext
import torch.nn.functional as F
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from contextlib import contextmanager
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, BitsAndBytesConfig, get_scheduler
import re
import ast
from datasets import Dataset, DatasetDict, load_from_disk
import sys
from peft import LoraConfig, get_peft_model, LoftQConfig
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from utils import synchronize, create_dataloaders, train_model, test_model, run_inference_and_collect_results, set_batch_size, clear_cuda_cache

quantized = True
gradient_checkpointing = True
use_dora = True
logging = False
model_name = "meta-llama/Meta-Llama-3.1-8B"
# model_name = "unsloth/Meta-Llama-3.1-8B"

"""Initialize DistributedDataParallel (DDP) if available."""
ddp = int(os.environ.get('RANK', -1)) != -1  # Check if this is a DDP run.
if ddp:
    # Initialize the process group for distributed training
    dist.init_process_group(backend='nccl')
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{local_rank}'
    torch.cuda.set_device(device)
    is_master = (local_rank == 0)
    if is_master:
        print("Distirbuted learning ON")
        print("Global Rank:", rank)
        print("Local Rank:", local_rank)
        print("World Size:", world_size)
else:
    rank = 0
    local_rank = 0
    world_size = 1
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    is_master = True

device_type = "cuda" if device.startswith("cuda") else "cpu"

memory = torch.cuda.get_device_properties(device).total_memory / 1e9
if memory >= 80:
    gpu = "H100"
elif memory >= 40:
    gpu = "A100"
else:
    gpu = "L4"

if not ddp or is_master:
    print("Device type:", device_type)
    # Only execute this block on the master process (rank 0)
    print(f"Total Memory: {memory:.1f} GB")
    # Determine the type of GPU based on memory
    print(f"The current device being used is {device} on {gpu}")


# token = ""
# login(token)

bnb_config = BitsAndBytesConfig(load_in_4bit=True,
                                bnb_4bit_quant_type='nf4',
                                bnb_4bit_compute_dtype='bfloat16',
                                bnb_4bit_use_double_quant=True)

model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config) if quantized else AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

clear_cuda_cache()

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left' # setting padding to left as decoder models run from left to right

tokenized_dataset = load_from_disk("Prepared_Datasets/Shifted/Tokenized")
prompts_dataset = load_from_disk("Prepared_Datasets/Shifted/Prompts") 

# inference_batch_size, finetuning_batch_size = set_batch_size(gpu, quantized)
finetuning_batch_size=8
inference_batch_size=8

train_dataloader, train_inference_dataloader, val_dataloader, test_dataloader = create_dataloaders(tokenized_dataset, finetuning_batch_size, inference_batch_size, ddp, world_size, rank, tokenizer, tokenized=True, shuffle=True)
train_prompts_dataloader, _, val_prompts_dataloader, test_prompts_dataloader = create_dataloaders(prompts_dataset, finetuning_batch_size, inference_batch_size, ddp, world_size, rank, tokenizer, tokenized=False, shuffle=True)

# # Loss Calculation
# initial_val_loss = test_model(model, val_dataloader, device, local_rank, fp16=quantized, ddp=ddp, rounds=50)
# if is_master:
#     print(f"Initial Validation Set Loss: {initial_val_loss:.4f}")
# synchronize(device_type) # Add CUDA synchronization to ensure GPU finishes work before moving on

# initial_test_loss = test_model(model, test_dataloader, device, local_rank, fp16=quantized, ddp=ddp, rounds=50)
# if is_master:
#     print(f"Initial Validation Set Loss: {initial_val_loss:.4f}")
# synchronize(device_type) # Add CUDA synchronization to ensure GPU finishes work before moving on

# # Results Inference
# results, accuracy = run_inference_and_collect_results(model, val_prompts_dataloader, tokenizer, device, local_rank, fp16=quantized, ddp=ddp, rounds=50)
# if is_master:
#     print(f"Initial validation accuracy % is {accuracy}")
#     results.to_csv("val_results")
# synchronize(device_type) # Add CUDA synchronization to ensure GPU finishes work before moving on

# results, accuracy = run_inference_and_collect_results(model, test_prompts_dataloader, tokenizer, device, local_rank, fp16=quantized, ddp=ddp, rounds=50)
# if is_master:
#     print(f"Initial accuracy % is {accuracy}")
#     results.to_csv("test_results")
# synchronize(device_type) # Add CUDA synchronization to ensure GPU finishes work before moving on

# Training


lora_config = LoraConfig(
    use_dora=use_dora,
    r=256,
    lora_alpha=128,
    lora_dropout=0.2,
    bias="none",
    target_modules=['q_proj', 'k_proj', 'v_proj', 'dense'],
    task_type="CAUSAL_LM"  # Task type is language modeling
)

model = get_peft_model(model, lora_config)

clear_cuda_cache()

# Enable gradient checkpointing
if gradient_checkpointing:
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False}) 

for name, param in model.named_parameters():
    if 'lora' not in name:
        param.requires_grad = False

trainable = 0
all = 0
for name, param in model.named_parameters():
    all += param.numel()
    if param.requires_grad:
        trainable += param.numel()

if is_master:
    print(f"Number of parameters: {all}\nNumber of trainable parameters: {trainable}\n% of trainable parameters: {(trainable/all) * 100:.2f}%")

lora_params = [param for param in model.parameters() if param.requires_grad]

num_epochs = 3
optimizer = torch.optim.AdamW(lora_params, lr=2e-4)

num_training_steps = num_epochs * len(train_dataloader)
scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
clear_cuda_cache()
model = train_model(model, train_dataloader, val_dataloader, train_inference_dataloader, val_prompts_dataloader, 
    optimizer, tokenizer, device, local_rank, scheduler=None, num_epochs=1, grad_clipping=None,
    grad_accum_steps=2, logging_interval=10, eval_interval=100, accuracy_interval=100, eval_rounds=100, fp16=quantized, ddp=ddp)



if ddp:
    dist.destroy_process_group()
