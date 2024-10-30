
import os
from torch import __version__
from packaging.version import Version as V

# Determine the appropriate xformers version based on the PyTorch version
xformers_version = "xformers==0.0.27" if V(__version__) < V("2.4.0") else "xformers"

# Define the list of packages to install
packages = [
    # 'unsloth',
    xformers_version,
    "trl",
    "peft",
    "accelerate",
    "bitsandbytes",
    "triton",
    "optuna"
]

# # Install the packages using pip
for package in packages:
    os.system(f"pip install --no-deps {package}")

from unsloth import FastLanguageModel
import torch
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, BitsAndBytesConfig
# import optuna
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import gc
# from huggingface_hub import login
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
from utils import synchronize, create_dataloaders, train_model, test_model, run_inference_and_collect_results, set_batch_size, clear_cuda_cache



gradient_checkpointing = True
use_dora = True
logging = True
# model_name = "meta-llama/Meta-Llama-3.1-8B"


max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
quantized = True
dtype = torch.bfloat16 if quantized else torch.float32 # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+



"""Initialize DistributedDataParallel (DDP) if available."""
ddp = int(os.environ.get('RANK', -1)) != -1  # Check if this is a DDP run.
world_size = int(os.environ['WORLD_SIZE'])

if ddp and world_size > 1:
    
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


token = "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"
# login(token)

bnb_config = BitsAndBytesConfig(load_in_4bit=True,
                                bnb_4bit_quant_type='nf4',
                                bnb_4bit_compute_dtype='bfloat16',
                                bnb_4bit_use_double_quant=True)


model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Meta-Llama-3.1-8B",
    max_seq_length = max_seq_length,
    dtype = dtype,
    quantization_config=bnb_config if quantized else None
    # load_in_4bit = quantized,
)

clear_cuda_cache()

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left' # setting padding to left as decoder models run from left to right

from datasets import load_from_disk
prompts_dataset = load_from_disk("Prepared_Datasets/Shifted/Prompts")
dataset = prompts_dataset.remove_columns('inputs')
train_dataset = dataset['train']
val_dataset = dataset['validation']
test_dataset = dataset['test']

# Training

model = FastLanguageModel.get_peft_model(
    model,
    use_dora=use_dora,
    r = 128, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0.2, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth" if gradient_checkpointing else False, # True or "unsloth" for very long context
)

clear_cuda_cache()



# for name, param in model.named_parameters():
#     if 'lora' not in name:
#         param.requires_grad = False

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

from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification, AutoTokenizer
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
from transformers import TrainerCallback, EarlyStoppingCallback
import random

train_batch_size = 32
val_batch_size = 4
grad_accum_steps = 2
lr_scheduler_type = "linear"
scheduler_warmup_steps = 10
num_train_epochs = 1
optimizer = "adamw_8bit"
lr = 2e-4
weight_decay = 0.01

save_steps = 500
eval_steps = 500
eval_metrics_every_n_steps = 1000
max_eval_batches = 10


# Define the compute_metrics function
def compute_metrics(p):
    predictions, labels = p
    preds = predictions.argmax(-1)
    
    # Compute accuracy, precision, recall, f1
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    
    # Return metrics as a dictionary
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }



args = TrainingArguments(
        logging_dir="./logs",
        output_dir="./results",
        per_device_train_batch_size = train_batch_size,
        per_device_eval_batch_size=4,          # Batch size for evaluation
        gradient_accumulation_steps = grad_accum_steps,
        warmup_steps = 10,
        num_train_epochs = 1, # Set this for 1 full training run.
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 10,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        eval_strategy="steps",
        eval_steps=eval_steps,        # Evaluate every `eval_steps` steps
        save_steps=save_steps,        # Save model checkpoints every `eval_steps` steps
        load_best_model_at_end=True,            # Load the best model (based on eval loss) at the end of training
    )

# Function to select a random subset of the evaluation dataset
def get_random_eval_subset(eval_dataset, max_eval_samples):
    # Shuffle the dataset with a random seed
    shuffled_dataset = eval_dataset.shuffle(seed=random.randint(0, 1000))
    # Select the first 'max_eval_samples' examples from the shuffled dataset
    return shuffled_dataset.select(range(max_eval_samples))

max_eval_samples = max_eval_batches * val_batch_size

# Custom Trainer class with conditional metric computation and accuracy printing
class MainTrainer(SFTTrainer):
    def __init__(self, *args, eval_metrics_every_n_steps=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_metrics_every_n_steps = eval_metrics_every_n_steps
        self.num_steps = 0  # To keep track of how many evaluations have been performed

    def evaluate(self, eval_dataset=None, **kwargs):

        self.num_steps += self.args.eval_steps

        # Select a random subset of the evaluation dataset
        eval_dataset = get_random_eval_subset(self.eval_dataset, max_eval_samples)

        # Compute metrics every eval by default (if eval_metrics_every_n_steps is not set)
        compute_metrics = True

        # If eval_metrics_every_n_steps is set, conditionally compute metrics
        if self.eval_metrics_every_n_steps is not None:
            if self.num_steps % self.eval_metrics_every_n_steps != 0:
                compute_metrics = False

        # Temporarily disable metric computation if not desired
        original_compute_metrics = self.compute_metrics
        if not compute_metrics:
            self.compute_metrics = None

        # Call the original evaluation method
        result = super().evaluate(eval_dataset=eval_dataset, **kwargs)

        # Restore original metric computation function
        if not compute_metrics:
            self.compute_metrics = original_compute_metrics

        # Print accuracy if metrics were computed
        if compute_metrics and 'accuracy' in result:
            print(f"Accuracy after evaluation {self.eval_count}: {result['accuracy']:.4f}")

        return result

trainer = MainTrainer(
    model = model,
    args=args,
    tokenizer = tokenizer,
    train_dataset = train_dataset,
    eval_dataset=val_dataset,
    dataset_text_field = "labels",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],
    eval_metrics_every_n_steps=eval_metrics_every_n_steps,
    compute_metrics=compute_metrics,

)

trainer.train()

# # Evaluate on validation dataset to see metrics with validation loss
# with torch.no_grad():
#     eval_results = trainer.evaluate()

# Start training

# Evaluate on validation dataset to see metrics with validation loss
eval_results = trainer.evaluate()

# Print the final evaluation metrics (validation loss and custom metrics)
print(f"Final Evaluation Metrics: {eval_results}")
