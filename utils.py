import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import gc
import pandas as pd
from contextlib import nullcontext
import torch.nn.functional as F
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, BitsAndBytesConfig, AdamW, get_scheduler
from peft import LoraConfig, get_peft_model
from datasets import Dataset, DatasetDict, ClassLabel
from contextlib import contextmanager
import re
import ast
import sys
import time
from datetime import datetime
from trl import SFTTrainer
import random

# -----------------------------
# Distributed Training Setup
# -----------------------------
def clear_cuda_cache():
    """
    Frees up memory by collecting garbage and emptying the CUDA cache.
    Useful to call during or after model training to reduce memory usage.
    """
    gc.collect()
    torch.cuda.empty_cache()


def synchronize(device_type):
    """
    Synchronize the device operations to ensure the GPU (CUDA or MPS) finishes work before moving on.
    For 'cuda', it uses torch.cuda.synchronize().
    For 'mps', it uses torch.mps.synchronize().
    No synchronization is needed for 'cpu'.
    """
    if device_type == "cuda":
        torch.cuda.synchronize()  # Synchronize CUDA operations
    elif device_type == "mps":
        torch.mps.synchronize()  # Synchronize MPS operations (for Apple Silicon GPUs)
    elif device_type == "cpu":
        # No synchronization needed for CPU since operations are synchronous
        pass
    else:
        raise ValueError(f"Unknown device type: {device_type}")


def clean_up(device_type):
    synchronize(device_type)
    clear_cuda_cache()

def create_dataloaders(tokenized_dataset, finetuning_batch_size, inference_batch_size, ddp, world_size, rank, tokenizer, tokenized=True, shuffle=True):
    """
    Create DataLoaders with DistributedSampler if in DDP mode. 
    If the dataset is tokenized, create two train DataLoaders: one for fine-tuning and one for inference.
    For non-tokenized data, create standard DataLoaders.
    """
    
    # Define the data collator (for language modeling tasks, assuming non-MLM)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    if tokenized:
        # If dataset is already tokenized, set format for PyTorch tensors
        tokenized_dataset['train'].set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        tokenized_dataset['validation'].set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        tokenized_dataset['test'].set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        
    # If using DDP, shard the dataset across processes
    if ddp:
        train_sampler = DistributedSampler(tokenized_dataset['train'], num_replicas=world_size, rank=rank, shuffle=shuffle)
        val_sampler = DistributedSampler(tokenized_dataset['validation'], num_replicas=world_size, rank=rank, shuffle=False)
        test_sampler = DistributedSampler(tokenized_dataset['test'], num_replicas=world_size, rank=rank, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None
        test_sampler = None

    if tokenized:
        # Remove custom labels column (if present)
        tokenized_dataset = tokenized_dataset.remove_columns("labels")
        
        # Create two DataLoaders for training: one for fine-tuning, one for inference
        finetuning_train_dataloader = DataLoader(
            tokenized_dataset['train'],
            batch_size=finetuning_batch_size,  # Fine-tuning uses the fine-tuning batch size
            shuffle=(train_sampler is None),  # Shuffle if no sampler
            sampler=train_sampler,
            collate_fn=data_collator  # Use the data collator for tokenized data
        )

        inference_train_dataloader = DataLoader(
            tokenized_dataset['train'],
            batch_size=inference_batch_size,  # Inference uses the inference batch size
            shuffle=False,  # No need to shuffle during inference
            sampler=train_sampler,
            collate_fn=data_collator  # Use the data collator for tokenized data
        )
    else:
        # For non-tokenized data, just return a single train DataLoader
        finetuning_train_dataloader = DataLoader(
            tokenized_dataset['train'],
            batch_size=finetuning_batch_size,  # Fine-tuning batch size
            shuffle=(train_sampler is None),  # Shuffle if no sampler
            sampler=train_sampler
        )
        inference_train_dataloader = None  # No inference DataLoader for non-tokenized case

    # Create DataLoader for validation (with inference batch size)
    val_dataloader = DataLoader(
        tokenized_dataset['validation'],
        batch_size=inference_batch_size,  # Inference uses the inference batch size
        shuffle=False,  # No need to shuffle validation data
        sampler=val_sampler,
        collate_fn=(data_collator if tokenized else None)  # Use the data collator only for tokenized data
    )
    
    # Create DataLoader for testing (with inference batch size)
    test_dataloader = DataLoader(
        tokenized_dataset['test'],
        batch_size=inference_batch_size,  # Inference uses the inference batch size
        shuffle=False,  # No need to shuffle test data
        sampler=test_sampler,
        collate_fn=(data_collator if tokenized else None)  # Use the data collator only for tokenized data
    )
    
    # Return two train dataloaders if tokenized, otherwise just one
    return finetuning_train_dataloader, inference_train_dataloader, val_dataloader, test_dataloader


def test_model(model, test_dataloader, device, local_rank, rounds=None, fp16=False, ddp=False, logging=False):
    """
    Evaluate the model on a dataset, with support for DDP (Distributed Data Parallelism).
    Logs the number of examples processed by the master process and the total across all processes if DDP is enabled.
    
    Parameters:
    - model: The model to evaluate.
    - test_dataloader: DataLoader for the test set.
    - device: The device (GPU/CPU) where the model is located.
    - local_rank: Rank ID for distributed training.
    - rounds: Number of rounds/batches to process (optional).
    - fp16: Whether to use automatic mixed precision (AMP) with half-precision.
    - ddp: Whether to use Distributed Data Parallel (DDP).
    - logging: Whether to log the total time taken and number of examples processed on the master process.
    
    Returns:
    - avg_loss: The average loss over the test dataset.
    """
    
    # Set the model to evaluation mode to disable dropout and other training-specific behaviors.
    model.eval()
    
    # Initialize variables to accumulate total loss and track examples processed.
    total_loss = 0.0
    total_examples = 0
    steps = 0  # Track the number of batches processed
    total_tokens = 0

    input_length = next(iter(test_dataloader))['input_ids'][0]

    if rounds is None:
        rounds = len(test_dataloader)

    # Start the total evaluation time only on the master process if logging is enabled.
    if logging and local_rank == 0:
        start_time = time.time()

    # Disable gradient calculations during evaluation to save memory and computation.
    with torch.no_grad():
        for step, batch in enumerate(test_dataloader):
            # If a limited number of rounds is specified, stop after the given number of batches.
            if rounds is not None and step >= rounds:
                break  # Stop if the specified number of rounds is reached

            # Move the batch data to the correct device (GPU/CPU)
            batch = {k: v.to(device) for k, v in batch.items()}

            # If using half-precision (fp16), enable automatic mixed precision.
            context_manager = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if fp16 else nullcontext()

            # Forward pass: Compute the model output and loss.
            with context_manager:
                outputs = model(**batch)
            loss = outputs.loss

            # Accumulate the total loss for the batch.
            loss = loss / rounds
            total_loss += loss.detach()
            

            # Accumulate the number of examples processed in this batch.
            total_tokens += input_length
        total_examples = len(test_dataloader) * test_dataloader.batch_size
    
    if ddp:
        total_examples_tensor = torch.tensor(total_examples, dtype=torch.float, device=device)
        dist.all_reduce(total_loss, op=dist.ReduceOp.AVG)
        dist.all_reduce(total_examples_tensor, op=dist.ReduceOp.SUM)

    # # Logging: Only the master process (rank 0) logs the total time and number of examples.
    # if logging and local_rank == 0:
    #     total_time = time.time() - start_time
    #     print(f"Master process {local_rank} processed {total_examples} examples.")
        
    #     # If using DDP, log the total number of examples processed across all processes.
    #     if ddp:
    #         total_examples_tensor = torch.tensor([total_examples], dtype=torch.float, device=device)
    #         dist.all_reduce(total_examples_tensor, op=dist.ReduceOp.SUM)
    #         print(f"Total examples processed across all processes: {int(total_examples_tensor.item())}")
        
    #     # Log the total evaluation time.
    #     print(f"Total evaluation time: {total_time:.2f} seconds")

    return total_loss


def run_inference_and_collect_results(model, text_dataloader, tokenizer, device, local_rank, rounds=None, fp16=False, ddp=False, logging=False):
    """
    Run inference on a model using a text dataloader and collect the results.
    Handles optional mixed precision (bfloat16) and distributed data parallel (DDP) support.
    Logs the total time and number of examples processed on the master GPU, as well as the total across all GPUs if DDP is enabled.
    
    Args:
        model (torch.nn.Module): The pre-trained model for inference.
        text_dataloader (DataLoader): A dataloader providing batches of input text data.
        tokenizer (PreTrainedTokenizer): A tokenizer to convert between tokens and text.
        device (torch.device or str): The device on which to run the model (e.g., 'cuda' or 'cpu').
        local_rank (int): Rank of the current process in distributed training (0 for master).
        rounds (int, optional): The number of batches to process (useful for debugging or limited runs). Defaults to None, meaning process all batches.
        fp16 (bool, optional): Whether to use mixed precision with bfloat16. Defaults to False.
        ddp (bool, optional): Whether to gather results across distributed processes (DDP). Defaults to False.
        logging (bool, optional): Whether to log the total time taken and number of examples processed. Defaults to False.
        
    Returns:
        tuple: A tuple containing:
            - results_df (pd.DataFrame): A DataFrame with columns ['Input', 'Output', 'Expected Output', 'Output Answer', 'Expected Output Answer', 'Match'].
            - accuracy (float): The percentage of exact matches between model output and expected output.
            If the process is not the master (local_rank != 0), both values will be None.
    """
    
    model.eval()  # Set the model to evaluation mode
    
    process_results = []  # List to store all inference results
    
    total_examples = 0  # Track total examples processed by this process

    if logging and local_rank == 0:  # Log the time only on master
        start_time = time.time()

    with torch.no_grad():  # Disable gradient tracking during inference
        for i, batch in enumerate(text_dataloader):
            if rounds is not None and i >= rounds:  # Stop after specified rounds
                break

            input_texts = batch['inputs']  # Input texts from batch
            label_texts = batch['labels']  # Expected output texts
            
            # Tokenize inputs and move to device
            tokenized_inputs = tokenizer(
                input_texts, 
                padding='max_length',  
                max_length=2048,        
                truncation=True,       
                return_tensors='pt'    
            ).to(device)
            
            context_manager = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if fp16 else nullcontext()
            with context_manager:
                output_ids = model.generate(
                    tokenized_inputs['input_ids'],    
                    attention_mask=tokenized_inputs['attention_mask'],
                    max_new_tokens=30,                
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )

            output_texts = [tokenizer.decode(output_id, skip_special_tokens=True) for output_id in output_ids]
            cleaned_output_texts = [remove_extra_output(output_text) for output_text in output_texts]

            for input_text, output_text, label_text in zip(input_texts, cleaned_output_texts, label_texts):
                # Remove the eos token from the label text if present
                label_text = label_text.replace(tokenizer.eos_token, "").strip()
                
                output_answer = isolate_output(output_text)  # Extract final answer from model's output
                label_answer = isolate_output(label_text)    # Extract final answer from label after cleaning
                
                match = (output_answer.strip() == label_answer.strip())  # Check if the answers match
                
                process_results.append((input_text, output_text, label_text, output_answer, label_answer, match))
            
            total_examples += len(input_texts)  # Accumulate number of examples processed

    if ddp:  # If using DDP, gather results from all processes
        gathered_results = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered_results, process_results)
        all_results = [item for sublist in gathered_results for item in sublist]
    else:
        all_results = process_results

    if local_rank == 0:  # Only master process returns final results
        results_df = pd.DataFrame(all_results, columns=['Input', 'Output', 'Expected Output', 'Output Answer', 'Expected Output Answer', 'Match'])
        accuracy = (results_df['Match'].mean()) * 100  # Calculate accuracy
        
        if logging:
            total_time = time.time() - start_time
            print(f"Master process {local_rank} processed {total_examples} examples.")
            if ddp:
                total_examples_tensor = torch.tensor([total_examples], dtype=torch.float, device=device)
                dist.all_reduce(total_examples_tensor, op=dist.ReduceOp.SUM)
                print(f"Total examples processed across all processes: {int(total_examples_tensor.item())}")
            print(f"Total inference time: {total_time:.2f} seconds")
        
        return results_df, accuracy
    else:
        return None, None
    

def train_model(model, train_dataloader, val_dataloader, train_inference_dataloader, val_prompts_dataloader, 
    optimizer, tokenizer, device, local_rank, scheduler=None, num_epochs=5, grad_clipping=None, 
    eval_rounds=50, grad_accum_steps=1, logging_interval=10, eval_interval=50, accuracy_interval=None, fp16=False, 
    ddp=False, log_dir="logs", logging=False, checkpoint_dir="checkpoints", log_file="training_log.txt", resume_checkpoint=None):

    # Load LoRA checkpoint if resuming
    start_epoch = 0
    start_step = 0
    if resume_checkpoint:
        start_epoch, start_step = load_lora_checkpoint(model, optimizer, scheduler, resume_checkpoint, device)

    try:
        # Initialize device and move model to the appropriate device
        device_type = "cuda" if device.startswith("cuda") else "cpu"
        model.to(device)

        if ddp:
            model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        local_effective_batch_size = train_dataloader.batch_size * grad_accum_steps

        for epoch in range(start_epoch, num_epochs):
            logging_loss = 0.0
            avg_train_loss = 0.0
            loader = iter(train_dataloader)
            max_steps = len(train_dataloader) // grad_accum_steps
            logging_start_time = time.time()
            accumulated_time = 0.0

            if local_rank == 0:
                log_to_file(log_file, f"Starting Epoch {epoch}/{num_epochs} - {max_steps} steps")

            for step in range(start_step, max_steps):
                # Pause the logging timer before evaluation
                logging_end_time = time.time()  # End the logging timer
                accumulated_time += logging_end_time - logging_start_time  # Accumulate time difference

                if (step % eval_interval == 0 or step == max_steps - 1) and step > 0:
                    model.eval()
                    eval_start_time = time.time()
                    with torch.no_grad():
                        avg_val_loss = test_model(model.module, val_dataloader, device, local_rank, ddp=True, fp16=fp16, rounds=eval_rounds)
                        clear_cuda_cache()
                        if accuracy_interval is not None and step % accuracy_interval == 0:
                            results, accuracy = run_inference_and_collect_results(model.module, val_prompts_dataloader, tokenizer, device, local_rank, ddp=True, fp16=fp16, rounds=eval_rounds)
                            clear_cuda_cache()
                            log_message = f"Epoch {epoch}, Step {step}, Training Loss: {avg_train_loss/eval_interval:.4f}, Validation Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.2f}%" if local_rank == 0 else None
                        else:
                            log_message = f"Epoch {epoch}, Step {step}, Training Loss: {avg_train_loss/eval_interval:.4f}, Validation Loss: {avg_val_loss:.4f}" if local_rank == 0 else None

                        # Stop timing the evaluation
                        eval_end_time = time.time()
                        eval_time = eval_end_time - eval_start_time
                        if local_rank == 0:
                            log_message += f", Eval Time: {eval_time:.2f} seconds"
                            print(log_message)
                            log_to_file(log_file, log_message)

                    avg_train_loss = 0.0

                    # Save only LoRA modules after evaluation
                    if local_rank == 0 and step > 0:
                        save_lora_checkpoint(model.module, optimizer, scheduler, epoch, step, avg_val_loss, checkpoint_dir, local_rank)

                # Restart the logging timer after evaluation
                logging_start_time = time.time()

                model.train()
                optimizer.zero_grad(set_to_none=True)

                loss_accum = 0.0
                for micro_step in range(grad_accum_steps):
                    next_batch = next(loader)
                    batch = {k: v.to(device) for k, v in next_batch.items()}
                    if ddp:
                        model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
                    context_manager = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if fp16 else nullcontext()
                    with context_manager:
                        outputs = model(**batch)
                    loss = outputs.loss / grad_accum_steps
                    loss_accum += loss.detach()
                    loss.backward()

                if ddp:
                    dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
                
                if grad_clipping is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clipping)

                logging_loss += loss_accum
                avg_train_loss += loss_accum
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                clean_up(device_type)

                if step % logging_interval == 0 and step > 0:
                    logging_end_time = time.time()  # End the logging timer
                    accumulated_time += logging_end_time - logging_start_time  # Accumulate time difference
                    if local_rank == 0:
                        log_message = f"Epoch {epoch}, Step {step}, Avg Training Loss: {logging_loss/logging_interval:.4f}, Time Taken: {accumulated_time:.2f} seconds"
                        print(log_message)
                        log_to_file(log_file, log_message)
                    logging_loss = 0.0
                    accumulated_time = 0.0  # Reset accumulated time
                    logging_start_time = time.time()  # Restart the timer for the next interval

    except KeyboardInterrupt:
        # Save the last LoRA checkpoint before exiting on KeyboardInterrupt
        save_lora_checkpoint(model.module, optimizer, scheduler, epoch, step, checkpoint_dir, local_rank)
        model.eval()
        return model, optimizer, scheduler

    model.eval()
    return model, optimizer, scheduler

                

def train_model_with_trainer(trainer, num_batches, train_inference_dataloader, val_dataloader, val_prompts_dataloader, eval_rounds, logging=False):
    """
    Run the trainer for a specific number of batches, pause to run custom functions, log time and examples processed.
    """
    total_examples_processed = 0  # Track the total number of examples processed
    start_time = time.time()  # Start timer for the training process

    for step in range(num_batches):
        # Run a single training step
        trainer.train(resume_from_checkpoint=False)

        # Track the number of examples processed (per device)
        examples_processed = trainer.args.per_device_train_batch_size * len(trainer.get_train_dataloader().batch_sampler)
        total_examples_processed += examples_processed

        # Log processing info on the master process if logging is enabled
        if logging and trainer.args.local_rank == 0:
            current_time = time.time()
            elapsed_time = current_time - start_time
            print(f"Step {step}: Master process processed {examples_processed} examples.")
            print(f"Total examples processed (master): {total_examples_processed}.")
            print(f"Time elapsed: {elapsed_time:.2f} seconds.")
        
        # Distributed processing: gather examples processed from all processes
        if dist.is_initialized():
            examples_tensor = torch.tensor([examples_processed], dtype=torch.float, device=trainer.device)
            dist.all_reduce(examples_tensor, op=dist.ReduceOp.SUM)
            if logging and trainer.args.local_rank == 0:
                print(f"Total examples processed across all processes: {int(examples_tensor.item())}.")

        # Pause training at intervals to run custom functions
        if step % num_batches == 0:
            if logging and trainer.args.local_rank == 0:
                print(f"Running custom test_model and run_inference at step {step}")

            # Run test_model function (on training and validation datasets)
            train_loss = test_model(trainer.model, train_inference_dataloader, trainer.device, trainer.args.local_rank, eval_rounds)
            val_loss = test_model(trainer.model, val_dataloader, trainer.device, trainer.args.local_rank, eval_rounds)
            
            if logging and trainer.args.local_rank == 0:
                print(f"Step {step}: Train Loss = {train_loss:.4f}, Validation Loss = {val_loss:.4f}")

            # Run run_inference_and_collect_results function
            results, accuracy = run_inference_and_collect_results(trainer.model, val_prompts_dataloader, trainer.device, trainer.args.local_rank, eval_rounds)
            
            if logging and trainer.args.local_rank == 0:
                print(f"Step {step}: Validation Accuracy = {accuracy:.2f}%")

        # Checkpoint saving can also be done here at intervals
        if step % trainer.args.save_steps == 0:
            trainer.save_checkpoint(step)

    # Final log of total examples and time
    total_time = time.time() - start_time
    if logging and trainer.args.local_rank == 0:
        print(f"Training completed. Total examples processed (master): {total_examples_processed}.")
        print(f"Total time for training: {total_time:.2f} seconds.")


# -----------------------------
# Utility Functions
# -----------------------------
# Suppress stdout and stderr
@contextmanager
def suppress_stdout_stderr():
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


def train_val_test_split(dataset, train_val_ratio, val_test_ratio, stratify_column=None):
    train_val = dataset.train_test_split(test_size=train_val_ratio, stratify_by_column=stratify_column) if stratify_column is not None else dataset.train_test_split(test_size=train_val_ratio)
    val_test = train_val['test'].train_test_split(test_size=val_test_ratio, stratify_by_column=stratify_column) if stratify_column is not None else train_val['test'].train_test_split(test_size=val_test_ratio)

    final_dataset = DatasetDict({
        'train': train_val['train'],
        'validation': val_test['train'],
        'test': val_test['test']
    })
    return final_dataset

def set_batch_size(gpu, quantized):
    if gpu == "A100":
        if quantized:
            inference_batch_size = 4
            finetuning_batch_size = 1
        else:
            inference_batch_size = 3
            finetuning_batch_size = 1
    elif gpu == "H100":
        if quantized:
            inference_batch_size = 6
            finetuning_batch_size = 2
        else:
            inference_batch_size = 3
            finetuning_batch_size = 1
    else:
        if quantized:
            inference_batch_size = 1
            finetuning_batch_size = 1
        else:
            inference_batch_size = 1
            finetuning_batch_size = 1
    return inference_batch_size, finetuning_batch_size


def log_to_file(log_file, message):
    with open(log_file, 'a') as f:
        f.write(message + '\n')


def log_message(message, log_file, local_rank=0):
    """
    Log the given message both to the log file and console (if in master process).
    """
    with open(log_file, "a") as log:
        log.write(message + "\n")
    
    if local_rank == 0:  # Only print to console if in master process
        print(message)


def save_lora_checkpoint(model, optimizer, scheduler=None, epoch=0, step=0, avg_val_loss=0.0, checkpoint_dir="checkpoints", local_rank=0):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    # Create the checkpoint file name with the avg_val_loss formatted to 4 decimal places
    avg_val_loss_str = f"{avg_val_loss:.4f}"
    checkpoint_path = os.path.join(checkpoint_dir, f"lora_checkpoint_epoch_{epoch}_step_{step}_valloss_{avg_val_loss_str}.pt")
    
    # Save only LoRA modules
    if local_rank == 0:  # Only the main process saves the checkpoint
        model.save_pretrained(checkpoint_path)  # Save only LoRA layers

        # Create the dictionary for the optimizer and (optionally) the scheduler state
        save_dict = {
            'epoch': epoch,
            'step': step,
            'optimizer_state_dict': optimizer.state_dict(),
        }

        # Add scheduler state dict only if the scheduler is not None
        if scheduler is not None:
            save_dict['scheduler_state_dict'] = scheduler.state_dict()

        # Save optimizer and scheduler state
        torch.save(save_dict, os.path.join(checkpoint_path, "optimizer_scheduler.pt"))
        
        print(f"LoRA checkpoint saved at {checkpoint_path}")


def load_lora_checkpoint(model, optimizer, scheduler, checkpoint_path, device):
    # Load the LoRA layers back into the model
    model = model.from_pretrained(checkpoint_path)
    
    # Load optimizer and scheduler states
    optimizer_scheduler_path = os.path.join(checkpoint_path, "optimizer_scheduler.pt")
    checkpoint = torch.load(optimizer_scheduler_path, map_location=device)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    epoch = checkpoint['epoch']
    step = checkpoint['step']
    print(f"LoRA checkpoint loaded: Resuming from epoch {epoch}, step {step}")
    return epoch, step


# Clean and convert strings to lists
def clean_and_convert_to_list(option_str):
    if 'nan' in option_str:
        option_str = option_str.replace('nan', 'None')
    try:
        options = ast.literal_eval(option_str)
        if isinstance(options, list):
            # Remove None from the list
            options = [option for option in options if option is not None]
        return options
    except (ValueError, SyntaxError):
        return []


# Assign safety labels and answers
def format_safety_and_assign_labels(safety_column):
    answer_letters = []
    answers = []

    safety_labels = ['A', 'B']
    safety_descriptions = ['safe', 'vulnerable']

    for safety in safety_column:
        if safety == 'safe':
            answer_letters.append(safety_labels[0])
            answers.append(safety_descriptions[0])
        else:
            answer_letters.append(safety_labels[1])
            answers.append(safety_descriptions[1])

    return answer_letters, answers


# Remove extra output from generated text
def remove_extra_output(text):
    output_keyword = "Answer:"
    output_start = text.find(output_keyword)
    if output_start != -1:
        end_of_output = text.find('\n', output_start)
        if end_of_output != -1:
            return text[:end_of_output].strip()
    return text.strip()


# Isolate the answer from the output
def isolate_output(text):
    output_keyword = 'Answer:'
    correct_answer_keyword = 'The correct answer is '
    answer = text.split(output_keyword)[-1].strip()
    if answer.startswith(correct_answer_keyword):
        output = answer[len(correct_answer_keyword):].strip()
    else:
        output = answer.strip()
    output = re.sub(r'\.$', '', output)
    return output


# Define the prompt template outside the function
prompt_template = """Analyze the following code and determine if it is safe or vulnerable.

### Code Snippet ({language}) ###
{code}

### Instructions ###
Step 1. Carefully read the code snippet to understand its structure and functionality.
Step 2. Focus on key functions or operations that could introduce vulnerabilities or unsafe behavior.
Step 3. Ensure that your answer is either "safe" or "vulnerable" and output it in the correct format.

### Answer ###
Answer:"""


def create_shifted_prompts(examples):
    prompts = {
        'inputs': [
            f"{prompt_template.format(code=c, language=l)} {a}" for c, l, a in zip(examples['code'], examples['language'], examples['answer'])
        ]
    }
    return prompts


# Define the function to create prompts
def create_label_prompts(examples):
    prompts = {
        'inputs': [
            prompt_template.format(code=c, language=l) for c, l in zip(examples['code'], examples['language'])
        ],
        'labels': [
            f" {a}" for a in examples['answer']
        ]
    }
    return prompts


def create_completion_prompts(examples):
    prompts = {
        'inputs': [
            prompt_template.format(code=c, language=l) for c, l in zip(examples['code'], examples['language'])
        ],
        'labels': [
            f"{prompt_template.format(code=c, language=l)} {a}" for c, l, a in zip(examples['code'], examples['language'], examples['answer'])
        ]
    }
    return prompts


def tokenize_and_shift_prompts(inputs, tokenizer, max_length=1024):
    """
    Tokenizes the inputs and creates labels by shifting the tokenized inputs
    for causal language modeling (CLM).
    """
    # Step 1: Tokenize the inputs (these will serve as both inputs and labels)
    model_inputs = tokenizer(inputs, padding='max_length', max_length=max_length, truncation=True)

    # Step 2: Create labels by shifting the input tokens
    shifted_labels = []
    
    for input_ids in model_inputs["input_ids"]:
        # Shift the input to create the labels (next token prediction)
        shifted_label = input_ids[1:] + [tokenizer.pad_token_id]  # Remove the first token, pad at the end
        shifted_labels.append(shifted_label)

    # Step 3: Assign the shifted labels to the model inputs
    model_inputs["labels"] = shifted_labels

    return model_inputs


# Tokenize prompts
def tokenize_prompts(inputs, labels, tokenizer, max_length=1024, max_label_length=None):
    if max_label_length is None:
        max_label_length = max_length
    model_inputs = tokenizer(inputs, padding='max_length', max_length=max_length, truncation=True)
    model_labels = tokenizer(labels, padding='max_length', max_length=max_label_length, truncation=True)

    model_inputs["labels"] = model_labels["input_ids"]
    return model_inputs


def longest_tokenization(dataset, tokenizer):
    def tokenize(examples):
        model_labels = tokenizer(examples['labels'], padding='longest', return_tensors='pt', truncation=True)
        return model_labels
    tokenized_dataset = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)
    max_length = max(len(input_ids) for input_ids in tokenized_dataset['input_ids'])

    return max_length


def label_preprocess_function(examples, tokenizer, max_length=1024, max_label_length=None):
    if max_label_length is None:
        max_label_length = max_length
    label_prompts = create_label_prompts(examples)
    model_inputs = tokenize_prompts(label_prompts['inputs'], label_prompts['labels'], tokenizer, max_length, max_label_length)
    return model_inputs


def completion_preprocess_function(examples, tokenizer, max_length=1024):
    label_prompts = create_completion_prompts(examples)
    model_inputs = tokenize_prompts(label_prompts['inputs'], label_prompts['labels'], tokenizer, max_length)
    return model_inputs


def completion_preprocess_function_with_shifting(examples, tokenizer, max_length=1024):
    shifted_prompts = create_shifted_prompts(examples)
    model_inputs = tokenize_and_shift_prompts(shifted_prompts['inputs'], tokenizer, max_length)
    return model_inputs


# Define the function to filter examples
def filter_examples(example, tokenizer, max_length=1024):
    # Format the input prompt using the external template
    input_prompt = prompt_template.format(
        code=example['code'],
        language=example['language']
    )

    # Tokenize the input prompt
    tokenized_input = tokenizer(input_prompt, truncation=False, padding=False)

    # Return True if the tokenized input fits within the max length, False otherwise
    return len(tokenized_input['input_ids']) <= max_length


# Define the function to filter examples based on the input prompt and answer length
def filter_examples_with_answer(example, tokenizer, max_length=1024):
    # Format the input prompt using the external template and include the answer
    input_prompt = prompt_template.format(
        code=example['code'],
        language=example['language']
    )
    
    # Combine the input prompt and the answer into a single string
    combined_prompt = f"{input_prompt} {example['answer']}"

    # Tokenize the combined input prompt and answer
    tokenized_input = tokenizer(combined_prompt, truncation=False, padding=False)

    # Return True if the combined tokenized input fits within the max length, False otherwise
    return len(tokenized_input['input_ids']) <= max_length


def create_classlabel_column(dataset, original_column):
    """
    Create a ClassLabel column for stratification from a string-based column.
    """
    # Define the mapping from string labels to integers
    label_mapping = {"safe": 0, "vulnerable": 1}

    # Create a new ClassLabel feature
    safety_classlabel = ClassLabel(num_classes=2, names=["safe", "vulnerable"])

    # Map the original string values to integers
    dataset = dataset.map(lambda example: {"safety_label": label_mapping[example[original_column]]})

    # Cast the new column as ClassLabel
    dataset = dataset.cast_column("safety_label", safety_classlabel)

    return dataset

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # Example metric: accuracy
    accuracy = (predictions.argmax(axis=-1) == labels).mean()
    return {"accuracy": accuracy}

# Function to select a random subset of the evaluation dataset
def get_random_eval_subset(eval_dataset, max_eval_samples):
    # Shuffle the dataset with a random seed
    shuffled_dataset = eval_dataset.shuffle(seed=random.randint(0, 1000))
    # Select the first 'max_eval_samples' examples from the shuffled dataset
    return shuffled_dataset.select(range(max_eval_samples))

# Custom Trainer class with conditional metric computation and accuracy printing
class MainTrainer(SFTTrainer):
    def __init__(self, *args, max_eval_samples=None, eval_metrics_every_n_steps=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_eval_samples = max_eval_samples
        self.eval_metrics_every_n_steps = eval_metrics_every_n_steps
        self.num_steps = 0  # To keep track of how many evaluations have been performed

    def evaluate(self, eval_dataset=None, **kwargs):

        self.num_steps += self.args.eval_steps

        # Select a random subset of the evaluation dataset
        eval_dataset = get_random_eval_subset(self.eval_dataset, self.max_eval_samples) if self.max_eval_samples is not None else self.eval_dataset

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

# ====== Custom Trainer Class ======
class HyperparameterFinetuningTrainer(SFTTrainer):
    def __init__(self, *args, max_eval_samples=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_eval_samples = max_eval_samples

    def evaluate(self, eval_dataset=None, **kwargs):
        eval_dataset = get_random_eval_subset(self.eval_dataset, self.max_eval_samples) if self.max_eval_samples is not None else self.eval_dataset
        result = super().evaluate(eval_dataset=eval_dataset, **kwargs)
        return result