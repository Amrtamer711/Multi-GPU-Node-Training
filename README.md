# Distributed Training with LoRA Fine-Tuning for Causal Language Models
This repository contains a Python script designed to fine-tune causal language models, such as LLaMA, using Parameter Efficient Fine-Tuning (PEFT) techniques like LoRA and DoRA. It contains some optimzations such as gradient checkpointing. It supports distributed training via PyTorch's Distributed Data Parallel (DDP) and includes tools for dataset handling, evaluation, and inference.

## Features

- **Distributed Training (DDP)**: Easily scale training across multiple GPUs using torch.distributed.
- **LoRA/DoRA Fine-Tuning**: Use PEFT methods to efficiently fine-tune large language models with minimal trainable parameters.
- **Quantization Support**: Optimize memory usage with BitsAndBytesConfig for 4-bit quantization.
- **Custom Tokenization**: Tokenize and preprocess datasets for efficient causal language modeling.
- **Gradient Checkpointing**: Reduce VRAM usage during training.
- **Evaluation and Inference**: Comprehensive tools for validation, testing, and collecting results with metrics like accuracy and F1 score.
- **Custom Training and Dataloaders**: Fine-tuned dataloader creation for tokenized and non-tokenized datasets, and an adaptable training loop.

## Requirements
- ```torch```
- ```transformers```
- ```datasets```
- ```peft```
- ```torch.distributed```
- ```scikit-learn```
- ```huggingface_hub (if you are using a private model)```

# Usage
### 1. Prepare Datasets
Load your dataset and use ```dataset_prep.ipynb``` to prepare the dataset by tokenizing it

### 2. Run 
Run the training using ```torchrun -standalone -nproc_per_node=x main.py``` where x is the amount of GPUs you have

# Future Work

Currently working on improving the efficiency of the models by exploring more VRAM efficient techniques.
