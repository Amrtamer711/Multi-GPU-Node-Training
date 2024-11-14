# Distributed Training with LoRA Fine-Tuning for Causal Language Models
This repository contains a Python script designed to fine-tune causal language models, such as LLaMA, using Parameter Efficient Fine-Tuning (PEFT) techniques like LoRA and DoRA. It supports distributed training via PyTorch's Distributed Data Parallel (DDP) and includes tools for dataset handling, evaluation, and inference.

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
