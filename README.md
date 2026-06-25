# MiniGPT - Transformer Language Model Built From Scratch

## Overview

MiniGPT is a character-level Transformer language model built entirely from scratch using PyTorch.

The project implements the core ideas behind modern Large Language Models (LLMs), including:

* Character Tokenization
* Multi-Head Self-Attention
* Transformer Blocks
* Positional Embeddings
* Autoregressive Text Generation
* Checkpoint Saving and Loading
* Streaming Dataset Training

The model was trained on the TinyStories dataset and is capable of generating coherent short stories from text prompts.

---

## Features

### Transformer Architecture

* Multi-Head Self-Attention
* Residual Connections
* Layer Normalization
* Feed Forward Networks
* Positional Embeddings

### Training Pipeline

* Streaming Dataset Loader
* Memory Efficient Training
* Checkpoint Saving
* CPU Compatible Training

### Text Generation

* Temperature Sampling
* Top-K Sampling
* Autoregressive Generation

---

## Model Configuration

```python
VOCAB_SIZE = 230

BLOCK_SIZE = 64

BATCH_SIZE = 32

EMBED_DIM = 128

NUM_HEADS = 4

NUM_LAYERS = 4

DROPOUT = 0.1

LEARNING_RATE = 3e-4

EPOCHS = 5
```

---

## Dataset

Training Dataset:

TinyStories

The dataset contains millions of short stories designed for language model training.

Dataset size used:

* 3.6 GB
* JSON Story Files
* Converted into a streaming text corpus

---

## Training Results

Training was performed on:

* CPU Only
* 16 GB RAM
* 512 GB SSD

Example Training Loss:

```text
Step 0       Loss 4.51
Step 100     Loss 2.44
Step 1000    Loss 1.80
Step 10000   Loss 1.20
Step 50000   Loss 1.00
Step 90000   Loss 0.98
```

---

## Example Generation

Prompt:

```text
Once upon a time
```

Output:

```text
Once upon a time, there was a little girl named Lily.
She loved to play with her friends.
One day, Max saw a lot of noise...
```

---

## Project Structure

```text
MiniGPT/

├── data/
├── checkpoints/
├── tokenizer.py
├── dataset.py
├── attention.py
├── transformer.py
├── model.py
├── train.py
├── generate.py
├── config.py
├── requirements.txt
└── README.md
```

---

## Installation

```bash
git clone <repository-url>

cd MiniGPT

python -m venv venv

venv\Scripts\activate

pip install -r requirements.txt
```

---

## Training

```bash
python train.py
```

---

## Generate Text

```bash
python generate.py
```

Example prompt:

```text
Once upon a time
```

---

## Future Improvements

* Byte Pair Encoding (BPE)
* SentencePiece Tokenization
* Validation Dataset
* Learning Rate Scheduler
* Mixed Precision Training
* Streamlit Web Interface
* GPT-2 Style Architecture
* Fine-Tuning Support

---

## Learning Outcomes

This project demonstrates understanding of:

* Deep Learning
* Natural Language Processing
* Transformer Architecture
* PyTorch
* Language Model Training
* Attention Mechanisms
* Model Inference Pipelines

---

## Author

Khurshid Alam

AI / Machine Learning Engineer

Built as a hands-on implementation of Transformer-based Language Models using PyTorch.
