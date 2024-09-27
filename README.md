# Modularized GPT Training Repository

This repository is a refactor of Andrej Karpathy's [minGPT](https://github.com/karpathy/build-nanogpt). The codebase has been modularized to simplify implementation and execution, making it easier to run experiments with smaller GPT models.

## Features
- **Modular Code Structure**: The code is reorganized into smaller, reusable components for flexibility.
- **Single & Multi-GPU Training**: Supports both single and multi-GPU setups.

## Getting Started

### Prerequisites
- Python 3.x
- [PyTorch](https://pytorch.org/get-started/locally/)
- CUDA (for GPU support)


For running on a single GPU
```bash

python train_script.py
```

To run on "n" GPUs

```bash

torchrun --standalone --nproc_per_node=n train_script.py
