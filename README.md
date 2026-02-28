# ZeRO Optimizations & LoRA

This repository explores and implements memory-efficient fine-tuning techniques for Large Language Models (LLMs). By combining **Zero Redundancy Optimizer (ZeRO)** techniques with **Low-Rank Adaptation (LoRA)**, this project demonstrates how to scale model training and inference while drastically reducing GPU memory footprints.

## 🚀 Overview

Fine-tuning massive parameter models often hits memory walls on standard hardware. This repository provides scripts and experiments leveraging:
* **DeepSpeed ZeRO (Stages 1, 2, and 3):** To partition optimizer states, gradients, and model parameters across distributed devices.
* **LoRA (Low-Rank Adaptation):** To freeze pre-trained model weights and inject trainable rank decomposition matrices into each layer of the Transformer architecture, significantly reducing the number of trainable parameters.

## 📂 Repository Structure

```text
├── config/                   # Config files for different ZeRO Optimization levels
│   ├── ds_z1.yaml     
│   ├── ds_z2.yaml      
│   └── ds_z3.yaml             
├── scripts/                  # Execution scripts for various training/profiling runs
│   ├── baseline.sh
│   ├── lora.sh
│   └── zero.sh
├── lora.py                   # Core script for applying LoRA
├── main.py                   # Main execution/training script
└── README.md                 # Project documentation

```

## 📚 References
If you want to dive deeper into the theory behind these optimization techniques, check out the original research papers:
* **ZeRO**: [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054)
* **LORA**: [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
