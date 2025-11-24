---
title: Hands-on Demonstrations
---

## Learning Objectives

By the end of this module, you will be able to:
- Load and use pre-trained VLMs from Hugging Face
- Perform image captioning and visual question answering
- Prepare custom datasets for VLM fine-tuning
- Configure and apply LoRA for parameter-efficient fine-tuning
- Train and evaluate fine-tuned VLMs
- Deploy VLMs for inference with best practices

---

## VLM Inference Notebook: Using Pre-trained VLMs

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DCMoe/MAP6905_Adv_Deep_Learning/blob/main/notebooks/vlm_inference.ipynb)

### Purpose
This notebook demonstrates how to use pre-trained VLMs for various tasks **without any training**. You'll learn to leverage existing models for immediate applications.

### Topics Covered

#### 1. Environment Setup
- Installing required libraries (`transformers`, `accelerate`, `pillow`)
- GPU configuration and memory management
- Loading models efficiently

#### 2. Image Captioning with BLIP-2
- Loading BLIP-2 model and processor
- Preparing images for inference
- Generating captions with different decoding strategies:
  - Greedy decoding
  - Beam search
  - Nucleus sampling
- Comparing caption quality
- Visual Question Answering

---

## VLM LoRA Notebook: Fine-tuning VLMs with LoRA

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DCMoe/MAP6905_Adv_Deep_Learning/blob/main/notebooks/vlm_lora.ipynb)

### Purpose

This notebook shows you how to **quickly and efficiently fine-tune Google’s Gemma-3 (4B multimodal)** model on a visual question answering (VQA) task using **LoRA + SFTTrainer** — all in a single Colab runtime (T4/A100).

### Video Resource

Before or during this notebook, watch this tutorial on fine-tuning with LoRA:

<iframe width="560" height="315" src="https://www.youtube.com/embed/3ypHZayanBI?si=ivB8xpzYlSAcUtA7" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

:::{admonition} Video Guide
:class: note
The video provides additional context on:
- LoRA fundamentals and intuition
- Practical tips for hyperparameter selection
- Common pitfalls and how to avoid them
- Real-world fine-tuning examples
:::

### Topics Covered

#### 1. Dataset Preparation
- Loading `lmms-lab/VQAv2` (or `VQAv2_TOY` if available)
- Converting image + question → Gemma-3 chat template
- Tiny train/validation split (60 examples total)

#### 2. Model & Quantization
- Loading **Gemma-3-4B-IT** in 4-bit (fits on free Colab)
- Using `BitsAndBytesConfig` + `device_map="auto"`

#### 3. LoRA Configuration
- Simple `LoraConfig` (r=16, target all linear layers)
- Applied automatically via `peft_config` in SFTTrainer

#### 4. Training with SFTTrainer
- One-liner training using `trl.SFTTrainer`
- Proper label masking (ignore padding + image tokens)
- Gradient checkpointing + bf16

---
