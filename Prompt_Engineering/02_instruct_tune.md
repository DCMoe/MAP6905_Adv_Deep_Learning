# Module 2: LLM Training and Instruction Tuning

## Module Overview

Welcome to Module 2 of the Prompt Engineering curriculum! This module demystifies the inner workings of large language models (LLMs), focusing on their training processes and how these underpin effective prompting. By understanding pre-training, fine-tuning, and instruction tuning, you'll see why prompt design isn't arbitrary—it's a direct extension of how models learn to interpret and respond to human-like inputs. This foundation will enhance your ability to craft prompts that align with model behaviors, reducing issues like hallucinations or off-topic responses.

Our objectives are threefold: First, trace the LLM lifecycle from pre-training to alignment. Second, explore instruction tuning's role in making models prompt-responsive. Third, analyze training implications for prompt strategies, with ties to Module 1's basics and previews of advanced techniques in later modules. This module is designed for self-paced learning, allowing you to progress at your own speed.

## Learning Objectives

By the end of this module, you should be able to:
- Describe the stages of LLM training, including pre-training, supervised fine-tuning, and RLHF [1].
- Explain how instruction tuning uses datasets like Alpaca or FLAN to enable models to follow prompts [2].
- Analyze why training processes (e.g., next-token prediction) make techniques like role-playing or specificity effective in prompts.
- Identify limitations of training, such as biases or forgetting, and their impact on prompt reliability [1].
- Experiment with base vs. tuned models to observe differences in prompt responses.

---

## LLM Training Stages

LLMs undergo a multi-stage training pipeline to evolve from raw text predictors to instruction-following systems [1]. Pre-training exposes models to vast corpora (e.g., trillions of tokens from books, web pages) via next-token prediction, building broad knowledge but lacking task-specific guidance. Supervised fine-tuning (SFT) refines this on curated datasets, while instruction tuning and RLHF align outputs with human preferences—directly influencing why prompts must be clear and contextual.

---

## Pre-Training: Building the Foundation

Pre-training is the unsupervised phase where models learn language patterns through masked or next-token objectives on massive, diverse data [1]. For example, GPT-like models predict the next word in sequences, capturing grammar, facts, and semantics. This stage equips models with emergent abilities but doesn't teach instruction-following. Prompts here often yield generic or incoherent responses, as seen in base models like GPT-2.

---

## Supervised Fine-Tuning and Instruction Tuning

SFT adapts pre-trained models on labeled data, with instruction tuning (a subset) using (instruction, response) pairs from datasets like Alpaca (52K synthetic examples) or FLAN (multi-task benchmarks) [2]. Models learn to map prompts to desired outputs, enabling zero/few-shot learning. This is why role-playing (e.g., "Act as a historian") works—tuning exposes models to persona adoption. However, over-reliance on synthetic data can amplify biases from the base corpus.

---

## Reinforcement Learning from Human Feedback (RLHF)

RLHF builds on SFT by using human rankings to train a reward model, then optimizing via PPO (Proximal Policy Optimization) [3]. Pioneered in InstructGPT, it reduces harmful outputs and boosts helpfulness, making models more robust to ambiguous prompts. Yet, it introduces challenges like reward hacking, where models exploit shortcuts rather than truly understanding intent.

---

## Why Training Matters for Prompting

Training bridges raw prediction to human-aligned generation: Pre-training provides knowledge, SFT/instruction tuning adds structure, and RLHF ensures safety [1]. Specificity in prompts leverages tuning's emphasis on constraints, while chain-of-thought (Module 4) exploits learned step-by-step reasoning. Limitations include catastrophic forgetting (losing pre-training knowledge) and data biases, which can make prompts unreliable—always validate outputs critically.

---

## Suggested Reading & Viewing

1. Zhang, S., et al. (2025). _Instruction Tuning for Large Language Models: A Survey_. [arXiv link](https://arxiv.org/abs/2308.10792)
2. Longpre, S., et al. (2023). _FLAN: A Survey of Instruction Tuning Datasets_. [arXiv link](https://arxiv.org/abs/2301.13688)
3. Ouyang, L., et al. (2022). _Training Language Models to Follow Instructions with Human Feedback_. [arXiv link](https://arxiv.org/abs/2203.02155)
4. How Large Language Models Work (YouTube, IBM Technology). [YouTube link](https://www.youtube.com/watch?v=5sLYAQS9sWQ)
5. Google's 9 Hour AI Prompt Engineering Course In 20 Minutes (YouTube, Tina Huang). [YouTube link](https://www.youtube.com/watch?v=p09yRj47kNM)
6. Hugging Face: Fine-Tuning LLMs Guide. [Hugging Face link](https://huggingface.co/docs/transformers/en/training)

---

## Suggested Activities (Optional)

- Review a public instruction-tuning dataset (e.g., Alpaca on Hugging Face) and adapt 5 examples into prompts; test them on a base vs. tuned model [2].
- Compare responses from a pre-trained model (e.g., GPT-2) and an instruction-tuned one (e.g., Llama-3.2-3B-Instruct) to the same ambiguous prompt; note differences in alignment [3].
- Reflect on a bias example from readings and craft a prompt that mitigates it; evaluate for reliability [1].