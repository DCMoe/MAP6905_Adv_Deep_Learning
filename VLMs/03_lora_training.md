---
title: Parameter-Efficient Fine-Tuning & Evaluation
---

## Learning Objectives

By the end of this module, you will be able to:
- Understand the mathematical foundations of Low-Rank Adaptation (LoRA)
- Derive the intrinsic rank hypothesis and its implications
- Implement LoRA for VLM fine-tuning with proper hyperparameter choices
- Evaluate VLMs using standardized benchmarks and metrics
- Interpret evaluation results and identify model strengths/weaknesses
- Select appropriate benchmarks for different VLM capabilities

---

## Part I: Parameter-Efficient Fine-Tuning

### Motivation: The Cost of Full Fine-Tuning

**The Problem**: Modern VLMs contain billions of parameters:
- LLaVA-13B: 13 billion parameters
- GPT-4V: 100B+ parameters (estimated)
- Full fine-tuning requires:
  - Storing gradients: O(|θ|) memory
  - Optimizer states (Adam): 2 × |θ| additional memory
  - Total: approximately 4 × |θ| memory requirement

**Example**: Fine-tuning GPT-3 175B with Adam:
$$
\text{Memory} = 4 \times 175B \times 4\text{ bytes} = 2.8\text{ TB}
$$

This requires hundreds of GPUs, making it **prohibitively expensive** for most researchers and applications.

### The Intrinsic Rank Hypothesis

**Citation**: Aghajanyan et al. (2021). "Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning" {cite}`aghajanyan2021intrinsic`.

**Key Insight**: The weight updates during fine-tuning have low **intrinsic dimensionality**.

**Formal Definition**: Given pre-trained weights $\mathbf{W}_0 \in \mathbb{R}^{d \times k}$, fine-tuning produces:

$$
\mathbf{W} = \mathbf{W}_0 + \Delta \mathbf{W}
$$

**Hypothesis**: $\text{rank}(\Delta \mathbf{W}) \ll \min(d, k)$

**Evidence** {cite}`aghajanyan2021intrinsic`:
- Measured intrinsic dimension $d_{90}$ (dimensions needed to capture 90% of performance)
- Pre-training: $d_{90} \approx$ full model size
- Fine-tuning: $d_{90} \ll$ full model size
- **Finding**: Can achieve 90% of full fine-tuning performance by optimizing in a subspace of dimension < 1000, even for models with > 100M parameters!

This motivates low-rank adaptation methods.

---

## 1. LoRA: Low-Rank Adaptation

**Citation**: Hu et al. (2021). "LoRA: Low-Rank Adaptation of Large Language Models" {cite}`hu2021lora`.

### Core Principle

Instead of updating the full weight matrix $\mathbf{W} \in \mathbb{R}^{d \times k}$, represent the update as a low-rank decomposition:

$$
\mathbf{W} = \mathbf{W}_0 + \Delta \mathbf{W} = \mathbf{W}_0 + \mathbf{B}\mathbf{A}
$$

where:
- $\mathbf{W}_0 \in \mathbb{R}^{d \times k}$: Frozen pre-trained weights
- $\mathbf{B} \in \mathbb{R}^{d \times r}$: Trainable down-projection
- $\mathbf{A} \in \mathbb{R}^{r \times k}$: Trainable up-projection
- $r \ll \min(d, k)$: Rank (typically $r \in [4, 64]$)

### Forward Pass

For input $\mathbf{x} \in \mathbb{R}^k$:

$$
\mathbf{h} = \mathbf{W}_0\mathbf{x} + \Delta \mathbf{W} \mathbf{x} = \mathbf{W}_0\mathbf{x} + \mathbf{B}\mathbf{A}\mathbf{x}
$$

**Scaling Factor**: To reduce hyperparameter sensitivity:

$$
\mathbf{h} = \mathbf{W}_0\mathbf{x} + \frac{\alpha}{r}\mathbf{B}\mathbf{A}\mathbf{x}
$$

where $\alpha$ is a constant (typically $\alpha = r$, so scaling is 1).

### Initialization Strategy

**Critical for Training Stability**:

1. **Matrix $\mathbf{A}$**: Random Gaussian initialization
   $$
   A_{ij} \sim \mathcal{N}\left(0, \frac{1}{r}\right)
   $$

2. **Matrix $\mathbf{B}$**: Zero initialization
   $$
   \mathbf{B} = \mathbf{0}_{d \times r}
   $$

**Why?** At initialization:
$$
\Delta \mathbf{W} = \mathbf{B}\mathbf{A} = \mathbf{0} \Rightarrow \mathbf{W} = \mathbf{W}_0
$$

This ensures training starts from the pre-trained model, preserving its capabilities.

### Parameter Reduction

**Trainable Parameters**:
- Full fine-tuning: $d \times k$
- LoRA: $r(d + k)$

**Reduction Factor**:
$$
\text{Reduction} = \frac{dk}{r(d + k)} \approx \frac{d}{r} \quad \text{(when } k \approx d\text{)}
$$

**Example**: For a 4096×4096 weight matrix:
- Full fine-tuning: $4096^2 = 16.8M$ parameters
- LoRA with $r=8$: $8(4096 + 4096) = 65.5K$ parameters
- **Reduction**: $256\times$ fewer parameters!

### Which Layers to Apply LoRA?

**Empirical Finding** {cite}`hu2021lora`:
- Applying LoRA to **attention weights** ($\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V, \mathbf{W}_O$) is most effective
- Applying to MLPs provides marginal additional gains
- **Recommendation**: Apply to attention layers only for best efficiency-performance trade-off

---

## 2. QLoRA: 4-bit Quantized LoRA {cite}`dettmers2023qlora`

**Key Innovations**:
- 4-bit NormalFloat (NF4) quantization
- Double quantization
- Paged optimizers (for CPU offloading)

**Results**:
- Fine-tune 65B model on a single 48GB GPU
- < 1% performance degradation vs. 16-bit

---

## Part II: Evaluation of Vision-Language Models

### Key Benchmarks

| Benchmark   | Task Type           | Year | # Questions | Key Feature                     |
|-------------|---------------------|------|-------------|---------------------------------|
| VQAv2       | Visual QA           | 2017 {cite}`goyal2017vqav2`  | 1.1M        | Real images, diverse questions  |
| GQA         | Compositional QA    | 2019 {cite}`hudson2019gqa`  | 22M         | Balanced, grounded reasoning    |
| MMBench     | Comprehensive eval  | 2023 {cite}`liu2023mmbench`  | 3K+         | CircularEval, 20 capabilities   |
| MMMU        | Expert-level        | 2023 {cite}`yue2023mmmu`     | 11.5K       | College-level, 30+ subjects     |

### Evaluation Metrics

**Classification Tasks**:
- Accuracy
- F1 score

**Generation Tasks**:
- BLEU, CIDEr, SPICE
- BERTScore (semantic similarity)

**OCR-heavy Tasks**:
- ANLS (Average Normalized Levenshtein Similarity)

### Statistical Significance Testing

**Recommended Practice**:
1. Run model 3–5 times with different seeds
2. Compute mean and 95% confidence interval via bootstrapping

**Example**:
- Model A: 85.2% ± 0.4%
- Model B: 85.8% ± 0.5%
- Conclusion: Significant difference (non-overlapping CIs)

### Failure Analysis

**Beyond Aggregate Metrics**: Analyze error patterns

**Breakdown by**:
1. **Question Type**: Which types are hardest?
2. **Image Properties**: Complex vs. simple scenes
3. **Error Categories**:
   - Visual errors: Misidentified objects
   - Reasoning errors: Incorrect inference
   - Knowledge errors: Lack of domain knowledge

**Example Analysis**:
```
MMBench Results for Model X:

Overall: 72.3%

By Category:
- Coarse Perception: 89.2% [check]
- Fine-Grained Perception: 71.4%
- Spatial Reasoning: 58.7% [cross]
- Logical Reasoning: 76.5%

Key Weakness: Spatial relationships
→ Recommend: Fine-tune with spatial reasoning data
```

---

## Module Summary

### Parameter-Efficient Fine-Tuning

**LoRA** {cite}`hu2021lora`:
- Represents updates as low-rank decomposition: $\Delta\mathbf{W} = \mathbf{B}\mathbf{A}$
- Reduces trainable parameters by >100×
- Reduces memory by approximately 4×
- No inference latency (merge adapters)
- Rank $r \in [8,16]$ typically sufficient

**QLoRA** {cite}`dettmers2023qlora`:
- Combines 4-bit quantization with LoRA
- Enables fine-tuning 65B models on single GPU
- Minimal performance degradation

### Evaluation Benchmarks

**VQA Benchmarks**:
- **VQAv2** {cite}`goyal2017vqav2`: Basic visual understanding (1.1M questions)
- **GQA** {cite}`hudson2019gqa`: Compositional reasoning (22M questions)
- **MMBench** {cite}`liu2023mmbench`: Comprehensive evaluation with CircularEval (20 dimensions)

**Expert-Level**:
- **MMMU** {cite}`yue2023mmmu`: College-level knowledge (11.5K questions, 30 subjects)
- 31.8% gap between GPT-4V and human experts

### Critical Insights

1. **Efficiency**: LoRA enables fine-tuning massive models on modest hardware
2. **Rank Selection**: Start with $r=16$, ablate if needed
3. **Multi-Benchmark**: Evaluate on diverse benchmarks for comprehensive assessment
4. **Statistical Rigor**: Report confidence intervals and perform failure analysis
5. **Task Match**: Choose benchmarks aligned with target application

---

### Additional Reading

**Parameter-Efficient Fine-Tuning**:
- {cite}`hu2021lora`. "LoRA: Low-Rank Adaptation of Large Language Models." ICLR.
- {cite}`dettmers2023qlora`. "QLoRA: Efficient Finetuning of Quantized LLMs." NeurIPS.
- Aghajanyan et al. (2021). "Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning."

**Evaluation Benchmarks**:
- {cite}`goyal2017vqav2`. "Making the V in VQA Matter." CVPR.
- {cite}`hudson2019gqa`. "GQA: A New Dataset for Real-World Visual Reasoning." CVPR.
- {cite}`liu2023mmbench`. "MMBench: Is Your Multi-modal Model an All-around Player?" arXiv.
- {cite}`yue2023mmmu`. "MMMU: A Massive Multi-discipline Multimodal Understanding." CVPR.

---

## References

```{bibliography}
:style: unsrt
:filter: False

aghajanyan2021intrinsic
hu2021lora
dettmers2023qlora
goyal2017vqav2
hudson2019gqa
liu2023mmbench
yue2023mmmu
```

