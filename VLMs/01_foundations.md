---
title: VLM Foundations & Architecture Overview
---

## Learning Objectives

By the end of this module, you will be able to:
- Understand the mathematical foundations of attention mechanisms and transformers
- Explain the evolution from unimodal to multimodal AI systems with theoretical grounding
- Analyze key VLM architectures and their training objectives
- Identify and explain the core components of VLMs with mathematical formulations
- Recognize fundamental challenges in vision-language understanding

---

## Prerequisites Review

### Convolutional Neural Networks (CNNs)

Convolutional Neural Networks form the foundation of traditional computer vision. A convolution operation can be mathematically expressed as:

$$
(f * g)(x, y) = \sum_{m}\sum_{n} f(m, n) \cdot g(x-m, y-n)
$$

where $f$ is the input image and $g$ is the filter/kernel.

**Key Concepts:**

1. **Convolutional Layers**: Apply learnable filters to extract local features. For an input $\mathbf{X} \in \mathbb{R}^{H \times W \times C}$ (height, width, channels) and a filter $\mathbf{W} \in \mathbb{R}^{k \times k \times C \times D}$:

$$
\mathbf{Y}_{i,j,d} = \sigma\left(\sum_{c=1}^{C}\sum_{m=0}^{k-1}\sum_{n=0}^{k-1} \mathbf{W}_{m,n,c,d} \cdot \mathbf{X}_{i+m,j+n,c} + b_d\right)
$$

where $\sigma$ is a non-linear activation function (e.g., ReLU) and $b_d$ is a bias term.

2. **Pooling Layers**: Reduce spatial dimensions while preserving important features. Max pooling over a $p \times p$ region:

$$
\mathbf{Y}_{i,j,c} = \max_{m,n \in [0,p)} \mathbf{X}_{pi+m,pj+n,c}
$$

3. **Feature Hierarchies**: Early layers detect simple patterns (edges, textures), while deeper layers learn complex, compositional features through successive convolutions.

**Limitations for Multimodal Learning:**
- Fixed receptive fields limit global context understanding
- Require many layers to capture long-range dependencies
- Lack explicit mechanism for cross-modal alignment

### Transformers and Attention Mechanisms

As discussed in the LLM introduction module, the transformer architecture {cite}`vaswani2017attention` revolutionized sequence modeling through self-attention mechanisms.

#### Self-Attention Mechanism

Given an input sequence $\mathbf{X} \in \mathbb{R}^{n \times d}$ with $n$ tokens of dimension $d$, self-attention computes:

1. **Linear Projections**: Create query ($\mathbf{Q}$), key ($\mathbf{K}$), and value ($\mathbf{V}$) matrices:

$$
\mathbf{Q} = \mathbf{X}\mathbf{W}_Q, \quad \mathbf{K} = \mathbf{X}\mathbf{W}_K, \quad \mathbf{V} = \mathbf{X}\mathbf{W}_V
$$

where $\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V \in \mathbb{R}^{d \times d_k}$ are learnable weight matrices.

2. **Attention Scores**: Compute similarity between queries and keys:

$$
\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V}
$$

The scaling factor $\sqrt{d_k}$ prevents the dot products from becoming too large, which would push the softmax into regions with extremely small gradients.

**Intuition**: Each token attends to all other tokens, with attention weights determined by the similarity between queries and keys. The output is a weighted sum of values.

3. **Multi-Head Attention**: Allows the model to attend to different representation subspaces:

$$
\text{MultiHead}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)\mathbf{W}_O
$$

where each head is:

$$
\text{head}_i = \text{Attention}(\mathbf{Q}\mathbf{W}^Q_i, \mathbf{K}\mathbf{W}^K_i, \mathbf{V}\mathbf{W}^V_i)
$$

**Why Multiple Heads?** Different heads can learn to attend to different aspects: syntactic relationships, semantic relationships, positional information, etc.

#### Positional Encodings

Since transformers have no inherent notion of sequence order, positional information must be injected. The original transformer uses sinusoidal encodings:

$$
\begin{align}
\text{PE}_{(pos, 2i)} &= \sin\left(\frac{pos}{10000^{2i/d}}\right) \\
\text{PE}_{(pos, 2i+1)} &= \cos\left(\frac{pos}{10000^{2i/d}}\right)
\end{align}
$$

where $pos$ is the position and $i$ is the dimension. These encodings have the property that $\text{PE}_{pos+k}$ can be expressed as a linear function of $\text{PE}_{pos}$, allowing the model to learn relative positions.

**Computational Complexity**: Self-attention has $O(n^2 \cdot d)$ complexity, where $n$ is the sequence length. This becomes problematic for long sequences or high-resolution images.

---

## Evolution from Unimodal to Multimodal Models

### The Unimodal Era: Separate Modalities

Traditional AI systems processed vision and language independently:

**Vision Models**:
- Trained on labeled image datasets (ImageNet, COCO)
- Fixed ontologies: $f: \mathcal{X} \rightarrow \mathcal{Y}$

**Language Models**:
- Trained on massive text corpora
- Next-token prediction objective {cite}`radford2019language,brown2020language`

### The Multimodal Revolution: Joint Modeling

<iframe width="560" height="315" src="https://www.youtube.com/embed/lOD_EE96jhM?si=Tc3peoEL2EckTE9t" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

<p></p>

Vision-Language Models (VLMs) learn joint representations of images and text. Two dominant paradigms emerged:

#### 1. Contrastive Learning (CLIP-style)

**Core Idea**: Align image and text embeddings in a shared space using contrastive loss {cite}`radford2021learning`.

<iframe width="560" height="315" src="https://www.youtube.com/embed/KcSXcpluDe4?si=NTzequl6IGyXz7on" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

<p></p>

**Training Objective** (InfoNCE):

$$
\mathcal{L}_{i2t} = -\frac{1}{B}\sum_{i=1}^B \log \frac{\exp(s(i,i)/\tau)}{\sum_{j=1}^B \exp(s(i,j)/\tau)}
$$

where $s(i,j) = f(x_i)^T g(t_j)$ is the cosine similarity, $\tau$ is temperature.

Symmetric text-to-image loss $\mathcal{L}_{t2i}$ → total loss $\mathcal{L} = \frac{1}{2}(\mathcal{L}_{i2t} + \mathcal{L}_{t2i})$.

**Key Properties**:
- Zero-shot transfer: Classify images using text prompts
- Robust to distribution shift
- No generation capability

#### 2. Generative Modeling (LLaVA-style)

**Core Idea**: Project image features into LLM token space and train with language modeling objective {cite}`liu2023visual`.

**Two-Stage Training**:

**Stage 1**: Pre-train vision encoder + projection on image-caption pairs  
**Stage 2**: Instruction tuning with diverse multimodal instruction-following data

**Training Objective**: Standard next-token prediction over concatenated vision+text sequences.

**Key Properties**:
- Natural language generation about images
- Instruction following
- Reasoning capabilities

---

## Core VLM Architectures

### CLIP: Contrastive Language-Image Pretraining {cite}`radford2021learning`

```
Image Encoder (ViT) → Image Features
Text Encoder (Transformer) → Text Features
→ Contrastive Alignment in Shared Space
```

**Vision Transformer (ViT)** {cite}`dosovitskiy2021image`:
- Split image into 16×16 patches
- Linear projection + positional encoding
- Standard transformer encoder
- Complexity: $O(N^2 \cdot D)$ where $N = \frac{HW}{P^2}$

### LLaVA: Large Language and Vision Assistant {cite}`liu2023visual`

```
Vision Encoder (CLIP-ViT) → Image Features
→ Projection Layer → LLM Token Space
→ Large Language Model (Vicuna/LLaMA)
```

**Projection Strategies**:
- Linear layer
- MLP with non-linearities
- Q-Former (learned queries)

---

## Fundamental Challenges in Vision-Language Models

1. **The Alignment Problem**
   - Vision: continuous, high-dimensional
   - Language: discrete, symbolic
   - How do we bridge this modality gap?

2. **The Grounding Problem**
   - Which visual regions correspond to which language tokens?
   - Requires fine-grained alignment beyond image-level

3. **Hallucination**
   - Models generate plausible but factually incorrect content
   - Especially problematic in medical/legal domains

4. **Compositionality**
   - Understanding novel combinations of known concepts
   - "The red cube on the blue sphere" vs. training examples

5. **Scalability**
   - Quadratic complexity of self-attention:  
     $\text{Complexity}_{\text{ViT}} = O\left(\left(\frac{HW}{P^2}\right)^2 \cdot D\right)$

**Mitigation Strategies**:
- **Efficient Attention**: Linear attention, sparse attention, flash attention
- **Hierarchical Processing**: Multi-scale representations
- **Quantization**: Reduce precision (INT8, INT4)
- **Parameter-Efficient Fine-Tuning**: LoRA, adapters (covered in Module 3)
- **Mixed-Resolution**: Process high-res only when needed

---

## Module Summary

### Key Takeaways

**Mathematical Foundations**:
- Self-attention mechanism: $\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}(\mathbf{Q}\mathbf{K}^T/\sqrt{d_k})\mathbf{V}$
- Vision Transformer: Treats images as sequences of patches
- Contrastive learning: Aligns modalities through symmetric cross-entropy

**Architectural Insights**:
- **CLIP**: Dual encoders + contrastive learning → excellent zero-shot transfer
- **LLaVA**: Vision encoder + projection + LLM → instruction-following capabilities
- Two main fusion strategies: contrastive (CLIP) vs. generative (LLaVA)

**Core Components**:
1. **Vision Encoder**: ViT with patch embeddings and self-attention
2. **Language Model**: Autoregressive transformers for generation
3. **Fusion Mechanism**: Contrastive learning or projection-based alignment

**Fundamental Challenges**:
1. **Alignment**: Bridging continuous vision and discrete language
2. **Grounding**: Connecting language to specific visual regions
3. **Hallucination**: Generating factually incorrect descriptions
4. **Compositionality**: Understanding attribute-object-relation compositions
5. **Scalability**: Balancing model size, data size, and computational cost

### Critical Numbers to Remember

- **CLIP**: 400M training pairs, 76.2% zero-shot ImageNet top-1
- **ViT-B/16**: 86M parameters, 224×224 input → 196 patches
- **LLaVA**: 595K+158K training samples, 92.53% on Science QA
- **Attention Complexity**: $O(n^2 \cdot d)$ where $n$ is sequence length

---

### Additional Reading

**Foundational Papers**:
- {cite}`vaswani2017attention`. "Attention Is All You Need." NeurIPS.
- {cite}`dosovitskiy2021image`. "An Image Is Worth 16x16 Words: Transformers for Image Recognition at Scale." ICLR.
- {cite}`radford2021learning`. "Learning Transferable Visual Models From Natural Language Supervision." ICML.
- {cite}`liu2023visual`. "Visual Instruction Tuning." NeurIPS.

**Surveys and Tutorials**:
- [Hugging Face: Vision Language Models Explained](https://huggingface.co/blog/vlms)
- [NVIDIA: What are Vision-Language Models?](https://www.nvidia.com/en-us/glossary/vision-language-models/)
- [A Dive into Vision-Language Models](https://huggingface.co/blog/vision_language_pretraining)

**Interactive Resources**:
- [Hugging Face Model Hub: VLMs](https://huggingface.co/models?pipeline_tag=image-text-to-text)
- [Vision Arena Leaderboard](https://huggingface.co/spaces/WildVision/vision-arena)
- [Open VLM Leaderboard](https://huggingface.co/spaces/opencompass/open_vlm_leaderboard)

---

## References

```{bibliography}
:style: unsrt
:filter: False

vaswani2017attention
radford2019language
brown2020language
radford2021learning
dosovitskiy2021image
liu2023visual
```

