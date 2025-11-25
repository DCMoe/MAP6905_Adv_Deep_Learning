---
title: Introduction to Large Language Models (LLMs)
---

```{note}
**Learning Objectives**

By the end of this module, you will be able to:

- Explain what a Large Language Model (LLM) is and how it processes text.
- Understand the basic architecture of the Transformer.
- Describe how tokens, embeddings, and attention mechanisms interact.
- Run and visualize basic LLM operations in PyTorch.
```



<iframe width="560" height="315" src="https://www.youtube.com/embed/5sLYAQS9sWQ?si=AtFx17RCIfVCg_m7" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>




<iframe width="560" height="315" src="https://www.youtube.com/embed/LPZh9BOjkQs?si=S7eYoUVl3TvOz108" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>




# 1 What is a Large Language Model?

Large Language Models (LLMs) are neural networks trained to model **the probability of text sequences**.  
At their core, they learn how likely a sequence of tokens is, and how to predict the **next token** given previous ones.

```{admonition} Discussion
**Why probability modeling?**  
Next-token prediction converts the open-ended task of “understand language” into a precise statistical objective. When trained at scale, the model implicitly picks up syntax, semantics, factual associations, and task structure as *by-products* of minimizing predictive error {cite}`vaswani2017attention,brown2020language,touvron2023llama`.  

**Measuring quality.**  
Two ubiquitous metrics are **negative log-likelihood** (NLL) and **perplexity**:  

$$
\text{PPL} = \exp\!\left(\frac{1}{T}\sum_{t=1}^{T} -\log P_\theta(w_t \mid w_{<t})\right).
$$

Lower is better; it corresponds to the model assigning higher probability to the observed text. Perplexity correlates with intrinsic modeling quality but not always with downstream utility; evaluations on tasks/benchmarks are also important {cite}`brown2020language`.
```

## 1.1 Problem Setup (Language Modeling)

Let $w_{1:T} = (w_1, w_2, \ldots, w_T)$ represent a sequence of tokens drawn from a fixed vocabulary $\mathcal{V}$ of size $V$.  
Using the chain rule of probability, a language model defines:

$$
P(w_{1:T}) = \prod_{t=1}^{T} P\!\left(w_t \mid w_{<t}\right),
$$

where $w_{<t}=(w_1,\ldots,w_{t-1})$.  

Most LLMs are **autoregressive**, meaning they predict each token one at a time based on the previous context.  
The training objective is to **maximize the log-likelihood** of the observed data:

$$
\max_{\theta}\; \sum_{(w_{1:T})\in \mathcal{D}} \sum_{t=1}^{T}
\log P_{\theta}\!\left(w_t \mid w_{<t}\right),
$$

where $\theta$ are the model parameters (weights and biases of the neural network).

```{admonition} Intuition
:class: tip
Think of the model as a powerful **autocomplete**: given a prefix, it learns to rank all possible next tokens and choose the most likely one—or sample from that probability distribution.
```

```{admonition} Discussion
**Causal masking.**  
During training/inference, attention is masked so each position $t$ can only attend to $w_{\le t}$; this enforces autoregressive factorization.  

**Optimization.**  
In practice, the objective becomes **cross-entropy** loss optimized by first-order methods like Adam/AdamW {cite}`kingma2015adam,loshchilov2019decoupled`, often with warmup + cosine decay schedules.  

**Decoding.**  
Greedy decode maximizes local probability; *stochastic* decoding (top-$k$, nucleus/top-$p$) often yields higher-quality generations. Nucleus sampling selects the smallest set of tokens whose cumulative probability exceeds $p$ {cite}`holtzman2020curious`.
```

## 1.2 From Tokens to Probabilities

LLMs process **tokens**, not raw text.  
Each token $w_t$ is first mapped to an integer index in $\{1,\ldots,V\}$, then to a **vector embedding** through a learned matrix $E \in \mathbb{R}^{V\times d}$:

$$
x_t = E[w_t] \in \mathbb{R}^{d}.
$$

The sequence of embeddings $X = (x_1, x_2, \ldots, x_T)$ is fed into a **Transformer**, which outputs contextualized representations $h_t$ for each token.

A **decoder head** then transforms each hidden state $h_t$ into a set of **logits** over the vocabulary:

$$
z_t = W_o h_t + b_o \in \mathbb{R}^{V},
$$

and applies the **softmax** function to convert these logits into probabilities:

$$
P(w_t = v_i \mid w_{<t})
=
\frac{\exp(z_{t,i})}
{\sum_{j=1}^{V} \exp(z_{t,j})},
$$

where $z_{t,i}$ is the logit corresponding to token $v_i$, and the denominator ensures all probabilities sum to 1.

```{admonition} Key Terms
- **Token:** a discrete unit of text (word, subword, or character).  
- **Vocabulary ($V$)**: set of all tokens known to the model.  
- **Embedding ($x_t$)**: dense vector representing token $w_t$.  
- **Logits ($z_t$)**: unnormalized scores for all tokens.  
- **Softmax:** normalizes logits into a probability distribution.
```

```{admonition} Discussion
**Weight tying.**  
Many LLMs tie the output projection $W_o$ with the embedding matrix $E$ (or its transpose) to reduce parameters and improve perplexity {cite}`press2017using`.  

**Temperature.**  
Sampling temperature $\tau$ rescales logits: $\text{softmax}(z_t/\tau)$.  
Higher $\tau$ → more diverse; lower $\tau$ → more focused.  

**Calibration.**  
Raw softmax probabilities from LLMs are often overconfident; post-hoc calibration methods can adjust them {cite}`guo2017calibration`.
```

```{admonition} Example
:class: note
Suppose our tiny vocabulary is {“a”, “cat”, “sat”, “on”, “the”, “mat”}, and logits for the next token are $[0.2, 1.5, -0.3, 0.8, 2.1, 0.4]$.  

Softmax turns this into probabilities:  
$$
[0.07, 0.25, 0.04, 0.13, 0.46, 0.09],
$$  
suggesting “the” is most likely.
```
```{code-cell} python
import torch
logits = torch.tensor([0.2, 1.5, -0.3, 0.8, 2.1, 0.4])
probs = torch.softmax(logits, dim=-1)
print(probs.round(2))
```
```{code-cell} output
tensor([0.0700, 0.2500, 0.0400, 0.1300, 0.4600, 0.0900])
```

## 1.3 Tokenization: From Text to Indices

Text is split into **tokens** using algorithms like **Byte-Pair Encoding (BPE)** {cite}`sennrich2016neural` or **SentencePiece** {cite}`kudo2018sentencepiece`.  

Tokens are not always words—common phrases become single tokens, rare words split into subwords (e.g., “unhappiness” → “un##”, “happ##”, “iness”).  
This balances vocabulary size (typically 32k–100k) with expressiveness.

```{admonition} Example
:class: note
GPT-2 uses **byte-level BPE** {cite}`radford2019language`, allowing arbitrary text without unknown tokens.  

“The quick brown fox” might tokenize to [464, 2069, 7586, 41151], where each number indexes into the embedding matrix.
```

## 1.4 Embeddings: From Indices to Vectors

Embeddings map discrete tokens to continuous vectors in $\mathbb{R}^d$ (e.g., $d=768$ for BERT-base).  

- **Learned during training.**  
- **Semantically similar tokens cluster together.**  
- **Dimensionality trades off expressivity vs. computation.**

```{code-cell} python
import torch.nn as nn
vocab_size, embed_dim = 10000, 8
embedding_layer = nn.Embedding(vocab_size, embed_dim)
token_ids = torch.tensor([42, 1337])
embeddings = embedding_layer(token_ids)
print(embeddings.shape)
```
```{code-cell} output
torch.Size([2, 8])
```

# 2 The Transformer Architecture

The Transformer {cite}`vaswani2017attention` is a stack of $L$ identical blocks, each with two sub-layers:  

1. **Multi-Head Self-Attention** (context mixing).  
2. **Positionwise Feedforward Network** (nonlinear transformation).  

Each sub-layer is wrapped in a **residual connection** and **layer normalization** {cite}`xiong2020layer`:  

$$
X' = \mathrm{LayerNorm}(X + \mathrm{SubLayer}(X)).
$$

## 2.1 Self-Attention: The Core Mechanism

Self-attention computes weighted averages of all tokens, with weights from pairwise similarities.  

For input $X \in \mathbb{R}^{T \times d}$:  

- Project to queries/keys/values: $Q = XW_Q$, $K = XW_K$, $V = XW_V$.  
- Compute **attention scores**: $A = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)$.  
- Output: $Z = A V$.  

The scaling by $\sqrt{d_k}$ stabilizes gradients.

```{admonition} Intuition
:class: tip
Attention asks: “For each token, which others are relevant?” The matrix $A \in [0,1]^{T\times T}$ encodes pairwise “relevance weights,” allowing dynamic context integration.
```

## 2.2 Multi-Head Attention

Run $H$ attention heads in parallel, concatenate, and project:  

$$
\mathrm{MultiHead}(X) = \mathrm{Concat}(Z_1, \ldots, Z_H) W_O.
$$

Each head learns different relationships (e.g., syntax vs. semantics).  

```{admonition} Discussion
:class: important
**Pre- vs. Post-Norm.**  
Original Transformer uses post-norm; modern variants often pre-norm for stability.  
**Activation.**  
FFNs use GELU/SwiGLU {cite}`shazeer2020glu` over ReLU.  
**Efficient Implementations.**  
FlashAttention {cite}`dao2022flashattention,dao2023flashattention2` fuses operations for speed/memory gains; efficient variants like Performer {cite}`choromanski2021rethinking` approximate softmax.  
**Heads Ablation.**  
Not all heads are equally important—many can be pruned {cite}`michel2019sixteen`.
```

```{code-cell} python
import torch.nn.functional as F
X = torch.rand(1, 5, 8)  # (batch, tokens, dim)
Q = K = V = X  # simplified
attn_scores = F.softmax(Q @ K.transpose(-2,-1) / torch.sqrt(torch.tensor(8.)), dim=-1)
print(attn_scores.shape)
```
```{code-cell} output
torch.Size([1, 5, 5])
```

## 2.3 Feedforward Network

A simple two-layer MLP applied independently to each token:  

$$
\mathrm{FFN}(x) = \sigma(x W_1 + b_1) W_2 + b_2.
$$

Expands to higher dimension (e.g., $4d$) for capacity, then projects back.

# 3 Positional Encoding

Transformers are permutation-invariant, so we add **positional encodings** to embeddings:  

$$
x_t = E[w_t] + \mathrm{PE}(t).
$$

Original sinusoidal:  

$$
\mathrm{PE}(t, 2i) = \sin\!\left(\frac{t}{10000^{2i/d}}\right), \quad
\mathrm{PE}(t, 2i+1) = \cos\!\left(\frac{t}{10000^{2i/d}}\right).
$$

```{code-cell} python
import numpy as np
def positional_encoding(pos, dim, base=10000.):
    i = np.arange(dim//2)
    angles = pos / (base ** (2*i/dim))
    return np.concatenate([np.sin(angles), np.cos(angles)])

pos_enc = np.stack([positional_encoding(p, 8) for p in range(5)])
print(torch.tensor(pos_enc).round(2))
```
```{code-cell} output
tensor([[ 0.0000,  0.0000,  0.0000,  0.0000,  1.0000,  1.0000,  1.0000,  1.0000],
        [ 0.8400,  0.0100,  0.0000,  0.0000,  0.5400,  1.0000,  1.0000,  1.0000],
        [ 0.9100,  0.0200,  0.0000,  0.0000, -0.4200,  1.0000,  1.0000,  1.0000],
        [ 0.1400,  0.0300,  0.0000,  0.0000, -0.9900,  1.0000,  1.0000,  1.0000],
        [-0.7600,  0.0400,  0.0000,  0.0000, -0.6500,  1.0000,  1.0000,  1.0000]])
```

```{admonition} Discussion
**Alternatives in practice.**  
Many LLMs learn **absolute** position embeddings; others use **rotary** (RoPE) to encode **relative** phase information and extrapolate better to long context {cite}`su2021roformer`. **ALiBi** adds head-specific slopes to attention scores for strong length generalization without extra parameters {cite}`press2022train`.
```

# 5 Putting It All Together

## 5.1 From Tokens to Predictions

1. **Input Tokens → Embeddings**  
   $x_t = E[w_t] + \mathrm{PE}(t)$

2. **Self-Attention**  
   $Q = X W_Q,\; K = X W_K,\; V = X W_V$  
   $A = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)$  
   $Z = A V$

3. **Feedforward Transformation**  
   $Z' = \sigma(Z W_1 + b_1) W_2 + b_2$

4. **Residual + Normalization**  
   $X_{l+1} = \mathrm{LayerNorm}(X + Z')$

After $L$ layers, project to logits:

$$
z_t = W_o h_t + b_o, \qquad
P(w_t \mid w_{<t}) = \mathrm{softmax}(z_t)
$$

```{code-cell} python
import torch, torch.nn as nn, torch.nn.functional as F

class MiniTransformerLayer(nn.Module):
    def __init__(self, d_model=8, nhead=2, dim_ff=16):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.ReLU(),
            nn.Linear(dim_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, X):
        attn_out, _ = self.mha(X, X, X)
        X = self.norm1(X + attn_out)
        ffn_out = self.ffn(X)
        X = self.norm2(X + ffn_out)
        return X

X = torch.rand(1, 5, 8)  # (batch, tokens, dim)
layer = MiniTransformerLayer()
out = layer(X)
out.shape
```

```{code-cell} output
torch.Size([1, 5, 8])
```

## 5.2 Conceptual Summary

| Stage                       | Operation                       | Mathematical Representation                       |
|-----------------------------|---------------------------------|---------------------------------------------------|
| **1. Tokenization**         | Convert text → token IDs        | —                                                 |
| **2. Embedding + Position** | $x_t = E[w_t] + \mathrm{PE}(t)$ | $X \in \mathbb{R}^{T \times d}$                   |
| **3. Attention**            | Weighted context mixing         | $Z = \mathrm{softmax}(QK^\top / \sqrt{d_k}) V$    |
| **4. Feedforward**          | Nonlinear transformation        | $\sigma(ZW_1 + b_1)W_2 + b_2$                     |
| **5. Normalization**        | Stabilize training              | $X' = \mathrm{LayerNorm}(X + \text{SubLayer}(X))$ |
| **6. Decoder Output**       | Predict next token              | $\mathrm{softmax}(W_o h_t)$                       |

## 5.3 Building Intuition

Each block in the Transformer refines the token representations by repeatedly answering the question:

> “Which other tokens in this sequence are relevant to this one?”

The deeper the stack, the richer the contextual relationships become—from syntactic to semantic to conceptual levels.

## 5.4 Next Steps

Run the interactive version in Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DCMoe/MAP6905_Adv_Deep_Learning/blob/main/notebooks/llm_intro.ipynb)

---

# References

```{bibliography}
:style: unsrt
:filter: False

vaswani2017attention
brown2020language
touvron2023llama
kingma2015adam
loshchilov2019decoupled
holtzman2020curious
press2017using
guo2017calibration
sennrich2016neural
kudo2018sentencepiece
radford2019language
xiong2020layer
shazeer2020glu
dao2022flashattention
dao2023flashattention2
choromanski2021rethinking
michel2019sixteen
su2021roformer
press2022train
```
