# Introduction to Large Language Models (LLMs)

```{note}
**Learning Objectives**

By the end of this module, you will be able to:

- Explain what a Large Language Model (LLM) is and how it processes text.
- Understand the basic architecture of the Transformer.
- Describe how tokens, embeddings, and attention mechanisms interact.
- Run and visualize basic LLM operations in PyTorch.
```

# 1 What is a Large Language Model?

Large Language Models (LLMs) are neural networks trained to model **the probability of text sequences**.  
At their core, they learn how likely a sequence of tokens is, and how to predict the **next token** given previous ones.

```{admonition} Discussion
**Why probability modeling?**  
Next-token prediction converts the open-ended task of “understand language” into a precise statistical objective. When trained at scale, the model implicitly picks up syntax, semantics, factual associations, and task structure as *by-products* of minimizing predictive error [1, 14, 15].  

**Measuring quality.**  
Two ubiquitous metrics are **negative log-likelihood** (NLL) and **perplexity**:  

$$
\text{PPL} = \exp\!\left(\frac{1}{T}\sum_{t=1}^{T} -\log P_\theta(w_t \mid w_{<t})\right).
$$

Lower is better; it corresponds to the model assigning higher probability to the observed text. Perplexity correlates with intrinsic modeling quality but not always with downstream utility; evaluations on tasks/benchmarks are also important [14].
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
In practice, the objective becomes **cross-entropy** loss optimized by first-order methods like Adam/AdamW [16, 17], often with warmup + cosine decay schedules.  

**Decoding.**  
Greedy decode maximizes local probability; *stochastic* decoding (top-$k$, nucleus/top-$p$) often yields higher-quality generations. Nucleus sampling selects the smallest set of tokens whose cumulative probability exceeds $p$ [18].
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
Many LMs tie the output projection $W_o$ with the embedding matrix $E$ (or its transpose) to reduce parameters and improve perplexity [19].  

**Temperature.**  
Sampling temperature $\tau$ rescales logits: $\text{softmax}(z_t/\tau)$.  
Higher $\tau$ → more diverse; lower $\tau$ → more deterministic.  

**Calibration and confidence.**  
Logits are not probabilities until normalized; even then, probabilities can be miscalibrated—especially OOD. Techniques like temperature scaling can help [20].
```

# 2 From Words to Tokens

## 2.1 Why Tokenization Matters

Raw text is composed of characters, but models operate on discrete **vocabulary indices**.  
Tokenization bridges this gap by splitting text into subword units and mapping them to integers.  
This allows the model to handle any text (including rare or unseen words) efficiently.

```{admonition} Discussion
**Subword methods.**  
Byte-Pair Encoding (BPE) and its modern variants learn merges that balance vocabulary size with coverage [2]. SentencePiece implements **unigram** and **BPE** with language-agnostic, whitespace-free processing [3]. GPT-2 uses byte-level BPE to robustly handle Unicode and rare strings [4].  
```

## 2.2 Example: GPT-2 Tokenizer

```{code-cell} python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")
text = "Transformers are powerful models."
tokens = tokenizer.tokenize(text)
token_ids = tokenizer.encode(text)
print('Tokens:', tokens)
print('Token IDs:', token_ids)
```

```{code-cell} output
Tokens: ['Transform', 'ers', 'Ġare', 'Ġpowerful', 'Ġmodels', '.']
Token IDs: [41762, 364, 389, 3665, 4981, 13]
```

## 2.3 Mathematical Representation

After tokenization, each token index $w_t$ is mapped to a vector embedding $x_t$ using an **embedding matrix** $E\in\mathbb{R}^{V\times d}$:

$$
x_t = E[w_t], \quad x_t \in \mathbb{R}^d.
$$

The model receives the input sequence as a matrix:

$$
X = [x_1, x_2, \ldots, x_T]^\top \in \mathbb{R}^{T\times d}.
$$

```{code-cell} python
import torch
import torch.nn as nn

V, d = 50000, 8  # toy vocabulary and embedding size
embedding = nn.Embedding(V, d)

# Example: 5-token sequence
token_ids = torch.tensor([48423, 389, 20133, 2147, 13])
X = embedding(token_ids)
X.shape
```

```{code-cell} output
torch.Size([5, 8])
```

```{admonition} Discussion
**Special tokens.**  
Real systems include special IDs for BOS/EOS, padding, and sometimes system-role markers. Padding masks must be respected in attention and loss.  

**Byte-level vs. wordpiece.**  
Byte-level tokenizers guarantee coverage (everything is representable) at the cost of longer sequences on average [4]; wordpiece/unigram trade vocabulary size against splitting frequency [3].
```

# 3 Transformer Architecture: The Engine of LLMs

## 3.1 High-Level Structure

A Transformer consists of multiple stacked **blocks**, each containing:

1. Multi-Head Self-Attention (MHA)  
2. Feedforward Neural Network (FFN)  
3. Layer Normalization  
4. Residual Connections

Formally, for the input matrix $X_l \in \mathbb{R}^{T \times d}$ at layer $l$:

$$
\begin{aligned}
H_l &= X_l + \text{MHA}(X_l), \\
X_{l+1} &= H_l + \text{FFN}(H_l).
\end{aligned}
$$

Both MHA and FFN are followed by **Layer Normalization** to stabilize training.

```{admonition} Discussion
**Pre-LN vs Post-LN.**  
Modern LLMs use **Pre-LayerNorm** (normalize *before* sublayers) for stability at depth [21].  

**FFN activations.**  
SwiGLU/GeGLU activations often outperform ReLU/GELU at similar compute [22].  

**Scaling.**  
Compute/memory grow with depth, width, and sequence length. Architectural choices (attention type, FFN size, normalization) trade efficiency vs. quality [1, 14, 23].
```

## 3.2 Self-Attention Mechanism

### Step 1: Compute Query, Key, and Value matrices

$$
Q = XW_Q, \quad K = XW_K, \quad V = XW_V,
$$

where $W_Q, W_K, W_V \in \mathbb{R}^{d \times d_k}$ are learned projection matrices.

### Step 2: Compute attention weights

$$
A = \mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right).
$$

### Step 3: Weighted sum of values

$$
Z = AV.
$$

```{code-cell} python
import torch
import torch.nn.functional as F

T, d = 4, 8
X = torch.rand(T, d)
W_Q = torch.rand(d, d)
W_K = torch.rand(d, d)
W_V = torch.rand(d, d)

Q = X @ W_Q
K = X @ W_K
V = X @ W_V

A = F.softmax(Q @ K.T / (d ** 0.5), dim=-1)  # attention weights
Z = A @ V

print('Attention weights shape:', tuple(A.shape))
print('Context shape:', tuple(Z.shape))
```

```{code-cell} output
Attention weights shape: (4, 4)
Context shape: (4, 8)
```

```{admonition} Discussion
**Causal mask & complexity.**  
Autoregressive models add a triangular mask so tokens can’t peek ahead. Vanilla attention is $O(T^2)$ in memory and compute. **FlashAttention** reorders computations to reduce memory traffic, enabling longer contexts at high throughput [5]. Other efficient variants approximate attention with kernels or sparsity [6].  
```

## 3.3 Multi-Head Attention (MHA)

Each head $h_i$ performs self-attention independently:

$$
h_i = \mathrm{Attention}(Q_i, K_i, V_i),
$$

and their outputs are concatenated:

$$
\mathrm{MHA}(X) = \mathrm{Concat}(h_1, h_2, \ldots, h_H)W_O,
$$

where $W_O \in \mathbb{R}^{Hd_k \times d}$.

```{code-cell} python
import torch.nn as nn

mha = nn.MultiheadAttention(embed_dim=8, num_heads=2, batch_first=True)
X = torch.rand(1, 4, 8)  # (batch, tokens, dim)
out, weights = mha(X, X, X)
out.shape, weights.shape
```

```{code-cell} output
(torch.Size([1, 4, 8]), torch.Size([1, 2, 4, 4]))
```

```{admonition} Discussion
**Heads as pattern detectors.**  
Some heads learn syntactic relations; others track long-range discourse. Pruning or merging heads can save compute with minimal loss in large models [24].
```

## 3.4 Feedforward Network (FFN)

$$
\text{FFN}(x) = \sigma(x W_1 + b_1) W_2 + b_2,
$$

where $\sigma$ is typically GELU or (Swi)GLU in modern LLMs.

```{code-cell} python
ffn = nn.Sequential(
    nn.Linear(8, 16),
    nn.ReLU(),
    nn.Linear(16, 8)
)
X_ffn = ffn(out.squeeze(0))
X_ffn.shape
```

```{code-cell} output
torch.Size([4, 8])
```

## 3.5 Residual Connections and Normalization

$$
\begin{aligned}
Y &= \text{LayerNorm}(X + \text{MHA}(X)), \\
Z &= \text{LayerNorm}(Y + \text{FFN}(Y)).
\end{aligned}
$$

```{code-cell} python
norm = nn.LayerNorm(8)
residual_out = norm(X.squeeze(0) + out.squeeze(0))
residual_out.shape
```

```{code-cell} output
torch.Size([4, 8])
```

# 4 Positional Encoding

## 4.1 Why We Need Position Information

Without positional information, the model would see the same embedding vector for the word “dog” whether it appears first or last in a sentence.

```{admonition} Example
:class: tip
Consider the sentences:  
- “The cat chased the dog.”  
- “The dog chased the cat.”  
Without position encodings, these two inputs would look identical to the Transformer.
```

## 4.2 The Sine–Cosine Encoding Formula

$$
\begin{aligned}
\mathrm{PE}(pos,2i)   &= \sin\!\left(\frac{pos}{10000^{2i/d}}\right), \\
\mathrm{PE}(pos,2i+1) &= \cos\!\left(\frac{pos}{10000^{2i/d}}\right).
\end{aligned}
$$

## 4.3 Adding Positional Encodings to Embeddings

$$
x_t' = x_t + \mathrm{PE}(t).
$$

```{code-cell} python
import math

def positional_encoding(max_len, d_model):
    PE = torch.zeros(max_len, d_model)
    for pos in range(max_len):
        for i in range(0, d_model, 2):
            angle = pos / (10000 ** (2 * i / d_model))
            PE[pos, i] = math.sin(angle)
            if i + 1 < d_model:
                PE[pos, i + 1] = math.cos(angle)
    return PE

PE = positional_encoding(max_len=5, d_model=8)
PE
```

```{code-cell} output
tensor([[ 0.0000,  1.0000,  0.0000,  1.0000,  0.0000,  1.0000,  0.0000,  1.0000],
        [ 0.8415,  0.5403,  0.0090,  1.0000,  0.0000,  1.0000,  0.0000,  1.0000],
        [ 0.9093, -0.4161,  0.0180,  0.9998,  0.0000,  1.0000,  0.0000,  1.0000],
        [ 0.1411, -0.9900,  0.0270,  0.9996,  0.0000,  1.0000,  0.0000,  1.0000],
        [-0.7568, -0.6536,  0.0360,  0.9993,  0.0000,  1.0000,  0.0000,  1.0000]])
```

```{admonition} Discussion
**Alternatives in practice.**  
Many LLMs learn **absolute** position embeddings; others use **rotary** (RoPE) to encode **relative** phase information and extrapolate better to long context [9]. **ALiBi** adds head-specific slopes to attention scores for strong length generalization without extra parameters [10].  
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

# References

```{bibliography}
[1] Vaswani, A., et al. (2017). *Attention is All You Need*. NeurIPS 2017.
[2] Sennrich, R., et al. (2016). Neural MT of Rare Words with Subword Units. ACL (BPE).
[3] Kudo, T., Richardson, J. (2018). SentencePiece: A simple and language independent subword tokenizer. EMNLP: System Demos.
[4] Radford, A., et al. (2019). Language Models are Unsupervised Multitask Learners. OpenAI Technical Report (GPT-2, byte-level BPE).
[5] Dao, T., et al. (2022). FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness. NeurIPS.
[6] Choromanski, K., et al. (2021). Rethinking Attention with Performers. ICLR.
[7] Kaplan, J., et al. (2020). Scaling Laws for Neural Language Models. arXiv:2001.08361.
[8] Hoffmann, J., et al. (2022). Training Compute-Optimal Large Language Models. *Chinchilla*, arXiv:2203.15556.
[9] Su, J., et al. (2021). RoFormer: Transformer with Rotary Position Embedding. NeurIPS.
[10] Press, O., et al. (2022). Train Short, Test Long: Attention with Linear Biases (ALiBi). arXiv:2108.12409.
[14] Brown, T. B., et al. (2020). Language Models are Few-Shot Learners. NeurIPS (GPT-3).
[15] Touvron, H., et al. (2023). LLaMA: Open and Efficient Foundation Language Models. arXiv:2302.13971.
[16] Kingma, D. P., Ba, J. (2015). Adam: A Method for Stochastic Optimization. ICLR.
[17] Loshchilov, I., Hutter, F. (2019). Decoupled Weight Decay Regularization (AdamW). ICLR.
[18] Holtzman, A., et al. (2020). The Curious Case of Neural Text Degeneration (nucleus sampling). ICLR.
[19] Press, O., Wolf, L. (2017). Using the Output Embedding to Improve Language Models (Weight Tying). EACL.
[20] Guo, C., et al. (2017). On Calibration of Modern Neural Networks. ICML.
[21] Xiong, R., et al. (2020). On Layer Normalization in the Transformer Architecture. ICML.
[22] Shazeer, N. (2020). GLU Variants Improve Transformer. arXiv:2002.05202 (SwiGLU).
[23] Dao, T. (2023). FlashAttention-2 & Practical Long-Context Training.
[24] Michel, P., Levy, O., Neubig, G. (2019). Are Sixteen Heads Really Better than One? NeurIPS.
[25] Ouyang, L., et al. (2022). Training language models to follow instructions with human feedback. NeurIPS (RLHF).
```