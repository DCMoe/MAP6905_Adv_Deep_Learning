# Introduction to Large Language Models (LLMs)

```{note}
**Learning Objectives**
By the end of this module, you will be able to:
- Explain what a Large Language Model (LLM) is and how it processes text.
- Understand the basic architecture of the Transformer.
- Describe how tokens, embeddings, and attention mechanisms interact.
- Run and visualize basic LLM operations in PyTorch.
```

---

## 1 What is a Large Language Model?

Large Language Models (LLMs) are neural networks trained to model **the probability of text sequences**.  
At their core, they learn how likely a sequence of tokens is, and how to predict the **next token** given previous ones.

---

### 1.1 Problem Setup (Language Modeling)

Let $w_{1:T} = (w_1, w_2, \ldots, w_T)$ represent a sequence of tokens drawn from a fixed vocabulary $\mathcal{V}$ of size $V$.  
Using the chain rule of probability, a language model defines:

$$
P(w_{1:T}) \;=\; \prod_{t=1}^{T} P\!\left(w_t \mid w_{<t}\right),
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

---

### 1.2 From Tokens to Probabilities

LLMs process **tokens**, not raw text.
Each token $w_t$ is first mapped to an integer index in ${1,\ldots,V}$, then to a **vector embedding** through a learned matrix $E \in \mathbb{R}^{V\times d}$:

$$
x_t ;=; E[w_t] \in \mathbb{R}^{d}.
$$

The sequence of embeddings $X = (x_1, x_2, \ldots, x_T)$ is fed into a **Transformer**, which outputs contextualized representations $h_t$ for each token.

A **decoder head** then transforms each hidden state $h_t$ into a set of **logits** over the vocabulary:

$$
z_t ;=; W_o, h_t ;+; b_o ;\in; \mathbb{R}^{V},
$$

and applies the **softmax** function to convert these logits into probabilities:

$$
P(w_t = v_i \mid w_{<t})
;=;
\frac{\exp(z_{t,i})}
{\displaystyle \sum_{j=1}^{V} \exp(z_{t,j})},
$$

where:

* $z_{t,i}$ is the logit corresponding to token $v_i$, and
* the denominator ensures all probabilities sum to 1.

```{admonition} Key Terms
- **Token:** a discrete unit of text (word, subword, or character).  
- **Vocabulary (\(V\))**: set of all tokens known to the model.  
- **Embedding (\(x_t\))**: dense vector representing token \(w_t\).  
- **Logits (\(z_t\))**: unnormalized scores for all tokens.  
- **Softmax:** normalizes logits into a probability distribution.
```

---

### 1.3 Why “Large”?

“Large” refers to both the **scale of parameters** (often billions) and the **volume of training data** (web-scale text, code, academic papers, etc.).
Scaling improves the model’s ability to capture complex linguistic patterns, world knowledge, and reasoning.

```{admonition} Common Misconception
LLMs do **not** retrieve answers from a database during generation.  
They compute probabilities over tokens using their learned parameters—though retrieval-augmented systems (RAG) *can* add external memory on top.
```

---


## 2 From Words to Tokens

Before text can be processed by an LLM, it must be converted into **tokens** — numerical representations that the model can understand.

---

### 2.1 Why Tokenization Matters

Raw text is composed of characters, but models operate on discrete **vocabulary indices**.  
Tokenization bridges this gap by splitting text into subword units and mapping them to integers.  
This allows the model to handle any text (including rare or unseen words) efficiently.

---

### 2.2 Example: GPT-2 Tokenizer

Below we’ll use the GPT-2 tokenizer from Hugging Face to demonstrate how text becomes tokens and IDs.

```{code-cell}
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

Here, each token corresponds to a **subword** or full word depending on the vocabulary.
For example, “Transformers” appears as one token in GPT-2’s vocabulary, while longer or rarer words may be broken into smaller pieces.

---

### 2.3 Mathematical Representation

After tokenization, each token index $w_t$ is mapped to a vector embedding $x_t$ using an **embedding matrix** $E\in\mathbb{R}^{V\times d}$:

$$
x_t = E[w_t], \quad x_t \in \mathbb{R}^d.
$$

Here:

* $V$ = vocabulary size
* $d$ = embedding dimension
* $E[w_t]$ retrieves the $w_t^{\text{th}}$ row of $E$

The model receives the input sequence as a matrix:

$$
X = [x_1, x_2, \ldots, x_T]^\top \in \mathbb{R}^{T\times d}.
$$

Each row represents a token embedding in context order.

```{code-cell}
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

---

### 2.4 Conceptual Summary

| Concept                  | Description                                      |
| ------------------------ | ------------------------------------------------ |
| **Tokenizer**            | Converts raw text into integer token IDs.        |
| **Vocabulary**           | The set of all known tokens (size =(V)).         |
| **Embedding Matrix (E)** | Maps token IDs to dense vectors.                 |
| **Input Matrix (X)**     | Sequence of embeddings fed into the Transformer. |

---

## 3 Transformer Architecture: The Engine of LLMs

The **Transformer** is the foundational architecture that powers nearly all modern LLMs.  
It replaces recurrence (used in RNNs) with **self-attention**, allowing the model to process all tokens in parallel while still capturing long-range dependencies.

---

### 3.1 High-Level Structure

A Transformer consists of multiple stacked **blocks**, each containing:

1. Multi-Head Self-Attention (MHA)  
2. Feedforward Neural Network (FFN)  
3. Layer Normalization  
4. Residual Connections


### TODO: ADD TRANSFORMER BLOCK IMAGE
```{figure} ../images/transformer_block.png
---
height: 350px
name: transformer-block
---
Simplified transformer block: attention, feedforward, normalization, and residual paths.
```

Formally, for the input matrix (X_l \in \mathbb{R}^{T \times d}) at layer (l):

$$
\begin{aligned}
H_l &= X_l + \text{MHA}(X_l), \
X_{l+1} &= H_l + \text{FFN}(H_l).
\end{aligned}
$$

Both MHA and FFN are followed by **Layer Normalization** to stabilize training.

---

### 3.2 Self-Attention Mechanism

The **self-attention** operation allows each token to attend to every other token, computing a weighted combination of their representations.

#### Step 1: Compute Query, Key, and Value matrices

$$
Q = XW_Q, \quad
K = XW_K, \quad
V = XW_V,
$$

where $W_Q, W_K, W_V \in \mathbb{R}^{d \times d_k}$ are learned projection matrices.

#### Step 2: Compute attention weights

$$
A = \mathrm{softmax}!\left(\frac{QK^\top}{\sqrt{d_k}}\right).
$$

This produces a $T \times T$ matrix $A$ showing how much each token attends to the others.

#### Step 3: Weighted sum of values

$$
Z = A,V.
$$

Here, each output vector $z_t$ is a weighted combination of all token value vectors $v_j$.

---

```{code-cell}
# Manual single-head attention on small tensors
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

print('Attention weights shape:', tuple(A.shape))  # (T, T)
print('Context shape:', tuple(Z.shape))           # (T, d)
```

```{code-cell} output
Attention weights shape: (4, 4)
Context shape: (4, 8)
```

Each token now carries information from *every other token* in the sequence — a key property that enables context awareness.

---

### 3.3 Multi-Head Attention (MHA)

Instead of using one attention head, Transformers use multiple parallel heads to capture **different relationships** between tokens.

Each head $h_i$ performs self-attention independently:

$$
h_i = \mathrm{Attention}(Q_i, K_i, V_i),
$$

and their outputs are concatenated:

$$
\mathrm{MHA}(X) = \mathrm{Concat}(h_1, h_2, \ldots, h_H)W_O,
$$

where $W_O \in \mathbb{R}^{Hd_k \times d}$ projects the combined output back to the model dimension.

```{code-cell}
import torch.nn as nn

mha = nn.MultiheadAttention(embed_dim=8, num_heads=2, batch_first=True)
X = torch.rand(1, 4, 8)  # (batch, tokens, dim)
out, weights = mha(X, X, X)
out.shape, weights.shape
```

```{code-cell} output
(torch.Size([1, 4, 8]), torch.Size([1, 2, 4, 4]))
```

---

### 3.4 Feedforward Network (FFN)

After attention, each token’s representation is transformed independently by a small two-layer neural network:

$$
\text{FFN}(x) = \text{ReLU}(x W_1 + b_1) W_2 + b_2.
$$

```{code-cell}
import torch.nn as nn

ffn = nn.Sequential(
    nn.Linear(8, 16),
    nn.ReLU(),
    nn.Linear(16, 8)
)
X_ffn = ffn(out)
X_ffn.shape
```

```{code-cell} output
torch.Size([1, 4, 8])
```

---

### 3.5 Residual Connections and Normalization

Transformers use **residual connections** and **layer normalization** to help gradients flow through many layers:

$$
\begin{aligned}
Y &= \text{LayerNorm}(X + \text{MHA}(X)), \
Z &= \text{LayerNorm}(Y + \text{FFN}(Y)).
\end{aligned}
$$

```{code-cell}
norm = nn.LayerNorm(8)
residual_out = norm(X + out)
residual_out.shape
```

```{code-cell} output
torch.Size([1, 4, 8])
```

---

### 3.6 Why It Works

Self-attention gives Transformers:

* **Parallelism**: all tokens processed simultaneously.
* **Contextual Understanding**: each token sees all others.
* **Long-range Dependencies**: unlike RNNs, attention scales globally.

---

## 4 Positional Encoding

Transformers treat tokens as a **set**—there’s no built-in notion of order.  
However, language is inherently sequential.  
To encode the **position** of each token in the sequence, we add **positional encodings** to the token embeddings.

---

### 4.1 Why We Need Position Information

Unlike RNNs, Transformers process all tokens simultaneously.  
Without positional information, the model would see the same embedding vector for the word “dog” whether it appears first or last in a sentence.  
Adding positional encodings gives the model a sense of **token order**.

```{admonition} Example
:class: tip
Consider the sentences:
- “The cat chased the dog.”  
- “The dog chased the cat.”  
Without position encodings, these two inputs would look identical to the Transformer.
```

---

### 4.2 The Sine–Cosine Encoding Formula

Vaswani et al. (2017) proposed using fixed sinusoidal functions of different frequencies:

$$
\begin{aligned}
\text{PE}*{(pos,2i)} &= \sin!\left(\frac{pos}{10000^{2i/d}}\right), \
\text{PE}*{(pos,2i+1)} &= \cos!\left(\frac{pos}{10000^{2i/d}}\right),
\end{aligned}
$$

where:

* $pos$ = position index $0 to T − 1$
* $i$ = dimension index $0 to d/2 − 1$
* $d$ = embedding dimension

The even indices use sine and the odd indices use cosine so that each dimension captures a unique frequency pattern.

---

### 4.3 Adding Positional Encodings to Embeddings

For each token embedding $x_t$, we add its positional encoding vector $\text{PE}(t)$:

$$
x_t' = x_t + \text{PE}(t).
$$

This enriched representation $x_t'$ combines both **content** (from embeddings) and **position** (from PE).

---

```{code-cell}
import math
import torch

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

Each row represents one token’s position encoding across the eight dimensions.


---

### 4.4 Key Properties

| Property            | Description                                            |
| ------------------- | ------------------------------------------------------ |
| **Deterministic**   | No extra parameters — the same formula for all models. |
| **Continuous**      | Allows smooth interpolation for longer sequences.      |
| **Unique Patterns** | Each position produces a distinct encoding.            |
| **Additive**        | Can be directly added to embeddings.                   |

---


## 5 Putting It All Together

Now that we’ve explored each component — tokenization, embeddings, attention, feedforward layers, and positional encoding —  
we can combine them to understand the **full forward pass** of a Transformer-based LLM.

---

### 5.1 From Tokens to Predictions

Let’s walk through the data flow for a single Transformer layer.

1️⃣ **Input Tokens → Embeddings**

$$
x_t = E[w_t] + \mathrm{PE}(t)
$$

2️⃣ **Self-Attention**

$$
\begin{aligned}
Q &= X W_Q, \quad K = X W_K, \quad V = X W_V, \\
A &= \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right), \\
Z &= A V.
\end{aligned}
$$

3️⃣ **Feedforward Transformation**

$$
Z' = \mathrm{ReLU}(Z W_1 + b_1) W_2 + b_2
$$

4️⃣ **Residual + Normalization**

$$
X_{l+1} = \mathrm{LayerNorm}(X + Z')
$$

After stacking $L$ layers, we project the final hidden state to vocabulary logits:

$$
z_t = W_o h_t + b_o, \qquad
P(w_t \mid w_{<t}) = \mathrm{softmax}(z_t)
$$

---

```{code-cell}
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

Each layer preserves the token dimension while refining each embedding through **contextual mixing** and **nonlinear transformation**.

---

### 5.2 Conceptual Summary

| Stage                       | Operation                       | Mathematical Representation                       |
| --------------------------- | ------------------------------- | ------------------------------------------------- |
| **1. Tokenization**         | Convert text → token IDs        | —                                                 |
| **2. Embedding + Position** | $x_t = E[w_t] + \mathrm{PE}(t)$ | $X \in \mathbb{R}^{T \times d}$                   |
| **3. Attention**            | Weighted context mixing         | $Z = \mathrm{softmax}(QK^\top / \sqrt{d_k}) V$    |
| **4. Feedforward**          | Nonlinear transformation        | $\mathrm{ReLU}(ZW_1 + b_1)W_2 + b_2$              |
| **5. Normalization**        | Stabilize training              | $X' = \mathrm{LayerNorm}(X + \text{SubLayer}(X))$ |
| **6. Decoder Output**       | Predict next token              | $\mathrm{softmax}(W_o h_t)$                       |

---

### 5.3 Building Intuition

Each block in the Transformer refines the token representations by repeatedly answering the question:

> “Which other tokens in this sequence are relevant to this one?”

The deeper the stack, the richer the contextual relationships become — from syntactic to semantic to conceptual levels.

---

### 5.4 Next Steps

Now that we’ve assembled the pieces, you’re ready to **run the interactive version** in Colab.
In the next notebook, you’ll visualize attention maps, inspect intermediate tensors, and experiment with modifying embeddings.

### UPDATE COLAB LINK

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DCMoe/MAP6905_Adv_Deep_Learning/notebooks/llm_intro.ipynb)

---

```{admonition} Next Module
Continue to: [Training Objectives and Loss Functions](LLMs/02_transformer_math.md)
```

## References

[1] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention is all you need. In Advances in Neural Information Processing Systems (NeurIPS 2017). arXiv:1706.03762 [cs.CL]. https://doi.org/10.48550/arXiv.1706.03762