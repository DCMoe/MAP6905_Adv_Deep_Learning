---
title: Mathematical Anatomy of the Transformer
---

In Module 1, we introduced the Transformer’s architecture [1].  
Now let’s express its components formally in **matrix notation**, so we can later connect them to optimization and gradients.

<iframe width="560" height="315" src="https://www.youtube.com/embed/wjZofJX0v4M?si=MTudK5YA1moQJW8q" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

<iframe width="560" height="315" src="https://www.youtube.com/embed/eMlx5fFNoYc?si=hLoB1REfRj-1_SY9" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

---

## 1 Multi-Head Attention as a Matrix Operation

Each Transformer block processes an input sequence $X \in \mathbb{R}^{T \times d}$, where:

* $T$: number of tokens  
* $d$: embedding dimension

### Step 1: Linear projections

We compute **query**, **key**, and **value** matrices via learned projections:

$$
Q = XW_Q, \quad K = XW_K, \quad V = XW_V,
$$

where $W_Q, W_K, W_V \in \mathbb{R}^{d \times d_k}$ [1].

### Step 2: Scaled dot-product attention

$$
A = \mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right), \qquad
Z = AV
$$

The scaling factor $\sqrt{d_k}$ keeps the dot products numerically stable by preventing the softmax from becoming too sharp as $d_k$ grows [1].

```{admonition} Derivation Insight
:class: tip
The unscaled dot-product $QK^\top$ has variance proportional to $d_k$.  
Dividing by $\sqrt{d_k}$ ensures $\mathrm{Var}(QK^\top) \approx 1$, a key insight for stable training across model sizes [1].
```

---

### 1.1 Multi-Head Attention Summary

Each attention head performs this operation independently, producing outputs $Z_i$.  
All heads are concatenated and linearly combined:

$$
\mathrm{MHA}(X) = \mathrm{Concat}(Z_1, \ldots, Z_H) W_O,
$$

where $W_O \in \mathbb{R}^{H d_k \times d}$.

> **Intuition**: Each head learns to attend to different linguistic patterns — one may focus on local syntax, another on long-range semantics. This parallel processing is what makes Transformers so expressive [1].

---

### 1.2 Feedforward and Normalization Layers

After attention, the **Feedforward Network (FFN)** acts independently on each token:

$$
\mathrm{FFN}(x) = \sigma(x W_1 + b_1) W_2 + b_2,
$$

where $\sigma(\cdot)$ is a nonlinear activation (usually GELU [5] or ReLU).

Layer normalization ensures numerical stability and gradient flow:

$$
\mathrm{LayerNorm}(x) = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \odot \gamma + \beta,
$$

where $\mu$ and $\sigma^2$ are computed per feature [2].

```{admonition} Why LayerNorm and not BatchNorm?
:class: important
BatchNorm depends on batch statistics, which vary across sequence positions. LayerNorm normalizes **per token**, making it ideal for variable-length sequences and recurrent-free models [2].
```

---

### 1.3 The Full Layer Equation

Each Transformer layer performs two residual updates:

$$
\begin{aligned}
Y &= \mathrm{LayerNorm}(X + \mathrm{MHA}(X)), \\
Z &= \mathrm{LayerNorm}(Y + \mathrm{FFN}(Y)).
\end{aligned}
$$

Stacking $L$ such layers yields the hidden representation $H_L$.

> **Discussion: The Power of Residuals**  
> Residual connections [3] allow gradients to bypass deep nonlinear paths. In Transformers, they are **non-negotiable** — without them, training 100+ layer models would suffer catastrophic vanishing gradients [8].

---

## 2 Cross-Entropy Loss and Perplexity

Training an LLM means teaching it to **predict the next token** in a sequence.  
We minimize a loss that measures divergence between predicted and true distributions.

---

### 2.1 Next-Token Prediction Objective

For vocabulary size $V$, let  
- $z_t \in \mathbb{R}^V$: logits at position $t$  
- $y_t \in \{1,\dots,V\}$: true next token

After softmax:

$$
p_{t,i} = \frac{e^{z_{t,i}}}{\sum_{j=1}^{V} e^{z_{t,j}}}.
$$

We maximize $\log p_{t, y_t}$, or minimize:

$$
\mathcal{L}_{\text{NLL}}(t) = -\log p_{t, y_t}.
$$

---

### 2.2 Derivation of the Cross-Entropy Form

With one-hot target $y_t$:

$$
\mathcal{L}_{\text{CE}}(t) = - \sum_{i=1}^{V} y_{t,i} \log p_{t,i}.
$$

Averaged over $T$ tokens:

$$
\mathcal{L}_{\text{CE}} = -\frac{1}{T}\sum_{t=1}^{T} \sum_{i=1}^{V} y_{t,i} \log p_{t,i}.
$$

```{admonition} Key Identity
:class: note
Since $y_t$ is one-hot, $\mathcal{L}_{\text{CE}} = -\log p_{t,y_t}$.  
Minimizing cross-entropy **is** maximizing likelihood of the correct token [4].
```

---

### 2.3 Gradient of the Softmax–Cross-Entropy

$$
\frac{\partial \mathcal{L}_{\text{CE}}}{\partial z_{t,k}} = p_{t,k} - y_{t,k}.
$$

```{admonition} Derivation Outline
:class: tip
1. Chain rule on $\log p_{t,y_t}$  
2. For correct class: $p_{t,y_t} - 1$; others: $p_{t,k}$  
3. Result: $p_t - y_t$ [4]
```

This **clean subtraction** makes backprop through softmax extremely efficient.

---

### 2.4 Implementing Cross-Entropy in PyTorch

```python
import torch
import torch.nn.functional as F

logits = torch.tensor([[2.0, 1.0, 0.1, -1.0, 0.5],
                       [1.2, 0.0, 0.3, 2.0, -0.2]])
targets = torch.tensor([0, 3])

loss = F.cross_entropy(logits, targets)
print("Cross-entropy loss:", float(loss))
```

**Output**

```
Cross-entropy loss: 0.6063537001609802
```

---

### 2.5 Relating Loss to Perplexity

**Perplexity (PPL)** is the exponential of average NLL:

$$
\mathrm{PPL} = \exp\left(\frac{1}{T}\sum_{t=1}^{T} -\log p_{t,y_t}\right) = e^{\mathcal{L}_{\text{CE}}}.
$$

> **Intuition**: PPL answers: “How many equally likely tokens was the model unsure between?”  
> PPL = 20 → model behaves as if 20 tokens were equally probable [9].

```python
perplexity = torch.exp(loss)
print("Perplexity:", float(perplexity))
```

**Output**

```
Perplexity: 1.8337328433990479
```

---

### 2.6 Visualizing the Loss Surface

```python
import numpy as np
import matplotlib.pyplot as plt

z = np.linspace(-5, 5, 200)
p = np.exp(z) / (np.exp(z) + np.exp(-z))
loss_pos = -np.log(p)
loss_neg = -np.log(1 - p)

plt.figure(figsize=(6,3))
plt.plot(z, loss_pos, label="True label = 1")
plt.plot(z, loss_neg, label="True label = 0")
plt.title("Cross-Entropy Loss vs Logit Value")
plt.xlabel("Logit z")
plt.ylabel("Loss")
plt.legend()
plt.show()
```

```{figure} ../images/CE_loss.png
---
height: 280px
name: ce-loss-curve
---
Convex loss surface ensures reliable convergence.
```

---

## 3 Gradient Derivation & Optimization Intuition

Let’s explore **how gradients flow** and why Transformers train so effectively.

---

### 3.1 Gradient Flow in Softmax–Cross-Entropy

$$
\frac{\partial \mathcal{L}}{\partial z_{t,k}} = p_{t,k} - y_{t,k}.
$$

- High confidence in correct token → small gradient  
- High confidence in wrong token → large gradient

This creates a **self-correcting learning signal** [4].

```python
logits = torch.tensor([[5.0, 0.5]])
target = torch.tensor([0])
probs = F.softmax(logits, dim=-1)
grads = probs - F.one_hot(target, num_classes=2).float()
print("Gradients (p - y):", grads.tolist())
```

**Output**

```
Gradients (p - y): [[-0.010986924171447754, 0.010986942797899246]]
```

---

### 3.2 Backpropagation Through Attention

For $Z = AV$, $A = \mathrm{softmax}(S)$, $S = \frac{QK^\top}{\sqrt{d_k}}$:

$$
\frac{\partial A_{ij}}{\partial S_{mn}} = A_{ij} (\delta_{im} - A_{in}).
$$

This ensures attention weights stay normalized during backprop [1].

```{admonition} Gradient Stability
:class: important
Scaling by $\sqrt{d_k}$ keeps dot-product variance ~1, preventing explosion in early training [1].
```

---

### 3.3 Normalization and Residuals

**LayerNorm** [2] stabilizes activations per token.  
**Residual connections** [3] allow direct gradient flow.

> **Why do Transformers scale?**  
> 1. **Stable gradients** (scaling + LayerNorm)  
> 2. **Skip connections** (residuals)  
> 3. **Efficient updates** (softmax-CE gradient)  
> → Enables training of 1000+ layer models [8]

---

### 3.4 Visualization: Gradient Magnitudes per Layer

```python
import torch.nn as nn

torch.manual_seed(0)

class MiniBlock(nn.Module):
    def __init__(self, d=16):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))
        self.norm = nn.LayerNorm(d)
    def forward(self, x):
        return self.norm(x + self.fc(x))

x = torch.randn(1, 10, 16, requires_grad=True)
blocks = nn.ModuleList([MiniBlock() for _ in range(6)])

out = x
for block in blocks: out = block(out)
loss = out.pow(2).mean()
loss.backward()

grads = [p.grad.norm().item() for p in blocks.parameters() if p.grad is not None]
print("Gradient norms (first 10):", grads[:10])
```

**Output**

```
Gradient norms (first 10): [7.06e-07, 3.61e-07, ...]
```

LayerNorm keeps gradients balanced across layers.

---

### 3.5 Optimization: AdamW and Scheduling

Modern LLMs use **AdamW** [6]:

$$
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1 - \beta_1) g_t, \\
v_t &= \beta_2 v_{t-1} + (1 - \beta_2) g_t^2, \\
\theta_{t+1} &= \theta_t - \eta \frac{m_t / (1-\beta_1^t)}{\sqrt{v_t / (1-\beta_2^t)} + \epsilon} - \eta \lambda \theta_t.
\end{aligned}
$$

> **Why AdamW?** Decouples weight decay from adaptive steps → better generalization [6].

Often paired with **linear warmup + cosine decay** [7].

---

### 3.6 Loss Curve During Training

```{figure} ../images/loss_curve_example.png
---
height: 280px
name: training-loss-curve
---
Smooth, steady decline — a hallmark of stable optimization.
```

---

## 4 Training Pipeline: Mini Transformer on a Tiny Dataset

Let’s train a **2-layer, 2-head** Transformer from scratch.

---

### 4.1 Overview

| Component       | Choice |
|----------------|--------|
| Dataset         | `wikitext-2` (1% subset) |
| Tokenizer       | GPT-2 |
| Model           | $d=64$, 2 layers, 2 heads |
| Loss            | Cross-entropy |
| Optimizer       | AdamW |

---

### 4.2 Load and Tokenize

```python
from datasets import load_dataset
from transformers import AutoTokenizer

dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:1%]")

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

def tokenize_fn(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=64)

tokenized = dataset.map(tokenize_fn, batched=True)
input_ids = torch.tensor(tokenized["input_ids"][:256])
print("Shape:", input_ids.shape)
```

**Output**

```
Shape: torch.Size([256, 64])
```

---

### 4.3 Tiny Transformer Model

```python
import torch.nn as nn

class TinyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=64, nhead=2, nlayers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(64, d_model)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=nlayers)
        self.decoder = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device)
        h = self.embed(x) + self.pos(pos)
        out = self.encoder(h)
        logits = self.decoder(out)
        return logits, out
```

---

### 4.4 Training Loop

```python
from torch.optim import AdamW
import matplotlib.pyplot as plt

model = TinyTransformer(tokenizer.vocab_size)
optimizer = AdamW(model.parameters(), lr=1e-3)
losses = []

for epoch in range(5):
    total_loss = 0.0
    for batch in input_ids.split(16):
        optimizer.zero_grad()
        logits, _ = model(batch)
        loss = F.cross_entropy(logits[:, :-1, :].reshape(-1, tokenizer.vocab_size),
                               batch[:, 1:].reshape(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(input_ids)
    losses.append(avg_loss)
    print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")

plt.plot(losses, marker='o')
plt.title("Training Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Average Loss")
plt.show()
```

**Output**

```
Epoch 1: Loss = 0.4878
Epoch 2: Loss = 0.3715
Epoch 3: Loss = 0.2676
Epoch 4: Loss = 0.1982
Epoch 5: Loss = 0.1702
```

```{figure} ../images/mod2_4-4_loss.png
---
height: 280px
name: CE-loss-curve
---
```

---

### 4.5 Evaluate Perplexity

```python
with torch.no_grad():
    logits, _ = model(input_ids[:32])
    loss = F.cross_entropy(logits[:, :-1, :].reshape(-1, tokenizer.vocab_size),
                           input_ids[:32, 1:].reshape(-1))
    ppl = torch.exp(loss)
    print(f"Validation Loss: {loss:.4f} | Perplexity: {ppl:.2f}")
```

**Output**

```
Validation Loss: 3.2068 | Perplexity: 24.70
```

---

### 4.6 Summary Table

| Concept                | Description |
|------------------------|-----------|
| **Cross-Entropy Loss** | Measures prediction error in probability space |
| **Perplexity**         | $e^{\text{loss}}$ — human-interpretable metric |
| **AdamW**              | Stable optimizer with decoupled weight decay |

---

## 5 Colab Notebook & Reproducibility

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DCMoe/MAP6905_Adv_Deep_Learning/blob/main/notebooks/llm_training.ipynb)

```{admonition} Reproducibility Tips
:class: warning
- Use GPU runtime  
- Set `torch.manual_seed(0)`  
- Batch size ≤16  
- Log with `wandb` for larger runs
```

---

## References

1. Vaswani, A., et al. "Attention is All You Need." *NeurIPS*, 2017.  
2. Ba, J. L., et al. "Layer Normalization." *arXiv:1607.06450*, 2016.  
3. He, K., et al. "Deep Residual Learning for Image Recognition." *CVPR*, 2016.  
4. Goodfellow, I., et al. *Deep Learning*. MIT Press, 2016.  
5. Hendrycks, D., et al. "Gaussian Error Linear Units (GELUs)." *arXiv:1606.08415*, 2020.  
6. Loshchilov, I., et al. "Decoupled Weight Decay Regularization." *ICLR*, 2019.  
7. Gotmare, A., et al. "A Closer Look at Deep Learning Heuristics." *ICML*, 2019.  
8. Wang, H., et al. "DeepNet: Scaling Transformers to 1,000 Layers." *arXiv:2203.00555*, 2022.  
9. Jurafsky, D., et al. *Speech and Language Processing*. Prentice Hall, 2020.