---
title: Mathematical Anatomy of the Transformer
---

In Module 1, we introduced the Transformer’s architecture {cite}`vaswani2017attention`.  
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

where $W_Q, W_K, W_V \in \mathbb{R}^{d \times d_k}$ {cite}`vaswani2017attention`.

### Step 2: Scaled dot-product attention

$$
A = \mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right), \qquad
Z = AV
$$

The scaling factor $\sqrt{d_k}$ keeps the dot products numerically stable by preventing the softmax from becoming too sharp as $d_k$ grows {cite}`vaswani2017attention`.

```{admonition} Derivation Insight
:class: tip
The unscaled dot-product $QK^\top$ has variance proportional to $d_k$.  
Dividing by $\sqrt{d_k}$ ensures $\mathrm{Var}(QK^\top) \approx 1$, a key insight for stable training across model sizes {cite}`vaswani2017attention`.
```

---

### 1.1 Multi-Head Attention Summary

Each attention head performs this operation independently, producing outputs $Z_i$.  
All heads are concatenated and linearly combined:

$$
\mathrm{MHA}(X) = \mathrm{Concat}(Z_1, \ldots, Z_H) W_O,
$$

where $W_O \in \mathbb{R}^{Hd_k \times d}$.

> **Intuition**: Each head learns to attend to different linguistic patterns — one may focus on local syntax, another on long-range semantics. This parallel processing is what makes Transformers so expressive {cite}`vaswani2017attention`.

---

### 1.2 Feedforward and Normalization Layers

After attention, the **Feedforward Network (FFN)** acts independently on each token:

$$
\mathrm{FFN}(x) = \sigma(x W_1 + b_1) W_2 + b_2,
$$

where $\sigma(\cdot)$ is a nonlinear activation (usually GELU {cite}`hendrycks2016gaussian` or ReLU).

Layer normalization ensures numerical stability and gradient flow:

$$
\mathrm{LayerNorm}(x) = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \odot \gamma + \beta,
$$

where $\mu$ and $\sigma^2$ are computed per feature {cite}`xiong2020layer`.

```{admonition} Why LayerNorm and not BatchNorm?
:class: important
BatchNorm depends on batch statistics, which vary across sequence positions. LayerNorm normalizes **per token**, making it ideal for variable-length sequences and recurrent-free models {cite}`xiong2020layer`.
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
> Residual connections {cite}`he2016deep` allow gradients to bypass deep nonlinear paths. In Transformers, they are **non-negotiable** — without them, training 100+ layer models would suffer catastrophic vanishing gradients {cite}`wang2022deepnet`.

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
Minimizing cross-entropy **is** maximizing likelihood of the correct token {cite}`goodfellow2016deep`.
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

```{bibliography}
:style: unsrt
:filter: False

vaswani2017attention
xiong2020layer
he2016deep
goodfellow2016deep
hendrycks2016gaussian
loshchilov2019decoupled
gotmare2019closer
wang2022deepnet
```