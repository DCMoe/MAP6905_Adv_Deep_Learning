#  Mathematical Anatomy of the Transformer

In Module 1, we introduced the Transformer’s architecture.
Now, let’s express its components formally in **matrix notation**, so we can later connect them to optimization and gradients.

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

where $W_Q, W_K, W_V \in \mathbb{R}^{d \times d_k}$.

### Step 2: Scaled dot-product attention

$$
A = \mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right), \qquad
Z = AV
$$

The scaling factor $\sqrt{d_k}$ keeps the dot products numerically stable by preventing the softmax from becoming too sharp as $d_k$ grows.

---

```{admonition} Derivation Insight
The unscaled dot-product $QK^\top$ can have variance proportional to $d_k$.  
Dividing by $\sqrt{d_k}$ ensures that $\mathrm{Var}(QK^\top)$ remains roughly constant across model sizes.
```

---

### 1.1 Multi-Head Attention Summary

Each attention head performs this operation independently, producing outputs $Z_i$.
All heads are concatenated and linearly combined:

$$
\mathrm{MHA}(X) = \mathrm{Concat}(Z_1, \ldots, Z_H), W_O,
$$

where $W_O \in \mathbb{R}^{H d_k \times d}$.

---

### 1.2 Feedforward and Normalization Layers

After attention, the **Feedforward Network (FFN)** acts independently on each token:

$$
\mathrm{FFN}(x) = \sigma(x W_1 + b_1) W_2 + b_2,
$$

where $\sigma(\cdot)$ is a nonlinear activation (usually ReLU or GELU).

Layer normalization ensures numerical stability and gradient flow:

$$
\mathrm{LayerNorm}(x) = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \odot \gamma + \beta,
$$

where:

* $\mu$ and $\sigma^2$ are the mean and variance per feature,
* $\gamma, \beta$ are learnable scale and bias parameters.

---

### 1.3 The Full Layer Equation

Each Transformer layer performs two residual updates:

$$
\begin{aligned}
Y &= \mathrm{LayerNorm}(X + \mathrm{MHA}(X)), \\
Z &= \mathrm{LayerNorm}(Y + \mathrm{FFN}(Y)).
\end{aligned}
$$

Stacking $L$ such layers yields the hidden representation $H_L$, which is then passed to the decoder to produce logits over the vocabulary.


## 2 Cross-Entropy Loss and Perplexity

Training an LLM means teaching it to **predict the next token** in a sequence as accurately as possible.  
We do this by minimizing a loss function that compares the model’s predicted distribution to the true next-token distribution.

---

### 2.1 Next-Token Prediction Objective

For a vocabulary of size $V$, let  
- $z_t \in \mathbb{R}^V$ be the logits (unnormalized scores) for position $t$, and  
- $y_t \in \{1,\dots,V\}$ be the **true** token index at that position.

After the softmax, the model’s predicted probability for class $i$ is:

$$
p_{t,i} = \frac{e^{z_{t,i}}}{\displaystyle \sum_{j=1}^{V} e^{z_{t,j}}}.
$$

The training goal is to **maximize the likelihood** of the correct token $y_t$:

$$
\max_\theta \log p_{t, y_t}.
$$

Equivalently, we **minimize the negative log-likelihood (NLL)**:

$$
\mathcal{L}_{\text{NLL}}(t)
= -\log p_{t, y_t}
= -\log
\frac{e^{z_{t, y_t}}}{\sum_{j=1}^{V} e^{z_{t,j}}}.
$$

---

### 2.2 Derivation of the Cross-Entropy Form

If we represent the true target as a one-hot vector $y_t \in \{0,1\}^V$ where $y_{t,i}=1$ for the correct token, the average loss over all vocabulary entries is

$$
\mathcal{L}_{\text{CE}}(t)
= - \sum_{i=1}^{V} y_{t,i} \log p_{t,i}.
$$

For a batch of sequences with $T$ tokens each, the total loss is:

$$
\mathcal{L}_{\text{CE}}
= -\frac{1}{T}\sum_{t=1}^{T}
\sum_{i=1}^{V} y_{t,i} \log p_{t,i}.
$$

This is the **cross-entropy loss** between the true distribution $y_t$ and the predicted distribution $p_t$.

---

```{admonition} Mathematical Identity
Because $y_t$ is one-hot, $\displaystyle\mathcal{L}_{\text{CE}} = -\log p_{t,y_t}$.
Thus, minimizing cross-entropy is identical to maximizing the log-likelihood of the correct token.
```

---

### 2.3 Gradient of the Softmax–Cross-Entropy

Let’s derive the gradient with respect to each logit $z_{t,k}$.

We start from the probability definition:

$$
p_{t,k} = \frac{e^{z_{t,k}}}{\sum_j e^{z_{t,j}}}.
$$

The derivative of the loss w.r.t. $z_{t,k}$ is:

$$
\frac{\partial \mathcal{L}*{\text{CE}}}{\partial z*{t,k}}
= p_{t,k} - y_{t,k}.
$$

```{admonition} Derivation Outline
1️⃣ Differentiate $\log p_{t,y_t}$ using the chain rule.  
2️⃣ For the correct class, you get $p_{t,y_t}-1$; for others, $p_{t,k}$.  
3️⃣ The result elegantly simplifies to $p_t - y_t$.
```

This simplicity makes back-propagation efficient — the gradient of cross-entropy with softmax collapses to a clean subtraction between predicted probabilities and true labels.

---

### 2.4 Implementing Cross-Entropy in PyTorch

```{code-cell} python
import torch
import torch.nn.functional as F

# Example: batch of 2 time steps, vocab size 5
logits = torch.tensor([[2.0, 1.0, 0.1, -1.0, 0.5],
                       [1.2, 0.0, 0.3, 2.0, -0.2]])
targets = torch.tensor([0, 3])  # correct indices

loss = F.cross_entropy(logits, targets)
print("Cross-entropy loss:", float(loss))
```

```{code-cell} output
Cross-entropy loss: 0.6063537001609802
```

---

### 2.5 Relating Loss to Perplexity

**Perplexity (PPL)** is a common metric for evaluating language models.
It is the **exponential** of the average negative log-likelihood:

$$
\mathrm{PPL}
= \exp!\left(
\frac{1}{T}\sum_{t=1}^{T}
-\log p_{t,y_t}
\right)
= e^{\mathcal{L}_{\text{CE}}}.
$$

Intuitively, perplexity measures how “surprised” the model is by the data.
Lower perplexity → better predictive power.

```{code-cell} python
# Compute perplexity from loss
perplexity = torch.exp(loss)
print("Perplexity:", float(perplexity))
```

```{code-cell} output
Perplexity: 1.8337328433990479
```

---

### 2.6 Visualizing the Loss Surface (Optional)

We can visualize how cross-entropy behaves for a two-class softmax as logits vary.

```{code-cell} python
import numpy as np
import matplotlib.pyplot as plt

z = np.linspace(-5, 5, 200)
p = np.exp(z) / (np.exp(z) + np.exp(-z))
loss_pos = -np.log(p)         # when true label = 1
loss_neg = -np.log(1 - p)     # when true label = 0

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
name: training-loss-curve
---
```


## 3 Gradient Derivation & Optimization Intuition

Now that we’ve derived the cross-entropy loss, we can explore **how gradients flow** through the Transformer during training.

Our goal is to understand:
1. Why optimization works so effectively with softmax + cross-entropy.
2. How gradient scaling and normalization stabilize learning.
3. How these insights translate into practical PyTorch training behavior.

---

### 3.1 Gradient Flow in Softmax–Cross-Entropy

We derived in Section 2 that:

$$
\frac{\partial \mathcal{L}}{\partial z_{t,k}} = p_{t,k} - y_{t,k}.
$$

Let’s interpret this result.

- If the model predicts correctly (high $p_{t,y_t}$), the gradient magnitude is **small**.
- If it’s wrong (low $p_{t,y_t}$), the gradient is **large**, pushing the logit upward.

This dynamic creates a **self-correcting mechanism**:
> The model automatically adjusts its logits more strongly for incorrect predictions.

---

```{code-cell} python
# Demonstrate gradient magnitudes for different logits
import torch
import torch.nn.functional as F

logits = torch.tensor([[5.0, 0.5]])  # confident in class 0
target = torch.tensor([0])
loss = F.cross_entropy(logits, target, reduction='none')

# Compute gradients manually
probs = F.softmax(logits, dim=-1)
grads = probs - F.one_hot(target, num_classes=2).float()
print("Probabilities:", probs.tolist())
print("Gradients (p - y):", grads.tolist())
print("Loss:", loss.item())
```

```{code-cell} output
Probabilities: [[0.9890130758285522, 0.010986942797899246]]
Gradients (p - y): [[-0.010986924171447754, 0.010986942797899246]]
Loss: 0.011047743260860443
```

The gradient for the correct class $(-0.010987)$ is small — indicating the model is confident.
For an incorrect guess, this difference would be much larger, accelerating correction.

---

### 3.2 Backpropagation Through Attention

Let’s analyze how gradients propagate through the attention operation:

$$
Z = AV, \qquad A = \mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right).
$$

To update parameters in $W_Q$, $W_K$, and $W_V$, the model computes partial derivatives such as:

$$
\frac{\partial \mathcal{L}}{\partial W_Q}
= \frac{\partial \mathcal{L}}{\partial A}
\cdot
\frac{\partial A}{\partial (QK^\top)}
\cdot
\frac{\partial (QK^\top)}{\partial W_Q}.
$$

The softmax derivative plays a central role here:

$$
\frac{\partial A_{ij}}{\partial S_{mn}}
= A_{ij} (\delta_{im} - A_{in}),
\quad
\text{where } S = \frac{QK^\top}{\sqrt{d_k}}.
$$

This structure ensures that increasing attention to one token decreases it for others, maintaining normalization within each row of $A$.

---

```{admonition} Gradient Stability
The division by $\sqrt{d_k}$ stabilizes gradients by keeping the dot-product scale near zero mean and unit variance.
Without it, early training can produce exploding attention weights.
```

---

### 3.3 Normalization and Residuals

**LayerNorm** ensures gradients remain well-scaled across layers.
It rescales activations to unit variance before applying learned scale $(\gamma)$ and bias $(\beta)$:

$$
\begin{aligned}
\frac{\partial \mathcal{L}}{\partial x_i}
&\propto
\frac{1}{\sqrt{\sigma^2 + \epsilon}}
\Bigg[
\frac{\partial \mathcal{L}}{\partial \hat{x}_i}
- \frac{1}{d} \sum_j \frac{\partial \mathcal{L}}{\partial \hat{x}_j}
- \hat{x}_i
  \sum_j \frac{\partial \mathcal{L}}{\partial \hat{x}_j} \hat{x}_j
\Bigg].
\end{aligned}
$$

This expression shows that LayerNorm not only normalizes activations but also **balances gradient contributions** across features.

Residual connections complement this by allowing gradients to **bypass** nonlinear sublayers, reducing vanishing effects.

---

### 3.4 Visualization: Gradient Magnitudes per Layer

In deeper networks, gradient norms can vary drastically.
Below we simulate a random Transformer stack to observe this effect.

```{code-cell} python
import torch.nn as nn

torch.manual_seed(0)

# Build 6-layer mini transformer stack
class MiniBlock(nn.Module):
    def __init__(self, d=16):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))
        self.norm = nn.LayerNorm(d)
    def forward(self, x):
        return self.norm(x + self.fc(x))

x = torch.randn(1, 10, 16, requires_grad=True)
blocks = nn.ModuleList([MiniBlock() for _ in range(6)])

# Forward & backward pass
out = x
for block in blocks: out = block(out)
loss = out.pow(2).mean()
loss.backward()

grads = [p.grad.norm().item() for p in blocks.parameters() if p.grad is not None]
print("Gradient norms (first 10 params):", grads[:10])
```

```{code-cell} output
Gradient norms (first 10 params): [7.056015647322056e-07, 3.6120545132689585e-07, 1.0183679250985733e-06, 9.552564961268217e-07, 3.444132801178057e-07, 9.345910143565561e-07, 7.831649213585479e-07, 3.713057594723068e-07, 1.2017155768262455e-06, 9.130194484896492e-07]
```

Even with this small example, we can see that normalization helps keep gradients consistent across layers — preventing instability in deep models.

---

### 3.5 Optimization Intuition

During training, each gradient update approximately follows:

$$
\theta_{t+1} = \theta_t - \eta \frac{\partial \mathcal{L}}{\partial \theta_t},
$$

where $\eta$ is the learning rate.

However, large models often use adaptive optimizers (like AdamW) that maintain **moment estimates** of gradients:

$$
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1 - \beta_1) g_t, \\
v_t &= \beta_2 v_{t-1} + (1 - \beta_2) g_t^2, \\
\theta_{t+1} &= \theta_t - \eta \frac{m_t / (1-\beta_1^t)}{\sqrt{v_t / (1-\beta_2^t)} + \epsilon}.
\end{aligned}
$$

```{admonition} Intuition
AdamW smooths noisy gradient updates and decouples weight decay from learning rate,
allowing stable optimization even for very large Transformers.
```

---

### 3.6 Visualizing Loss Curve During Training (Preview)

In the next section, we’ll implement a **toy training loop** on a small Hugging Face dataset to visualize the **loss curve** across epochs:

```{figure} ../images/loss_curve_example.png
---
height: 280px
name: training-loss-curve
---
Loss decreasing smoothly over training steps.
```


## 4 Training Pipeline: Mini Transformer on a Tiny Dataset

Now we’ll put everything together into a **training loop** that learns to predict the next token  
on a small text dataset from the Hugging Face `datasets` library.

---

### 4.1 Overview of the Training Setup

We’ll use a lightweight model to ensure quick training:
- **Dataset:** `wikitext-2` (small subset of Wikipedia)
- **Tokenizer:** GPT-2 tokenizer from Hugging Face
- **Model:** a mini Transformer with 2 layers and 2 heads
- **Loss:** cross-entropy
- **Optimizer:** AdamW

Our goal is *not* high performance — just to visualize how an LLM learns token dependencies.

---

### 4.2 Load and Tokenize the Dataset

```{code-cell} python
from datasets import load_dataset
from transformers import AutoTokenizer

# Load a small text dataset
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:1%]")

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
def tokenize_fn(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=64)

tokenized = dataset.map(tokenize_fn, batched=True)
input_ids = torch.tensor(tokenized["input_ids"][:256])  # small subset
print("Input tensor shape:", input_ids.shape)
```

```{code-cell} output
Input tensor shape: torch.Size([256, 64])
```

Each row is a 64-token sequence that the model will try to predict one step ahead.

---

### 4.3 Define a Tiny Transformer Model

```{code-cell} python
import torch.nn as nn
import torch.nn.functional as F

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
        return logits, out  # return both logits and hidden states
```

---

### 4.4 Training Loop with Loss Tracking

We’ll train the model for a few epochs and visualize the loss curve.

```{code-cell}
from torch.optim import AdamW
import matplotlib.pyplot as plt

vocab_size = tokenizer.vocab_size
model = TinyTransformer(vocab_size)
optimizer = AdamW(model.parameters(), lr=1e-3)
losses = []

for epoch in range(5):
    total_loss = 0.0
    for batch in input_ids.split(16):  # small batch size
        optimizer.zero_grad()
        logits, _ = model(batch)
        targets = batch
        loss = F.cross_entropy(logits[:, :-1, :].reshape(-1, vocab_size), targets[:, 1:].reshape(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    losses.append(total_loss / len(input_ids))
    print(f"Epoch {epoch+1}: Loss = {losses[-1]:.4f}")

plt.plot(losses, marker='o')
plt.title("Training Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Average Loss")
plt.show()
```

```{code-cell} output
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

As training progresses, the loss decreases — meaning the model is improving its token predictions.

---

### 4.5 Evaluate Perplexity on Validation Samples

```{code-cell} python
with torch.no_grad():
    logits, _ = model(input_ids[:32])
    loss = F.cross_entropy(logits[:, :-1, :].reshape(-1, vocab_size),
                           input_ids[:32, 1:].reshape(-1))
    perplexity = torch.exp(loss)
    print(f"Validation Loss: {loss:.4f} | Perplexity: {perplexity:.2f}")
```

```{code-cell} output
Validation Loss: 3.2068 | Perplexity: 24.70
```

---

### 4.6 Summary of What We Learned

| Concept                | Description                                                             |
| ---------------------- | ----------------------------------------------------------------------- |
| **Cross-Entropy Loss** | Measures the divergence between predicted and true token distributions. |
| **Perplexity**         | Exponential of the average loss — lower means better prediction.        |
| **AdamW Optimization** | Balances gradient magnitudes and stabilizes training.                   |



## 5 Colab Notebook Link & Integration Notes

You now have a complete understanding of the mathematical foundation and a working example of how a Transformer is trained in practice.  
To reinforce this, we provide a companion **Google Colab notebook** that reproduces all experiments and visualizations interactively.

---

### 5.1 Notebook Overview

The Colab notebook (`llm_training.ipynb`) contains:

| Section | Interactive Content |
|----------|--------------------|
| **1. Setup & Imports** | Environment setup for PyTorch and Hugging Face |
| **2. Cross-Entropy Loss** | Step-by-step derivation with code and gradient verification |
| **3. Gradient Flow & Normalization** | Visualize gradient magnitudes across Transformer layers |
| **4. Mini Transformer Training** | Train a two-layer Transformer on a subset of WikiText-2 |
| **5. Visualizations** | Real-time loss output |
| **6. Evaluation** | Compute perplexity and analyze overfitting behavior |

---

### 5.2 Launch in Colab

Click below to open the notebook directly in Google Colab and run the full training pipeline:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DCMoe/MAP6905_Adv_Deep_Learning/blob/main/notebooks/llm_training.ipynb)


---

### 5.3 Reproducibility Tips

```{admonition} ⚙️ Best Practices for Running the Notebook
- **Runtime:** Use GPU acceleration in Colab (Runtime → Change runtime type → GPU).  
- **Batch Size:** Keep small (≤16) for fast iteration.  
- **Random Seeds:** Set `torch.manual_seed(0)` for reproducibility.  
- **Epochs:** 3–5 epochs are sufficient for visualization.  
- **Logging:** Track loss via a simple list or use `wandb` for extended tracking.
```

---

### 5.5 Summary

* The **Transformer’s math** links directly to optimization behavior through well-behaved gradients and normalization.
* The **cross-entropy objective** trains the model to minimize uncertainty (perplexity).
* Even a **tiny Transformer** demonstrates the same dynamics seen in billion-parameter models.

