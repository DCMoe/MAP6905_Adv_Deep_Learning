---
title: LLM Training Objectives
---

```{note}
**Learning Objectives**

By the end of this module, you will be able to:

- Explain how language models learn via next-token prediction or token masking.  
- Derive the loss functions used in **causal** and **masked** language modeling.  
- Understand how token probabilities are computed from logits.  
- Implement a small-scale masked-LM and causal-LM example in PyTorch.  
- Visualize loss convergence and token prediction behavior.
```

---

## 1 Causal vs. Masked Language Modeling

| Objective           | Description                                      | Example Models   | Context Masking                 |
| :------------------ | :----------------------------------------------- | :--------------- | :------------------------------ |
| **Causal LM (CLM)** | Predict next token given all previous tokens     | GPT-2/3/4, LLaMA | Look-ahead masked (causal mask) |
| **Masked LM (MLM)** | Predict randomly masked tokens within a sequence | BERT, RoBERTa    | Bidirectional attention         |

### 1.1 Causal LM

At time $t$, predict the next token $x_t$ given $x_{<t}$:

$$
\mathcal{L}_{\text{CLM}}
=
- \sum_{t=1}^{T} \log P_\theta(x_t \mid x_{<t}) \tag{1}
$$

### 1.2 Masked LM

Mask a subset of tokens $M \subset \{1,\dots,T\}$, and train the model to predict them:

$$
\mathcal{L}_{\text{MLM}}
=
- \sum_{t \in M} \log P_\theta(x_t \mid x_{\setminus M}) \tag{2}
$$

---

### 1.3 Example Code

```{code-cell} python
import torch
from torch.nn import functional as F

# Toy vocabulary: 0=<mask>, 1=a, 2=cat, 3=sat, 4=mat
logits = torch.tensor([[2.0, 1.0, 0.5, -0.5, 0.2]])  # model output
target = torch.tensor([2])  # true token index ("cat")

loss = F.cross_entropy(logits, target)
print("Cross-entropy loss:", float(loss))
```
```{code-cell} output
Cross-entropy loss: 2.10889
```

---

## 2 Masking Strategies

In masked language modeling (BERT-style) {cite}`devlin2018bert`:

* **15 %** of tokens are selected for prediction.
* Of those:

  * 80 % are replaced with `[MASK]`
  * 10 % with a random token
  * 10 % kept unchanged (to stabilize learning)

> The goal is to make the model infer missing tokens from bidirectional context.

---

## 3 Visualization — Training Loss Curve

```{code-cell} python
import numpy as np, matplotlib.pyplot as plt
steps = np.arange(1,101)
loss = 2.0 * np.exp(-0.03*steps) + 0.05*np.random.randn(100)

plt.plot(steps, loss, lw=2)
plt.xlabel("Training Steps")
plt.ylabel("Loss")
plt.title("Loss Convergence during Language Model Training")
plt.grid(True)
plt.show()
```

```{figure} ../images/mod3_loss.png
---
height: 280px
name: training-loss-curve
---
```

---

## 4 Summary

| Concept            | Formula                                                     | Interpretation                          |
| :----------------- | :---------------------------------------------------------- | :-------------------------------------- |
| **Causal LM Loss** | $\mathcal{L}=-\sum_t\log P(x_t\mid x_{<t})$                 | Next-token prediction                   |
| **Masked LM Loss** | $\mathcal{L}=-\sum_{t\in M}\log P(x_t\mid x_{\setminus M})$ | Fill-in-the-blank training              |
| **Softmax Prob.**  | $p_i = e^{z_i}/\sum_j e^{z_j}$                              | Converts logits to probabilities        |
| **Cross-Entropy**  | $\mathcal{L}=-\sum_i y_i\log p_i$                           | Penalizes low probability on true label |

---

## References

```{bibliography}
:style: unsrt
:filter: False

brown2020language
radford2019language
touvron2023llama
devlin2018bert
goodfellow2016deep
vaswani2017attention
```