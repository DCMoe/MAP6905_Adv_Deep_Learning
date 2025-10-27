# Inference, Prompting, and Fine-tuning

```{note}
**Learning Objectives**

By the end of this module, you will be able to:

- Explain how inference differs from training in a language model.  
- Describe the mechanics of token generation and decoding strategies.  
- Understand prompting as conditional inference.  
- Implement small-scale fine-tuning methods such as **LoRA**.  
- Explain the motivation for quantization during deployment.
```

---

## 1 From Training to Inference

During **training**, we minimize loss over known tokens.
During **inference**, we *sample* or *decode* the next token given context:

$$
\hat{x}*{t+1} = \arg\max*{x} P_\theta(x \mid x_{\le t})
$$

Inference proceeds **autoregressively** until a stop condition (e.g., `<eos>`).

---

### 1.1 Autoregressive Generation Loop

```{code-cell} python
import torch
from torch.nn import functional as F

def generate(model, tokenizer, prompt, max_new=20, temperature=1.0):
    tokens = tokenizer(prompt, return_tensors="pt")["input_ids"]
    for _ in range(max_new):
        logits = model(tokens).logits[:, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        tokens = torch.cat([tokens, next_token], dim=1)
    return tokenizer.decode(tokens[0])
```

---

## 2 Decoding Strategies

| Strategy            | Description                                             | Behavior                          |
| :------------------ | :------------------------------------------------------ | :-------------------------------- |
| **Greedy**          | Choose highest-prob token each step                     | Deterministic, may repeat         |
| **Sampling**        | Sample according to probabilities                       | Stochastic, diverse               |
| **Top-k**           | Keep top-k tokens, renormalize                          | Controlled randomness             |
| **Top-p (nucleus)** | Keep smallest set of tokens whose cumulative prob ≥ *p* | Adaptive diversity                |
| **Beam search**     | Track *k* best sequences                                | Balances exploration/exploitation |

```{code-cell} python
import numpy as np, matplotlib.pyplot as plt
probs = np.sort(np.random.dirichlet(np.ones(50)))[::-1]
cum = np.cumsum(probs)
plt.step(range(len(probs)), cum)
plt.axhline(0.9, color='r', ls='--', label='p=0.9')
plt.title("Top-p nucleus threshold")
plt.xlabel("Token rank"); plt.ylabel("Cumulative probability")
plt.legend(); plt.show()
```

```{figure} ../images/mod4_top-p.png
---
height: 280px
name: top-p-decode
---
```

---

## 3 Prompting as Conditional Inference

Prompts condition the model’s probability distribution:

$$
P(y \mid \text{prompt}) =
\prod_{t} P(y_t \mid \text{prompt}, y_{<t})
$$

Types of prompting:

| Type                 | Example                          | Use                 |
| :------------------- | :------------------------------- | :------------------ |
| **Zero-shot**        | “Translate to French: *apple* →” | Pure inference      |
| **Few-shot**         | “Q: 2 + 2 A: 4 \n Q: 3 + 5 A:”   | In-context learning |
| **Chain-of-Thought** | “Let’s reason step-by-step.”     | Improves reasoning  |
| **Instruction**      | “Explain why the sky is blue.”   | Aligned outputs     |

> Prompting acts like temporary fine-tuning in context—no weights are changed.

---

## 4 Fine-tuning

Fine-tuning adapts pretrained weights to a downstream dataset.

### 4.1 Full fine-tuning

All parameters updated—high cost.

### 4.2 Parameter-efficient fine-tuning (PEFT)

Only small subsets of weights or adapters are trained.

---

#### 4.2.1 Low-Rank Adaptation (LoRA)

LoRA inserts low-rank matrices into linear layers:

$$
W\prime = W + \Delta W = W + A B^\top
$$

where

* $A \in \mathbb{R}^{d \times r}$, $B \in \mathbb{R}^{r \times k}$,
* rank $r \ll \min(d,k)$.

Only $A,B$ are trained; base weights $W$ stay frozen.

```{code-cell} python
import torch.nn as nn
class LoRALinear(nn.Module):
    def __init__(self, in_f, out_f, r=4, alpha=16):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_f, in_f))
        self.A = nn.Parameter(torch.randn(out_f, r))
        self.B = nn.Parameter(torch.randn(r, in_f))
        self.scale = alpha / r
    def forward(self, x):
        return x @ (self.weight.T + self.scale * (self.A @ self.B).T)
```

---

## 5 Quantization for Efficient Inference

Quantization compresses weights to lower precision to save memory and speed inference.

| Precision | Bits | Memory ↓ | Accuracy ↓ | Typical Use  |
| :-------- | :--: | :------: | :--------: | :----------- |
| FP16      |  16  |     –    |    none    | baseline     |
| INT8      |   8  |    2×    |    small   | deployment   |
| INT4      |   4  |    4×    |  moderate  | edge devices |

```{code-cell} python
from torch.ao.quantization import quantize_dynamic
quant_model = quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
```

> Quantized models are often paired with LoRA adapters for fast task adaptation.

---

## 6 Putting It All Together

1. **Prompting** guides inference without updating weights.
2. **LoRA fine-tuning** updates small adapters for specific tasks.
3. **Quantization** enables deployment on low-resource hardware.
4. Combined, these yield *efficient*, *adaptable* language models.

---

```{admonition} Next Module
Continue to: [Inference, Prompting, and Fine-tuning](VLMs/01_intro_vlms.md)
```

