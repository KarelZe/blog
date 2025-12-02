---
title: "My thoughts on Be like a Goldfish, Don’t Memorize!🐠"
date: 2025-12-02T07:52:00+02:00
description: "My thoughts on the paper 'Be like a Goldfish, Don’t Memorize!' and how it mitigates memorization in LLMs."
Tags: ["causal-language-modelling", "llm", "memorization", "paper", "ml"]
Categories: ["ai"]
DisableComments: false
thumbnail: images/thumbnail_goldfish_loss.png
images:
  - images/thumbnail_goldfish_loss.png
---

Training AI models on vast datasets is a double-edged sword. While we want them to learn general patterns, we desperately want to avoid them memorizing sensitive data or copyrighted material verbatim. A recent paper titled **"Be like a Goldfish, Don’t Memorize!"** introduces a strikingly simple yet effective technique to tackle this: the **Goldfish Loss**.

The core idea? Make the model "forget" specific tokens during training, forcing it to learn generalizable patterns instead of rote memorization. Just like a goldfish (allegedly) has a short memory, this loss function prevents the model from recalling exact sequences.

## The Problem: Memorization Risks

Memorization in Large Language Models (LLMs) isn't just an academic curiosity; it's a significant liability.

*   **Copyright Infringement:** If a model memorizes lyrics or books, it can reproduce them verbatim, leading to lawsuits. A recent example is the lawsuit by **Helene Fischer vs. OpenAI**, where the model allegedly reproduced lyrics from her hit "Atemlos durch die Nacht". Similarly, lawsuits against Meta for training on Anna's Archive highlight the legal minefield of training data.
*   **Privacy Leaks:** Early versions of ChatGPT were shown to regurgitate real email footers and personal identifiable information (PII) because the model had memorized them from the training corpus.

> [!NOTE]
> **Memorization vs. Overfitting**: While related, they aren't identical. Overfitting usually means the model performs well on training data but poorly on unseen data. Memorization specifically refers to the ability to reproduce training examples verbatim. A model can be overfitted without memorizing everything, and conversely, a large model can memorize data even without classical overfitting (i.e., while still having low validation loss). European regulators are increasingly focusing on measures to prevent both.

## The Solution: Goldfish Loss 🐠

The authors propose **Goldfish Loss (GL)**, a modification to the standard training objective.

### How it Works

Standard **Causal Language Modeling (CLM)** trains the model to predict the next token $x_i$ given all previous tokens $x_{<i}$. The loss is calculated for *every* token in the sequence.

$$
\mathcal{L}(\theta)=-\frac{1}{L} \sum_{i=1}^L \log P\left(x_i \mid x_{<i} ; \theta\right)
$$

```python
# Standard Causal Language Modeling Loss (PyTorch)
import torch
import torch.nn.functional as F

def compute_clm_loss(logits, tokens):
    """Compute standard CLM loss on all tokens.

    Args:
        logits: Model predictions [batch_size, seq_len, vocab_size]
        tokens: Target tokens [batch_size, seq_len]
    """
    # Shift: predict token i+1 from tokens 0..i
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = tokens[:, 1:].contiguous()

    # Flatten the tokens
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)
    )
    return loss
```

**Goldfish Loss** modifies this by randomly masking a subset of tokens during the loss calculation. Specifically, it drops $1/k$ of the tokens.

$$
\mathcal{L}_{\text {goldfish }}(\theta)=-\frac{1}{|G|} \sum_{i=1}^L G_i\left(x_i\right) \log P\left(x_i \mid x_{<i} ; \theta\right)
$$

where $G_i \in \{0, 1\}$ is a mask. If $G_i = 0$, the token is ignored in the loss.

```python
# Goldfish Loss (PyTorch)
import torch
import torch.nn.functional as F

def compute_goldfish_loss(logits, tokens, mask):
    """Compute Goldfish loss only on unmasked tokens.

    Args:
        logits: Model predictions [batch_size, seq_len, vocab_size]
        tokens: Target tokens [batch_size, seq_len]
        mask: Binary mask [batch_size, seq_len] (1 = compute loss, 0 = skip)
    """
    # Shift: predict token i+1 from tokens 0..i
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = tokens[:, 1:].contiguous()
    shift_mask = mask[:, 1:].contiguous()

    # Flatten the tokens
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction='none'  # Don't reduce yet
    )

    # Apply mask and compute mean only over unmasked positions
    mask_flat = shift_mask.view(-1)
    masked_loss = loss * mask_flat
    return masked_loss.sum() / mask_flat.sum()
```

**Crucially:**
1.  **Forward Pass:** The model still sees *all* tokens in the context. It's not like masking in BERT where the input is corrupted. The input is intact.
2.  **Backward Pass:** The loss is only computed for the unmasked tokens. The model is never explicitly penalized for failing to predict the masked tokens, so it doesn't "learn" them as strongly.

![Goldfish Loss Diagram](/images/goldfish_loss_diagram.png)

### The Hashed Mask

You might wonder: "Why not just drop random tokens?"
If you use a truly random mask (like a coin flip), the model might see the same text multiple times across epochs with *different* masks. Eventually, it would see (and learn) every token.

To prevent this, the authors use a **Hashed Mask**. The decision to mask a token $x_i$ is deterministic based on its preceding context (the previous $h$ tokens).
$$
\text{hash}(x_{i-h}, \dots, x_{i-1}) \mod k == 0 \implies \text{Mask } x_i
$$
This ensures that whenever the model encounters the specific phrase "The cat sat on the...", it *always* skips learning the word "mat". It's a "pseudo-random" mask that is consistent across training epochs.

```python
# Hashed Mask Generation (PyTorch)
import torch
import hashlib

def generate_hashed_mask(tokens, k, context_width=1):
    """Generate deterministic mask based on context hash.

    Args:
        tokens: List of token strings or token IDs
        k: Masking parameter (masks ~1/k of tokens)
        context_width: Number of preceding tokens to use for hash (h)

    Returns:
        Binary mask tensor [seq_len] where 1 = compute loss, 0 = skip
    """
    mask = []
    for i in range(len(tokens)):
        # Get context: previous h tokens (or "START" if at beginning)
        if i < context_width:
            context = "START"
        else:
            # Convert token IDs to strings if needed
            context_tokens = [str(t) for t in tokens[i-context_width:i]]
            context = " ".join(context_tokens)

        # Hash the context to get deterministic value
        hash_value = int(hashlib.md5(context.encode()).hexdigest(), 16)

        # Mask if hash % k == 0 (drops ~1/k tokens)
        mask.append(0 if hash_value % k == 0 else 1)

    return torch.tensor(mask, dtype=torch.float32)

# Example usage
tokens = ["The", "cat", "sat", "on", "the", "mat"]
mask = generate_hashed_mask(tokens, k=4, context_width=1)
print(mask)  # tensor([1., 0., 1., 1., 0., 1.]) (deterministic for this sequence)

# For batched inputs
def generate_hashed_mask_batch(token_ids, k, context_width=1):
    """Generate masks for a batch of sequences.

    Args:
        token_ids: Token IDs [batch_size, seq_len]
        k: Masking parameter
        context_width: Number of preceding tokens (h)

    Returns:
        Binary mask tensor [batch_size, seq_len]
    """
    batch_size, seq_len = token_ids.shape
    masks = []
    for b in range(batch_size):
        mask = generate_hashed_mask(token_ids[b].tolist(), k, context_width)
        masks.append(mask)
    return torch.stack(masks)
```

> [!NOTE]
> **True vs. Pseudo-Randomness**: This distinction is vital. In cryptography, we need true randomness (like Cloudflare using **lava lamps** to generate entropy). Here, we *want* determinism disguised as randomness to ensure consistent masking.

### Interactive Visualization

Adjust the slider below to see how the parameter $k$ affects which tokens are masked. You can also switch between **Static Mask** (masking every $k$-th token) and **Hashed Mask** (masking based on the hash of the context).

{{< goldfish-slider >}}

## Experiments & Results

The authors tested Goldfish Loss against standard training using **"Extractable Memorization"** (defined by Carlini et al. as the ability to reproduce a training example given a prefix).

### Extreme Setup
They trained a LLaMA-2-7B model for **100 epochs** on a small set of Wikipedia articles—a recipe for disaster (memorization).
*   **Standard Training:** Memorized 84/100 articles verbatim.
*   **Goldfish Loss ($k=4$):** **Zero** verbatim memorization.

### Standard Setup
On a more realistic setup (TinyLLaMA-1.1B, single epoch), Goldfish Loss still significantly reduced the model's ability to reproduce training sequences compared to standard CLM.

> [!NOTE]
> **Dropout vs. Goldfish Loss**: While both are regularization techniques, they differ fundamentally. **Dropout** randomly disables neurons (architecture) to prevent feature co-adaptation. **Goldfish Loss** disables loss computation for specific data points (objective) to prevent verbatim recall.

## Limitations

There is no free lunch.
1.  **Training Efficiency:** Since you are ignoring $1/k$ of the signals, the model learns "slower" per batch. You effectively need to train on more data (or for longer) to reach the same validation loss as a standard model.
2.  **Near-Duplicates:** If the training data contains near-duplicates (e.g., the same article with a slightly different header), the hashed mask might be different for each version, allowing the model to piece together the full text from the different copies.

## Conclusion

Goldfish Loss is a clever, lightweight intervention that can be easily dropped into existing training pipelines. It offers a promising path for training powerful models that respect privacy and copyright by design, rather than by post-hoc filtering.

> We hope that goldfish loss paves the way for aiding copyright compliance rather than serving as a means to misuse private data maliciously.

However, I remain slightly skeptical about the "copyright compliance" angle. While it prevents *verbatim* reproduction, the model still learns the *information* and *style* from the copyrighted works. Is a paraphrased copy compliant? That's a question for the courts, not the loss function.

## References

*   Tyen et al., "Be like a Goldfish, Don’t Memorize!", 2024. [arXiv](https://arxiv.org/abs/2404.02936)
*   Carlini et al., "Quantifying Memorization Across Neural Language Models", 2023.
*   [Understanding Evaluation Metrics for Language Models](https://thegradient.pub/understanding-evaluation-metrics-for-language-models/)

```bibtex
@article{bilz2025goldfish,
  title={My thoughts on Be like a Goldfish, Don’t Memorize!},
  author={Bilz, Markus},
  journal={Markus Bilz Blog},
  year={2025},
  url={https://blog.markusbilz.com/post/goldfish-loss/}
}
```
