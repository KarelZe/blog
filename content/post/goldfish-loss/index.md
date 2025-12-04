---
title: "My thoughts on “Be like a Goldfish, Don't Memorize!”"
date: 2025-12-02T07:52:00+02:00
description: "My thoughts on the paper 'Be like a Goldfish, Don't Memorize!' and how it mitigates memorization in LLMs."
Tags: ["causal-language-modelling", "llm", "memorization", "paper", "ml"]
Categories: ["ai"]
DisableComments: false
thumbnail: images/thumbnail_goldfish_loss.png
images:
  - images/thumbnail_goldfish_loss.png
bib: goldfish
---

Training large language models (LLMs) on vast datasets is a double-edged sword. While we want them to learn general patterns, we must strictly avoid the verbatim memorization of sensitive data from the training corpus. A '24 NEURIPS paper titled "Be like a Goldfish, Don't Memorize!" {{< cite hansBeGoldfishDont2024 >}} introduces a surprisingly simple approach to address this issue: the *Goldfish Loss*.

The novel idea is to exclude specific tokens from the loss calculation during training, instead of incorporating all tokens up to the predicted one. This effectively forces the model to learn generalizable patterns instead of relying on rote memorization. Just like a goldfish with its famously short memory, this loss function forces the model to 'forget' specific tokens during training.[^1] Let's first understand why this matters.

## The Problem of Memorization

Memorization means that a generative model, like an LLM, fails to generalize and either copies or nearly replicates training samples in regions of the input space with poor coverage of training samples.[^2] Memorization in LLMs poses a severe risk to both LLM developers and data donors, whose data eventually end up in a training corpus. Risks brought up by the authors include:

*   **Copyright Risk for Providers/Customers:** If a model memorizes lyrics, books, or copyrighted code, it can reproduce them verbatim, leading to uncertainties and potential lawsuits for those hosting the models and consuming the output. Recent practical examples include the lawsuit against Meta for training Llama 3 on Anna's Archive and LibGen [^3] or a lawsuit by German songwriter Helene Fischer (represented by GEMA) against OpenAI for memorizing the lyrics of "Atemlos durch die Nacht"[^4],[^5].
*   **Privacy Risks:** Memorization in LLMs can also lead to leakage of personally identifiable or sensitive information. Remember the early days, when you could trick ChatGPT to leak real email footers and other personally identifiable information because the model had memorized them from the training corpus? [^6]

No wonder European regulators are increasingly pushing for measures to assess memorization. In my daily work at [Atruvia](https://atruvia.de/), I also have to assess the risk of memorization, conduct analysis, and implement countermeasures for our own models. Let's see if the *Goldfish Loss* could come to our rescue.

## The Goldfish Loss

The authors propose *Goldfish Loss (GL)*, a modification to the standard training objective used in Causal Language Modelling (CLM).

### The Standard Causal Language Modelling Objective

Standard CLM trains the model to predict the next token $x_i$ given all previous tokens $x_{<i}$. Tokens are nowadays mostly sub-words; e.g., the tokenizer of GPT-4 would split `Bilz` into `B`, `il`, `z`.[^7] The loss is calculated for *every* token in the sequence $x=\left\{x_i\right\}$ of $L$ training tokens, where $\theta$ represents the model parameters:

$$
\mathcal{L}(\theta)=-\frac{1}{L} \sum_{i=1}^L \log P\left(x_i \mid x_{<i} ; \theta\right).
$$

The objective is minimized if the model correctly predicts the entire sequence $\left\{x_i\right\}$ with high confidence. What you should remember: *all tokens* contribute to the final loss.

Here's a naive python implementation:
```python
import torch
import torch.nn.functional as F

def compute_clm_loss(logits: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    """Compute standard CLM loss on all tokens.

    Args:
        logits (torch.Tensor): Model predictions [batch_size, seq_len, vocab_size]
        tokens (torch.Tensor): Target tokens [batch_size, seq_len]

    Returns:
        torch.Tensor: loss.
    """
    # Shift: predict token i+1 from tokens 0..i
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = tokens[:, 1:].contiguous()

    # Reshape for cross-entropy calculation
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)
    )
    return loss
```

Now we are all set for the *Goldfish Loss*.

### The Goldfish Loss

The *GL* modifies this by randomly masking a subset of tokens during the loss calculation to mitigate verbatim generation of memorized training samples. Specifically, it drops $1/k$ of the tokens.

$$
\mathcal{L}_{\text{goldfish}}(\theta)=-\frac{1}{|G|} \sum_{i=1}^L G_i \log P\left(x_i \mid x_{<i} ; \theta\right)
$$

where $G_i \in \{0, 1\}$ is a binary mask. If $G_i = 0$, the token is ignored in the loss and contributes otherwise.

By intuition, hyperparameter $k$ controls the aggressiveness of masking. For very large values of $k$, the GL approaches the standard CLM objective, since $\lim_{k \to \infty} \frac{1}{k} = 0$ means almost no tokens are masked. In the paper the authors set $k=4$, meaning 25% of all tokens are dropped.

![meme on forgetful dory from finding nemo](meme-dory.jpg)

As for $G$, the mask is *pseudo-random*, meaning that a passage is always masked *in the same manner*, unless the sequence is ever-so-slightly different (wait for the section on limitations).[^9] We will discuss in the next section how to arrive at such a mask.

For now, I'd like to stress the following aspects:

1.  **Forward Pass:** The model still sees *all* tokens in the context. It's not masking like in BERT {{<cite "devlinBERTPretrainingDeep2019">}} or tabular pre-training objectives like the of FT-Transformer {{<cite "gorishniyRevisitingDeepLearning2021">}}, where the input is corrupted. The input remains intact!
2.  **Backward Pass:** The loss is only computed for the *unmasked tokens*. The model is never explicitly penalized for failing to predict the masked tokens, so it doesn't "learn" them as strongly. Critically, at *inference* time, the model must predict *all* tokens (including those that were masked during training). For identical sequences, the model must make an unsupervised guess for previously masked tokens, causing it to diverge from the training sequence and thereby impeding verbatim reproductions.

Here's a python implementation, adapted from the author's supplemental material [^11]:
```python
import torch
import torch.nn.functional as F

def compute_goldfish_loss(logits: torch.Tensor, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Compute Goldfish loss only on unmasked tokens.

    Args:
        logits (torch.Tensor): Model predictions [batch_size, seq_len, vocab_size]
        tokens (torch.Tensor): Target tokens [batch_size, seq_len]
        mask (torch.Tensor): Binary mask [batch_size, seq_len] (1 = compute loss, 0 = skip)

    Returns:
        torch.Tensor: loss.
    """
    # Shift: predict token i+1 from tokens 0..i
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = tokens[:, 1:].contiguous()
    shift_mask = mask[:, 1:].contiguous()

    # Reshape for cross-entropy calculation
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction='none'  # Don't reduce yet ;)
    )

    # Apply mask and compute mean only over unmasked positions
    mask_flat = shift_mask.view(-1)
    masked_loss = loss * mask_flat
    return masked_loss.sum() / mask_flat.sum()
```

### The Hashed Mask

Let's now focus on the token mask, the second main contribution of the paper.

Recall that most language models are trained on internet corpora and the internet is a fuzzy place[^10]. Texts may be copied around the web (BuzzFeed, I mean you), may be embedded into larger texts, or restructured; data curation thus makes up a large part of the effort spent on LLM training.

Ideally, we'd like to mask the same passages identically to prevent leakage.

Naive approaches, like masking every $k$-th token (referred to in the paper as *static mask*), don't help much here, as the mask would be aligned to the pre-training sequence and deviate if the text were chunked or prefixed differently. Eventually, the model could see (and learn) every token. Feel free to experiment with the downsides of *static masking* in the interactive visualization.

Another naive idea could be to mask purely randomly. If masks were purely random (referred to as *random mask* in the paper), however, the model could learn every token over the course of several epochs or from differently masked duplicates, impeding our original goal.

Thus, we need a mask that is:

- deterministic
- independent from the absolute position of a sequence within a longer sequence

Hence, the authors propose a *localized hashed mask*. The decision to mask a token $x_i$ is deterministic based on its immediate preceding context (the previous $h$ tokens) and the output of a hash function $\operatorname{hash}:|V|^h \rightarrow \mathbb{R}$. We mask $x_i$ (i.e., set $G_i=0$) if:
$$
\operatorname{hash}(x_{i-h}, \dots, x_{i-1}) <  \frac{1}{k} \implies \text{G}_i = 0
$$

Note that with the context width $h$, we introduce another hyperparameter that needs to be set carefully. An example from the paper makes this very clear: If $h=7$ is used, the model may never learn to produce the word "Power" at the end of the phrase "the Los Angeles Department of Water and Power.". Definitely unsatisfying. Equally, $h$ should not be too large, as then the hash is underdetermined for the first $h-1$ tokens in the document. In the reference implementation, the context width defaults to $h=4$.

A nerdy implementation detail: You might wonder what happens to the first few tokens of a document. Since they don't have enough preceding tokens to form a full context of size $h$, we can't compute a hash for them. Therefore, the first $h-1$ tokens are never masked (i.e., always included in the loss).

Now it's your turn to play. Adjust the slider below to see how the parameter $k$ affects which tokens are masked. Adjust the text and suffixes. You can also switch between *static mask* and the *hashed mask*. I also recommend varying punctuation to see how it affects masking.

{{< goldfish-slider >}}

Here's a Python implementation, adapted from the author's reference implementation, which uses a performant hash-table-based approach [^11].

```python
import torch

# Initialize a global hash table (simulated)
TABLE_SIZE = 1_000_003  # Choose large prime
HASH_TABLE = torch.rand(TABLE_SIZE)

def generate_hashed_mask(tokens: torch.Tensor, k: int = 4, context_width: int = 4) -> torch.Tensor:
    """Generate deterministic mask using a hash table strategy.

    Args:
        tokens (torch.Tensor): Tensor of token IDs
        k (int): Masking parameter (masks ~1/k of tokens). Defaults to 4.
        context_width (int): Number of tokens in the context window (h). Defaults to 4.

    Returns:
        torch.Tensor: Binary mask tensor [seq_len] where 1 = compute loss, 0 = skip
    """
    seq_len = tokens.size(0)
    mask = torch.ones(seq_len) # Don't mask by default

    # We can only mask if we have enough context
    if seq_len < context_width:
        return mask

    # Create sliding windows of size 'context_width'
    # Result shape: [num_windows, context_width]
    windows = tokens.unfold(0, context_width, 1)

    # Compute a hash for each window
    window_hashes = windows.prod(dim=1) % TABLE_SIZE

    random_values = HASH_TABLE[window_hashes]

    # Determine which to drop: value < 1/k
    # These correspond to tokens at indices [context_width-1, seq_len-1]
    tokens_to_drop = random_values < (1.0 / k)

    # Apply drops to the mask
    # We offset by (context_width - 1) because the first window ends at index (context_width - 1)
    mask[context_width-1:][tokens_to_drop] = 0.0

    return mask

# Example usage
tokens = torch.tensor([101, 2054, 2003, 1037, 2003, 1037, 2003, 1037])
mask = generate_hashed_mask(tokens, k=4, context_width=4)
print(mask)
```

Two remarks on the code:
- The hash function is the product of the tokens in the window. As reordered tokens produce the same hash within context, it may not always be the best design choice.
- The hash table should be reasonably large.

## Experiments & Results

The authors tested *Goldfish Loss* in diverse experiments w.r.t. memorization, training efficiency, generation quality, and robustness to adversarial attacks. For my humble blog post I'll focus on the first three.

They distinguish between two setups:
- **Extreme Setup (aka Recipe For Disaster 🤓):** a LLaMA-2-7B model for 100 epochs on a small set of Wikipedia articles. Temperature set to $0$. This setup is aimed to promote memorization.
- **Standard Setup:** A TinyLLaMA-1.1B model trained for 1 epoch. Once more they use greedy decoding.
- In both setups, the test sets consists of a subsample of training sequences, that have been split into a prefix and a length of $n$ tokens.

Memorization is quantified in terms of *extractable memorization* {{< cite carliniQuantifyingMemorizationNeural2023 >}} and *RougeL scores* {{< cite lin-2004-rouge >}}:
-  *Extractable memorization*  measures the LLM's ability to reproduce a training sequence verbatim given a prefix/prompt of length $p$ with greedy decoding.
- *RougeL scores* quantify the longest common, but not necessarily consecutive, subsequence of tokens shared with the sequence from the training set.

Here's a comparison of how the two metrics differ:

![exact-match-vs-rouge-l](rouge-l-exact-match.png)

Common for both metrics, a score of $1$ indicates perfect memorization.

### Memorization in the Extreme Setup
They trained a LLaMA-2-7B model for **100 epochs** on a small set of Wikipedia articles—a recipe for disaster (memorization).
*   **Standard Training:** Memorized 84/100 articles verbatim.
*   **Goldfish Loss ($k=4$):** **Zero** verbatim memorization.

### Standard Setup

In the standard setup, Goldfish Loss still significantly reduced the model's ability to reproduce training sequences compared to standard CLM.

## Limitations

There are some caveats though:

1.  **Training Efficiency:** Since in a setup with goldfish loss, we are ignoring $1/k$ of the training tokens, the model learns "slower" per batch. You effectively need to train on more data (or for longer) to reach the same validation loss as a standard model. The authors, however demonstrate (rather convincingly) on the [RedPajamaV2 dataset](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-V2), that if we compare the validation loss for the supervised tokens (aka unmasked) tokens with an equal number of input tokens in a standard training setup, both models end up with an approximately an equal validation loss (see Fig. 5 in paper).
2.  **Near-Duplicates:** The approach is still prone to near-duplicates. You can spot this in the interactive visualization above easily. E.g., small rewrites or some added punctuation or different unicode-encoding, the hashed mask might be different for each version, allowing the model to piece together the full text from the different copies. (see Sec. 6.3 in paper)

## My thoughts

The goldfish loss is a clever, lightweight adaption of the CLM that can be easily dropped into existing training recipes. This is a big plus for practitioners with limited resources.

It offers a promising alternative for training powerful models that respect privacy-by-design, rather than relying on complex machine unlearning strategies. I agree with the authors, that it's most useful on high-risk sources or late phases of training e.g., fine-tuning.

Practically, the positive effects from the *GL* will only be as good as the engineering that went into normalization (see remarks in Sec. 3.1 of paper), filtering and removal of near-duplicates of the training corpus. Common practice of training on rewritten synthetic texts or near-identical synthetic texts based on real seeds need to be rethought, as both would impede consistent masking. [^13]

Lastly, I remain slightly skeptical about their copyright compliance angle:

> We hope that goldfish loss paves the way for aiding copyright compliance rather than serving as a means to misuse private data maliciously. (Sec. 7)

While their loss function prevents *verbatim* reproduction, the model still learns the *information* and *style* from the copyrighted works. Is a paraphrased text more copyright-compliant? That's a question for the courts, not the loss function.🪝

## Bibliography

{{< bibliography>}}


[^1]: More than allegedly. As a child, I used to have a small goldfish living in a large bowl.
[^2]: While conceptually similar to overfitting, an overfitted model would fit the training distribution too precisely including noise and idiosyncrasies and perform poorly on the true underlying distribution.
[^3]: [This article](https://www.theatlantic.com/technology/archive/2025/03/libgen-meta-openai/682093/) by *the Atlantic* gives a good overview incl. [a search tool](https://www.theatlantic.com/technology/archive/2025/03/search-libgen-data-set/682094/).
[^4]: More details can be found [here](https://www.gesetze-bayern.de/Content/Document/Y-300-Z-GRURRS-B-2025-N-30204?hl=true).
[^5]: Haters would say, that reproducing the lyrics verbatim isn't too hard.
[^6]: Read [this article](https://www.zdnet.com/article/chatgpt-can-leak-source-data-violate-privacy-says-googles-deepmind/) for some background information on the attack vector.
[^7]: You can play around with different tokenizers on [tiktokenizer.vercel.app](https://tiktokenizer.vercel.app/?model=gpt-4). It's also a nifty tool, if your models need to run on a tight budget.
[^9]: In this context, pseudo-random doesn't refer to pseudo-random number generators (the most common variant in modern computers), but rather to the fact that masking of tokens is done randomly and identical sequences will be masked identically. If you are interested in true random number generators, you can read [this article](https://blog.cloudflare.com/lavarand-in-production-the-nitty-gritty-technical-details/) on a creative approach to generate truly random numbers using lava lamps at cloudflare.
[^10]: For some interesting infographics see this [nature article](https://www.nature.com/articles/d41586-024-03990-2)
[^11]: For original source code see [here.](https://proceedings.neurips.cc/paper_files/paper/2024/hash/2ad2dffba5079687651226ac8752df97-Abstract-Conference.html)
[^13]: For applications of these techniques see e.g., the [technical report of Phi-4](https://arxiv.org/pdf/2412.08905)
