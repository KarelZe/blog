---
bibliography: library.bib
Categories:
- ai
csl: chicago-author-date.csl
date: "2026-03-02T07:52:00+02:00"
description: My thoughts on the paper 'Be like a Goldfish, Don't Memorize!' and how it mitigates memorization in LLMs.
DisableComments: false
images:
- images/thumbnail_goldfish_loss.png
link-citations: true
Tags:
- softmax
- ring-attention
- paper
- ml
title: 🧨 Understanding Online Softmax
---

Softmax is foundational to modern large-language models (LLMs). It normalizes attention scores (([Vaswani et al. 2017](#ref-vaswaniAttentionAllYou2017))) and converts logits into a probability distribution for sampling.

In this post, we explore how Softmax is implemented in most frontier models. A recent discussion at work on computing attention over millions of tokens sent me down this engineering rabbit hole.

## Softmax

Let's first look at good-old Softmax. The Softmax function converts a vector of real numbers into a probability distribution.

The formula is given by: $$ \operatorname{softmax}: \mathbb{R}^n \rightarrow \mathbb{R}^n$$ $$ \operatorname{softmax}(\mathbf{x})_i = \frac{e^{x_i}}{\sum_{j=1}^n e^{x_j}} $$
where:
- $\mathbf{x} \in \mathbb{R}^n$ is an input vector (e.g., logits over a vocabulary) of size $n$.
- The output is a probability distribution over the $n$ elements. All elements lie in $[0,1]$ and sum to $1$.

Softmax uses the exponential function to amplify differences between values. The exponential function ensures *non-negativity* and as it is strictly monotonic, probabilities increase for larger values of $x_i$. Dividing by the sum of all exponentials normalizes the result into a valid probability distribution (([Zhang et al. 2021](#ref-zhangDiveDeepLearning2021))).

*example (adapted from ([Simon Oz 2024](#ref-simonozHowWriteFast2024))):*

$$
\left[\begin{array}{l}
3 \\
5 \\
2
\end{array}\right] \xrightarrow{\operatorname{softmax}}\left[\begin{array}{l}
\frac{e^3}{e^3+e^5+e^2} \\
\frac{e^5}{e^3+e^5+e^2} \\
\frac{e^2}{e^3+e^5+e^2}
\end{array}\right]=\left[\begin{array}{l}
0.114 \\
0.844 \\
0.042
\end{array}\right]
$$

Such a naive implementation of Softmax is prone to *numerical overflow* if input values $x_i$ become large. If calculations are done in standard `float16`, the maximum supported number would be 65536, which means that for $x \geqslant 11$, $e^x$ would exceed the effective range of `float16` ([Ye, n.d.](#ref-yeOnlineSoftmaxFlashAttention)). [^1] Interestingly, for `bfloat16` overflows would occur much later, as the exponent range is the same as with `float32` (["BFloat16," n.d.](#ref-BFloat16SecretHigh)). The overflow threshold would jump to roughly $x \geqslant 89$.

*example (continued):*

$$
\left[\begin{array}{l}
999 \\
988 \\
997
\end{array}\right] \xrightarrow{\operatorname{softmax}}\left[\begin{array}{l}
\frac{e^{999}}{e^{999}+e^{988}+e^{997}} \\
\frac{e^{988}}{e^{999}+e^{988}+e^{997}} \\
\frac{e^{997}}{e^{999}+e^{988}+e^{997}}
\end{array}\right]=\left[\begin{array}{c}
\text { nan } \\
\text { nan } \\
\text { nan }
\end{array}\right]
$$
\## Numerically Stable Softmax

To make Softmax numerically stable, the maximum input value ($\max(\mathbf{x})$) is subtracted from each element to prevent overflow:

$$ \operatorname{softmax}(\mathbf{x})_i = \frac{e^{x_i - \max(\mathbf{x})}}{\sum_{j=1}^n e^{x_j - \max(\mathbf{x})}} $$
While being mathematically equivalent, this trick ensures all exponents are non-positive (i.e., $x_i - \max(\mathbf{x})\leqslant 0$) and the negative inputs produce outputs $\leqslant 1$, which prevents overflow ([Ye, n.d.](#ref-yeOnlineSoftmaxFlashAttention)).

*example (continued):*
$$
\left[\begin{array}{c}
999 \\
988 \\
997
\end{array}\right] \xrightarrow{\text { softmax }}\left[\begin{array}{c}
\frac{e^{999-999}}{e^{999-999}+e^{988-999}+e^{997-999}} \\
\frac{e^{988-999}}{e^{999-999}+e^{988-999}+e^{997-999}} \\
\frac{e^{997-999}}{e^{999-999}+e^{988-999}+e^{997-999}}
\end{array}\right]=\left[\begin{array}{c}
0.881 \\
0.000 \\
0.119
\end{array}\right]
$$

## Online-Softmax

``` python
# TODO:
```

<div id="refs" class="references csl-bib-body hanging-indent">

<div id="ref-BFloat16SecretHigh" class="csl-entry">

"BFloat16: The Secret to High Performance on Cloud TPUs." n.d. In *Google Cloud Blog*. Https://cloud.google.com/blog/products/ai-machine-learning/bfloat16-the-secret-to-high-performance-on-cloud-tpus.

</div>

<div id="ref-simonozHowWriteFast2024" class="csl-entry">

Simon Oz. 2024. *How to Write a Fast Softmax Kernel*.

</div>

<div id="ref-vaswaniAttentionAllYou2017" class="csl-entry">

Vaswani, Ashish, Noam Shazeer, Niki Parmar, et al. 2017. "Attention Is All You Need." *Advances in Neural Information Processing Systems* (Long Beach, CA), NeurIPS 2017, vol. 30: 6000--6010.

</div>

<div id="ref-yeOnlineSoftmaxFlashAttention" class="csl-entry">

Ye, Zihao. n.d. *From Online Softmax to FlashAttention*.

</div>

<div id="ref-zhangDiveDeepLearning2021" class="csl-entry">

Zhang, Aston, Zachary C Lipton, Mu Li, and Alexander J Smola. 2021. *Dive into Deep Learning*.

</div>

</div>

[^1]: Note that $e^{11} ≈ 59874$, but we also need to compute the sum over all exponentials.
