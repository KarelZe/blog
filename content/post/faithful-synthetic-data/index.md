---
title: My thoughts on How Faithful is Your Synthetic Data?🛸
date: 2025-09-07T07:52:00+02:00
description: My thoughts on the paper "How Faithful is Your Synthetic Data?".
Tags: [synthetic-data, data-sampling, visual-data, vision-language-models, paper]
Categories: [ai]
DisableComments: false
---

Training AI models on synthetic data is a data scientist's (and management's) dream come true. It's easy to generate in vast amounts, contains no labeling errors, and privacy concerns are virtually nonexistent. However, a frequently overlooked aspect is how to assess the quality of these synthetic samples. How can we build rich synthetic datasets that both mimic the properties of real data and introduce genuine novelty? These are challenges I frequently face in my daily work at [atruvia](https://atruvia.de/).

A paper by Alaa et al. titled "How Faithful is Your Synthetic Data? Sample-Level Metrics for Evaluating and Auditing Generative Models"[^1] sheds light on these questions. More specifically, it introduces a three-dimensional metric to assess the quality of generative models. According to the authors, this new metric is both *domain-* and *model-agnostic*. Its novelty lies in being computable at the sample level (hurray 🎉), making it interesting for selecting high-quality samples for purely synthetic or hybrid datasets. Let's see if it holds up to scrutiny.

### What Makes a Good Synthetic Dataset?

Good synthetic data should fulfill the following three qualities:

1. **Fidelity:** A high-fidelity synthetic dataset should contain only "realistic" samples—for instance, photorealistic images. No [sixth fingers](https://medium.com/@sanderink.ursina/why-do-ai-models-sometimes-produce-images-with-six-fingers-da4cd53f3313) or [pasta-eating nightmares](https://en.wikipedia.org/wiki/Will_Smith_Eating_Spaghetti_test) in your dataset.
1. **Diversity:** The synthetic dataset should capture the full variability of the real data, including rare edge cases.
1. **Generalization:** A synthetic dataset should contain truly novel samples, not just be mere copies of the data the generative model was trained on. Without this third criterion, a strongly overfitted model could still score high on fidelity and diversity, but its outputs would just be copies, offering no real benefit.

All three aspects seem intuitive at first glance. As it happens, the authors propose a *three-dimensional* metric, $\\mathbf{\\Epsilon}$, that maps nicely to these three qualities.[^2] The mapping is as follows: $\\alpha$-precision captures *fidelity*, $\\beta$-recall captures *diversity*, and *authenticity* assesses *generalization*. While $\\alpha$-precision and $\\beta$-recall are generalizations of the classic precision and recall metrics, the concept of *authenticity* is what's truly new here. This focus on generalization versus memorization is one of the paper's key contributions. Let's take a closer look at the metric.

### The Core Metrics: α-Precision, β-Recall, and Authenticity

```yaml
TODO:
```

### A visual explanation α-Precision, β-Recall, and Authenticity

```yaml
TODO:
```

### Debugging Failure Modes with the Metric

```yaml
TODO:
```

### From Kittens to a Practical Implementation

```yaml
TODO:
```

### Does It Scale?

```yaml
TODO:
```

### Why Existing Metrics Fall Short

```yaml
TODO:
```

### Final Thoughts

```yaml
TODO:
```

[^1]: see https://arxiv.org/abs/2102.08921

[^2]: Seems like nobody has coined a name for the metric yet. Feel free to propose
