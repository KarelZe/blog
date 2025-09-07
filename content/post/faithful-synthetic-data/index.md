---
title: My thoughts on How Faithful is Your Synthetic Data?🛸
date: 2025-09-07T07:52:00+02:00
Description: My thoughts on the paper titled 'How faithful is your synthetic data?'.
Tags: [synthetic-data, data-sampling, visual-data, vision-language-models, paper]
Categories: [ai]
DisableComments: false
---

Training AI models on synthetic data is a data scientist's (and management's) dream come true. It's easy to generate in vast amounts, contains no labelling errors, and privacy concerns are virtually nonexistent. However, a frequently overlooked aspect is how to assess the quality of these synthetic samples. How can we construct rich synthetic datasets that both mimic the properties of real data and introduce genuine novelty? These are challenges I frequently face in my daily work at [atruvia](https://atruvia.de/).

A paper by Alaa et al. titled "How Faithful is Your Synthetic Data? Sample-Level Metrics for Evaluating and Auditing Generative Models"[^1] tries to shed light on these questions. More specifically, it introduces a 3-dimensional metric to assess the quality of generative models. According to the authors, the new metric is both *domain* and *model-agnostic*. Its novelty lies in being computable at the sample level (hurray 🎉), making it interesting for selecting high-quality synthetic samples for purely synthetic or hybrid datasets. Let's examine if it holds up to scrutiny.

## Core contribution - $\\alpha$-precision, $\\beta$-recall, and authenticity

## Problems of existing approaches

## Debugging different modes of failures

## Practical implementation

## Does it scale?

## Discussion

Synthetic data is often seen as the holy grail for your data problems.

[^1]: see https://arxiv.org/abs/2102.08921
