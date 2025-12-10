---
bibliography: library.bib
categories:
- ai
csl: chicago-author-date.csl
date: "2025-12-10T11:00:00+02:00"
description: My thoughts on the paper "Nested Tokenization for Larger Context in Large Images".
disableComments: false
draft: false
images:
- images/thumbnail_nested_tokenization.png
tags:
- tokenization
- vision understanding
- visual data
- vision-language-models
- paper
thumbnail: images/thumbnail_nested_tokenization.png
title: My thoughts on "Nested Tokenization for Larger Context in Large Images" ✂️
---

Processing images in Large Multimodal Models (LMMs) can be a tough challenge. Sometimes, we want the model to focus on the big picture only while other times we need to focus on tiny details in small parts of the image. Ideally, no tokens or inference time should be wasted on *irrelevant* regions.

I face this challenge frequently at work. At [Atruvia](https://atruvia.de/), we solve many document extraction tasks using LMMs. Images may be mostly whitespace, but can contain tiny details, such as [diacritics](https://en.wikipedia.org/wiki/Diacritic) in names like ◌̣ or delimiters in amounts that matter. In recent years, the size of image uploads has exploded due to higher camera resolutions. Wouldn't it be great if we only processed the relevant parts of the image, especially at high resolution? That's where the paper *"xT: Nested Tokenization for Larger Context in Large Images"* by Gupta et al. (2024) might come in handy.

Its core idea is simple. The paper introduces a two-stage framework, named $xT$, that allows existing vision backbones to process large-scale images at a fraction of the memory and compute cost. To accomplish this, $xT$ employs a divide-and-conquer strategy, referred to as *nested tokenization*: First, images are divided into coarse regions, then patches containing local details, which are passed through a *hierarchical region encoder* to obtain enriched region encodings. Subsequently, in a *context encoder*, the context from other regions is encoded onto the region encodings to create a sequence of context-aware region encodings, ready to be passed to a decoder.

The savings are twofold: As the region encoding is independent from other image regions and happening on smaller image regions, it can be done sequentially requiring less memory and compute. Also, the sequence of contextualized region encodings is typically shorter than in a comparable setup. Before going into the nitty-gritty details, let's see where previous approaches fall short.

## Where Existing Approaches Fall Short

## A Parameter Game

## One Year Later

## Resources

(Gupta et al. 2024)

- blog: https://bair.berkeley.edu/blog/2024/03/21/xt/
- open review: https://openreview.net/revisions?id=km5IHr5vFv
- Some ai-generated review: https://www.themoonlight.io/en/review/xt-nested-tokenization-for-larger-context-in-large-images
- project website: https://ai-climate.berkeley.edu/xt-website/
- github

Paper addresses the following problem:
- images needing to be handled by ai models are getting larger. E.g., compare the latest iPhone 16 Pro shooting images @ 45 MP that might be input to your vision-language model. On the other hand, handling large-scale images is typically sub-optimal. It either uses down-sampling, or cropping. Generally, we face a quadratic increase in memory usage as a function of image size.
- *comment:* Guess this is due to the fact that images are rectangular but also due to the fact that attention is quadratic and that a token sequence of a larger image becomes longer?
- *comment*: cropping is often done with global thumbnails from my experience.
- Both cropping and down-sampling have several down-sights. Context or information gets lost (cp. classical cropping/down-sampling.)
- The authors draw an analogy from soccer. For spectators it would not be enough to watch the game in low resolutions or see only the crops with the ball itself. Mare practically, to diagnose tiny patches of cancer every detail matters.
- *comment:* take care on your down-sampling strategy. Add some notes on pillows / common down-sampling algorithms.
- *comment:* We face similar challenges at work. Uploads get quickly shot with phone camera, but what matters most are details like punctuation, diacritics etc. which are much harder to see on gigantic images.
- They introduce $xT$, a framework to model large images end-to-end to contemporary GPUS while

## Core idea

- *analogy:* solve a giant jigsaw puzzle. Look at individual pieces smaller sections first, then figure out how they fit into a bigger picture. $xT$ chops large images into smaller pieces hierarchically and learn how they relate on a larger scale.
- Authors call it *nested tokenization*: Basically, an image is chopped into individual pieces, which map to tokens. The process is hierarchical/nested. That is, an image is split into regions, and each region can than be further subdivided into smaller regions, depending on the resolution of the vision backbone (which is called *region encoder* in the paper) before being patchified and processed by the region encoder.
- The nested/hierarchical approach allows to extract features at different scales on a local level.
- *comment:* Explain how this is different from:
  - what is done in classical vision towers
  - how window attention and swin relates. What ideas are similar, what are different. Probably that swin only sees cropped images.
- To encode both context and process regions the architecture features two components:
  - **region encoder:** convert independent regions into detailed representations i.e., extract image features. Might be a CNN or some transformer-based model like Swin
  - **context encoder:** take independent region encodings and stich them together so that insights from one token are considered int he context of other tokens. Can be *any* long sequence model. Authors use Transformer- and state-space-based models.
  - *Note:* Architectures were generally made for language, but authors demonstrate they are also suitable for this vision task (use in context encoders.)
- Combination of region and context encoders maintains the fidelty of original image's details while integrating long-distance context

![architectural overview xt](architectural_overview_xt.png)
The receptive field of vision backbones:

![receptive field](xt-receptive-field.png)

## Experiments

- classification (iNaturalist) of animals, context-dependent segmentation and object detection in context tasks
- In their experiment smaller models (fewer parameters) achieve higher accuracy on all downstream tasks, while consuming much less memory.
- Images as large as $29,000x25,000$ px can be processed on 40 GB A100s. Pretty impressive.

## Impact

In their blog they write:

> We're stepping into a new era where we don't have to compromise on the clarity or breadth of our vision. xT is our big leap towards models that can juggle the intricacies of large-scale images without breaking a sweat.

## References {#references .unnumbered}

<div id="refs" class="references csl-bib-body hanging-indent">

<div id="ref-guptaXTNestedTokenization2024" class="csl-entry">

Gupta, Ritwik, Shufan Li, Tyler Zhu, Jitendra Malik, Trevor Darrell, and Karttikeya Mangalam. 2024. *[xT]{.nocase}: Nested Tokenization for Larger Context in Large Images*. arXiv:2403.01915. arXiv. <https://doi.org/10.48550/arXiv.2403.01915>.

</div>

</div>
