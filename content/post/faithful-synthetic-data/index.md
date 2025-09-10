---
title: My thoughts on How Faithful is Your Synthetic Data?🛸
date: 2025-09-07T07:52:00+02:00
description: My thoughts on the paper "How Faithful is Your Synthetic Data?".
Tags: [synthetic-data, data-sampling, visual-data, vision-language-models, paper]
Categories: [ai]
DisableComments: false
---

Training AI models on synthetic data is a data scientist's (and management's) dream come true. It's easy to generate in vast amounts, contains no labeling errors, and privacy concerns are virtually nonexistent. However, a frequently overlooked aspect is how to assess the quality of these synthetic samples. How can we build rich synthetic datasets that both mimic the properties of real data and introduce genuine novelty? These are challenges I frequently face in my daily work at [Atruvia](https://atruvia.de/).

A paper by Alaa et al. titled "How Faithful is Your Synthetic Data? Sample-Level Metrics for Evaluating and Auditing Generative Models"[^1] sheds light on these questions and sparked my interest. More specifically, it introduces a three-dimensional metric to assess the quality of generative models.

This new metric is both *domain-* and *model-agnostic*. Its novelty lies in being computable at the sample level (hurray 🎉), making it interesting for selecting high-quality samples for purely synthetic or hybrid datasets. Let's see if it holds up to scrutiny.

## What Makes a Good Synthetic Dataset?

Good synthetic data should fulfill the following three desirable qualities:

1. **Fidelity:** A high-fidelity synthetic dataset should contain only "realistic" samples—for instance, photorealistic images. So, no [sixth fingers](https://medium.com/@sanderink.ursina/why-do-ai-models-sometimes-produce-images-with-six-fingers-da4cd53f3313) or [pasta-eating nightmares](https://en.wikipedia.org/wiki/Will_Smith_Eating_Spaghetti_test) in your dataset.
1. **Diversity:** The synthetic dataset should capture the full variability of the real data, including rare edge cases.
1. **Generalization:** A synthetic dataset should contain truly novel samples, not just mere copies of the data the generative model was trained on. Without this third criterion, a strongly overfitted model could still score high on fidelity and diversity, but its outputs would just be copies, offering no real benefit.

All three aspects seem intuitive at first glance. As it happens, the authors propose a *three-dimensional* metric, $\mathcal{E}$, that maps nicely to these three qualities.🤓[^2] The mapping is as follows: $\alpha$-precision captures *fidelity*, $\beta$-recall captures *diversity*, and *authenticity* assesses *generalization*.

While $\alpha$-precision and $\beta$-recall are generalizations of the classic precision and recall metrics, the concept of *authenticity* is what's truly new here. This focus on generalization versus memorization is one of the paper's key contributions. Let's take a closer look at the metric $\mathcal{E}$.

$$
\mathcal{E} \triangleq(\underbrace{\alpha \text{-Precision}}_{\text {Fidelity }}, \underbrace{\beta \text{-Recall}}_{\text {Diversity }}, \underbrace{\text { Authenticity }}_{\text {Generalization }}) .
$$

## $\alpha$-Precision, $\beta$-Recall, and Authenticity at two levels of understanding

**level 1:**

As we know the metric is 3-dimensional. Informally and from a 10,000ft view, its dimensions are:

1. **$\alpha$-precision:** is the rate at which the generative model produces realistic looking examples.
1. **$\beta$-recall:** is the fraction of real samples, that are covered by the distribution of synthetic samples.
1. **authenticity:** is the rate at which the generative model produces truly new samples. Or put differently, 1 minus the rate of samples being copied form the training set with some random noise.

That was easy, right?

**level 2:**

Yet, comparing distributions incl. all data points isn't often desirable. The $\alpha$ and $\beta$ in alpha-precision and beta-recall indicates that we do not necessarily consider all data points within $\mathbb{P}_g$ or $\mathbb{P}_r$ but rather allow for some data points to be *outliers*. Think of $\alpha$ and $\beta$ being the knobs to control outlierness for synthetic and real samples.

Conceptually, the authors draw on minimum volume sets - sets that contain a specified probability mass with the smallest possible volume. We assume that a fraction $1 - \alpha$ for real samples and $1 - \beta$ for synthetic samples are outliers, while $\alpha$ and $\beta$ are typical. $\alpha$ and $\beta$ are varied between 0 and 1 to obtain full recall and precision curves. Thereby, we can also cover all possible definitions of what is considered as an outlier. [^3] Without this (setting $\alpha=\beta=1$), the approach would be prone to very rare samples in both the real and synthetic dataset.

Synthetic and real samples are both embedded into hydrospheres, which have the nice property that in this space, typical examples are located in the centre (modes) and outliers are pushed further to the boundary of the sphere. The hydrospheres have spherical-shaped supports, which depend on how we set $\alpha$ and $\beta$. If the radius of the hypersphere changes and so does our definition of an outlier. To summarize, a (synthetic or real) sample must lie in the $\alpha$ or $\beta$ support of its hypersphere to be considered typical. We dive more into how this setup can be used to our examples in the section on model debugging.

With our newly gained understanding of $\alpha$ and $\beta$ as a hyperparameter to determine the supports of the real and synthetic hypersphere, we are all set for a more precise definition of $\alpha$-precision and $\beta$-recall:

1. **$\alpha$-precision:** The probability that a synthetic sample lies within the $\alpha$-support of the real distribution. Intuitively, $\alpha$ has an impact on the creativity of the generative models. For small $\alpha$ s the generative model must produce samples closest to the most typical examples to lie within the support. For larger $\alpha$s or a less restrictive outlier definition it becomes more likely that a generated sample sneaks into the real hypersphere.
1. **$\beta$-recall:** The fraction of real samples that reside within the $\beta$-support of the synthetic distribution for a given $\beta$. Being able to vary $\beta$, we can control the diversity of samples we allow for.
1. **Authenticity** is a hypothesis test (TODO:XXXX) It tries to prevent *memorization*. Memorization means that the generative model covers regions in the support of the synthetic data distribution, despite that only few data points lie within this region. While conceptually similar to the more common overfitting, a overfitted model would fit the original distribution/histogram. [^5]

Let's next look at a practical example from the paper and count some kittens 🐈.


```
Third level of understanding

real distribution, i.e. $\mathbb{P}\left(\widetilde{X}_g \in \mathcal{S}_r\right)$ (Sajjadi et al., 2018). We propose a more refined measure of sample fidelity, called the $\alpha$-Precision metric ( $P_\alpha$ ), which we define as follows:

$$
P_\alpha \triangleq \mathbb{P}\left(\widetilde{X}_g \in \mathcal{S}_r^\alpha\right), \text { for } \alpha \in[0,1] .
$$


That is, $P_\alpha$ is the probability that a synthetic sample resides in the $\alpha$-support of the real distribution.
$\boldsymbol{\beta}$-Recall. To assess diversity in synthetic data, we propose the $\beta$-Recall metric as a generalization of the conventional Recall metric. Formally, we define the $\beta$-Recall as follows:

$$
R_\beta \triangleq \mathbb{P}\left(\widetilde{X}_r \in \mathcal{S}_g^\beta\right), \text { for } \beta \in[0,1]
$$

i.e., $R_\beta$ is the fraction of real samples that reside within the $\beta$-support of the generative distribution.

Generalization is independent of precision and recall since a model can achieve perfect fidelity and diversity without truly generating any samples, simply by resampling training data. Unlike discriminative models for which generalization is easily tested via held-out data, evaluating generalization in generative models is not straightforward (Adlam et al., 2019; Meehan et al., 2020). We propose an authenticity score $A \in [0,1]$ to quantify the rate by which a model generates new samples. To pin down a mathematical definition for $A$, we reformulate $\mathbb{P}_g$ as a mixture of densities as follows:

$$
\mathbb{P}_g=A \cdot \mathbb{P}_g^{\prime}+(1-A) \cdot \delta_{g, \epsilon}
$$

where $\mathbb{P}_g^{\prime}$ is the generative distribution conditioned on the synthetic samples not being copied, and $\delta_{g, \epsilon}$ is a noisy distribution over training data. In particular, we define $\delta_{g, \epsilon}$ as $\delta_{g, \epsilon}=\delta_g * \mathcal{N}\left(0, \epsilon^2\right)$, where $\delta_g$ is a discrete distribution
```


## A visual guide to $\alpha$-Precision, $\beta$-Recall, and Authenticity

![core-concept](core_concept.png)

The figure above depicts the proposed evaluation metric. The blue sphere corresponds to the $\alpha$-support of the real distribution. Likewise, the red sphere is the $\beta$-support for the generative distributions. For visualization sake, $\alpha=\beta=0.9$. The blue and red dots correspond to real and synthetic samples.

The assumption is now, that data falling outside of the blue sphere will look unrealistic or noisy (case a). Overfitted generative models, will produce high-quality data samples, that are unauthentic, because they are blunt copies from the training data (case b). High-quality samples should end up in the blue sphere/the $\alpha$-support.

Let's now calculate the metrics, for a fixed $\alpha$ and $\beta$. By counting kittens, we observe that out of 10 synthetic samples, 9 are typical cat images and 1 is an outlier. Out of 9, 8 also lie within the blue hypersphere. That gives us an $\alpha$-precision of $8/9$. Similarly, out of 9 typical synthetic samples, 4 are in the red sphere, $\beta$-recall is $4/9$. Of all synthetic samples generated, only one is unauthentic, which yields an authenticity of $9/10$.

```yaml
TODO: It's not clear to my why outliers in the own hypersphere are also excluded. This would mean we both depend on beta and alpha. From the formulas I'd think, that we take *all* synthetic / real samples.

Ah guess, because we have the outlier definition. From Section "Interpreting alpha-precision and beta-recall.
Interpreting $\boldsymbol{\alpha}$-Precision and $\boldsymbol{\beta}$-Recall. To interpret (4) and (5), we revisit the notion of $\alpha$-support. From (2), we know that an $\alpha$-support hosts the most densely packed probability mass $\alpha$ in a distribution, hence $\mathcal{S}_r^\alpha$ and $\mathcal{S}_g^\beta$ always concentrate around the modes of $\mathbb{P}_r$ and $\mathbb{P}_g$ (Figure 3); samples residing outside of $\mathcal{S}_r^\alpha$ and $\mathcal{S}_g^\beta$ can be thought of as outliers. In this sense, $P_\alpha$ and $R_\beta$ do not count outliers when assessing fidelity and diversity. That is, the $\alpha$-Precision score deems a synthetic sample to be of a high fidelity not only if it looks "realistic", but also if it looks "typical". Similarly, $\beta$-Recall counts a real sample as being covered by $\mathbb{P}_g$ only if it is not an outlier in $\mathbb{P}_g$. By sweeping the values of $\alpha$ and $\beta$ from 0 to 1 , we obtain a varying definition of which samples are typical and which are outliers-this gives us entire $P_\alpha$ and $R_\beta$ curves as illustrated in Figure 3.
```

### Debugging Failure Modes with the Metric

As we can distinguish *typical* samples and *outliers*, we can also use $P_{\alpha}$ and $R_{\beta}$ for debugging purposes, as we see next. 

![Interpretation of $P_{$\alpha$}$ and $R_{$\beta$}$ curves](model_debugging.png)

In the graphics above, the real distribution is colored in blue, and the generative distribution is in red.  $\mathbb{P}_r$ is a a multimodal distribution of cat images with two modes -- one for the tabby cat and another one for the Calico cat. The Carcal cat (left most cat) is an outlier for the specific $\alpha$.The shaded areas represent the probability mess covered by $\alpha$ and $\beta$ supports. By definition, the *support* concentrate around the modes. 

1. A *perfect* generative model would result in a $\alpha$-precision and $\beta$ -recall following the diagonal.
1. The model $\mathbb{P}_g$ exhibits *mode collapse*, as it fails to represent all modes (mode for Calico cat missing). We'd get a suboptimal, concave $\alpha$-precision curve, as more synthetic samples are in the $\alpha$-support than there should be. Because it does not cover all modes, the model will have a sub-optimal (below diagonal) $R_\beta$ curve. The same model would achieve perfect precision scores ($P_1$), but poor recall ($R_1$).
1. The model nails support for $\mathbb{P}_r$, and hence achieves a perfect recall/precision ($P_1=R_1=1$) as the entire distribution is covered by support. The generative model, however, invents a new mode for the Carcal cat/outlier, resulting in a poor $P_{\alpha}$ and $R_{\beta}$ as neither typical synthetic samples nor typical real samples are well covered in the other distribution. 
1. The last case is more subtle. The model realizes both types of cats but estimates a slightly shifted support and density. Intuitively, the model is best of all three models but will appear inferior to 2 under $P_1$ and $R_1$. This "improvement" is reflected in a improved $P_\alpha$ score and (still) suboptimal $R_\beta$ curve.

## Use in evaluation and auditing tasks

Let's next see how we can use $\alpha$-precision, $\beta$-recall, and authenticity to our advantage for  auditing the generative model and evaluating $\mathcal{E}$ on the embedded images.

![evaluation and auditing pipeline](auditing_evaluation_pipeline.png)

The first application (a) lies in *auditing* the generative model. By embedding the input features $X_r \sim \mathbb{P}_r$ and $X_g \sim \mathbb{P}_g$ into a feature space using an evaluation embedding function $\Phi$, we can evaluate $\mathcal{E}$ on the embedded features $\widetilde{X}_r=\Phi\left(X_r\right)$ and $\widetilde{X}_g=\Phi\left(X_g\right)$. Thereby, we can assess the quality of our our synthetic data for the desired qualities and ultimately assess how faithful we can be in our synthetic data.

For practitioners an even more interesting application is found in post-hoc model auditing (b) [^3]. As the metric can be estimated on the sample-level, for each sample $X_{g,j}$ in the synthetic dataset $\mathcal{D}_\text {synth}$, we can use the approach to reject samples with low authenticity and/or $\alpha$-precision scores and select (and re-generate) high-quality samples. The auditor thereby acts as a rejection sampler. One nitty detail: for auditing, we don't care about $\beta$-recall. I suspect this is due to the fact that ...

```yaml
TODO: I'm not quite sure, why they omit beta-recall for rejection sampling. It might have something to do, if we have have access to (all) real samples?
```

Until now it remains unclear, what approach we can use to generate the embeddings, how we construct the hyperspheres, how we measure proximity, and how the metrics themselves are calculated over the hyperspheres. Let's tackle this next.

## From Kittens to a Practical Implementation

3 binary classifiers

```yaml
TODO: https://github.com/ahmedmalaa/evaluating-generative-models/blob/main/representations/OneClass.py
TODO: https://www.analyticsvidhya.com/blog/2024/03/one-class-svm-for-anomaly-detection/
```

**$\alpha$-precision and $\beta$-recall:**

*pytorch loss function:*

```python
import torch


def soft_boundary_loss(emb: torch.Tensor, r: float, c: torch.Tensor, nu: float) -> float:
    """Soft-boundary loss.

    Args:
        emb (torch.Tensor): embedding
        r (float): radius
        c (torch.Tensor): centroid
        nu (float): weight term

    Returns:
        float: loss
    """
    dist = torch.sum((emb - c) ** 2, dim=1)
    scores = dist - r**2
    loss = r**2 + (1 / nu) * torch.mean(torch.max(torch.zeros_like(scores), scores))

    return loss
```

**Authenticity:**

- https://arxiv.org/pdf/1802.06360

- custom loss function,
- inspired by outlier detection

- resource on one-class classifiers https://www.analyticsvidhya.com/blog/2024/03/one-class-svm-for-anomaly-detection/

![](separating-hyperplane.png)
![](non-linear-mapping.png)

```yaml
    that places an unknown probability mass on each training data point in $\mathcal{D}_{\text {real }}, \epsilon$ is an arbitrarily small noise variance, and * is the convolution operator. Essentially, (7) assumes that the model flips a (biased coin), pulling off a training sample with probability $1-A$ and adding some noise to it, or innovating a new sample with probability $A$.
    4. Estimating the Evaluation Metric

    With all the metrics in Section 3 being defined on the sample level, we can obtain an estimate $\widehat{\mathcal{E}}=\left(\widehat{P}_\alpha, \widehat{R}_\beta, \widehat{A}\right)$ of the metric $\mathcal{E}$, for a given $\alpha$ and $\beta$, in a binary classification fashion, by assigning binary scores $\widehat{P}_{\alpha, j}, \widehat{A}_j \in\{0,1\}$ to each synthetic sample $\widetilde{X}_{g, j}$ in $\mathcal{D}_{\text {synth }}$, and $\widehat{R}_{\beta, i} \in\{0,1\}$ to each real sample $\widetilde{X}_{r, i}$ in $\mathcal{D}_{\text {real }}$, then averaging over all samples, i.e., $\widehat{P}_\alpha=\frac{1}{m} \sum_j \widehat{P}_{\alpha, j}, \widehat{R}_\beta=\frac{1}{n} \sum_i \widehat{R}_{\beta, i}, \widehat{A}=\frac{1}{m} \sum_j \widehat{A}_j$. To assign binary scores to individual samples, we construct three binary classifiers $f_P, f_R, f_A: \widetilde{\mathcal{X}} \rightarrow\{0,1\}$, where $\widehat{P}_{\alpha, j}=f_P\left(\widehat{X}_{g, j}\right), \widehat{R}_{\beta, i}=f_R\left(\widehat{X}_{r, i}\right)$ and $\widehat{A}_j=f_A\left(\widehat{X}_{g, j}\right)$. We explain the operation of each classifier in what follows.

    Precision and Recall classifiers ( $f_p$ and $f_R$ ). Based on definitions (4) and (5), both classifiers check if a sample resides in an $\alpha$ - (or $\beta$-) support, i.e., $f_P\left(\tilde{X}_g\right)=\mathbf{1}\left\{\tilde{X}_g \in \widehat{\mathcal{S}}_r^\alpha\right\}$ and $f_R\left(\widetilde{X}_r\right)=\mathbf{1}\left\{\widetilde{X}_r \in \widehat{\mathcal{S}}_g^\beta\right\}$. Hence, the main difficulty in
    implementing $f_P$ and $f_R$ is estimating the supports $\widehat{\mathcal{S}}_r^\alpha$ and $\widehat{\mathcal{S}}_g^\beta$-in fact, even if we know the exact distributions $\mathbb{P}_r$ and $\mathbb{P}_g$, computing their $\alpha$ - and $\beta$-supports is not straightforward as it involves solving the optimization problem in (2).

    To address this challenge, we pre-process the real and synthetic data in a way that renders estimation of $\alpha$-and $\beta$ supports straightforward. The idea is to train the evaluation embedding $\Phi$ so as to cast $\mathcal{S}_r$ into a hypersphere with radius $r$, and cast the distribution $\mathbb{P}_r$ into an isotropic density concentrated around the center $c_r$ of the hypersphere. We achieve this by modeling $\Phi$ as a one-class neural network trained with the following loss function: $L=\sum_i \ell_i$, where

    $$
    \ell_i=r^2+\frac{1}{\nu} \max \left\{0,\left\|\Phi\left(X_{r, i}\right)-c_r\right\|^2-r^2\right\} .
    $$


    The loss is minimized over the radius $r$ and the parameters of $\Phi$; the output dimensions of $\Phi, c_r$ and $\nu$ are viewed as hyperparameters (see Appendix). The loss in (8) is based on the seminal work on one-class SVMs in (Schölkopf et al., 2001), which is commonly applied to outlier detection problems, e.g., (Ruff et al., 2018). In a nutshell, the evaluation embedding squeezes real data into the minimum-volume hypersphere centered around $c_r$, hence $\mathcal{S}_r^\alpha$ is estimated as:

    $$
    \widehat{\mathcal{S}}_r^\alpha=\boldsymbol{B}\left(c_r, \widehat{r}_\alpha\right), \widehat{r}_\alpha=\widehat{Q}_\alpha\left\{\left\|\widetilde{X}_{r, i}-c_r\right\|: 1 \leq i \leq n\right\},
    $$

    -----

    ## 🌵 junk 

    $\alpha$-Precision, $\beta$-Recall and Authenticity
    3.1. Definitions and notations

    Let $\widetilde{X}_r=\Phi\left(X_r\right)$ and $\widetilde{X}_g=\Phi\left(X_g\right)$ be the embedded real and synthetic data. For simplicity, we will use $\mathbb{P}_r$ and $\mathbb{P}_g$ to refer to distributions over raw and embedded features interchangeably. Let $\mathcal{S}_r=\operatorname{supp}\left(\mathbb{P}_r\right)$ and $\mathcal{S}_g=\operatorname{supp}\left(\mathbb{P}_g\right)$, where $\operatorname{supp}(\mathbb{P})$ is the support of $\mathbb{P}$. Central to our proposed metrics is a more general notion for the support of $\mathbb{P}$, which we dub the $\alpha$-support. We define the $\alpha$-support as the minimum volume subset of $\mathcal{S}=\operatorname{supp}(\mathbb{P})$ that supports a probability mass of $\alpha$ (Polonik, 1997; Scott \& Nowak, 2006), i.e.,

    $$
    \mathcal{S}^\alpha \triangleq \min _{s \subseteq \mathcal{S}} V(s), \text { s.t. } \mathbb{P}(s)=\alpha
    $$

    where $V(s)$ is the volume (Lebesgue measure) of $s$, and $\alpha \in[0,1]$. One can think of an $\alpha$-support as dividing the full support of $\mathbb{P}$ into "normal" samples concentrated in $\mathcal{S}^\alpha$, and "outliers" residing in $\overline{\mathcal{S}}^\alpha$, where $\mathcal{S}=\mathcal{S}^\alpha \cup \overline{\mathcal{S}}^\alpha$.
    Finally, define $d\left(X, \mathcal{D}_{\text {real }}\right)$ as the distance between $X$ and the closest sample in the training data set $\mathcal{D}_{\text {real }}$, i.e.,

    $$
    d\left(X, \mathcal{D}_{r e a l}\right)=\min _{1 \leq i \leq n} d\left(X, X_{r, i}\right)
    $$

    where $d$ is a distance metric defined over the input space $\mathcal{X}$.

    We denote real and generated data as $X_r \sim \mathbb{P}_r$ and $X_g \sim \mathbb{P}_g$, respectively, where $X_r, X_g \in \mathcal{X}$, with $\mathbb{P}_r$ and $\mathbb{P}_g$ being the real and generative distributions. The real and synthetic data sets are $\mathcal{D}_{\text {real }}=\left\{X_{r, i}\right\}_{i=1}^n$ and $\mathcal{D}_{\text {synth }}=\left\{X_{g, j}\right\}_{j=1}^m$.
````

```yaml
TODO:
```

## Does It Scale?

Yes, probably? The used datasets from the paper range between 6k to 10k samples.

The computational demand is mostly affected by the embedding dimension $d_{\text{emb}}$; not so much by the input dimension, as both $k$-nearest neighbour for mode estimation and distance computation is done in the embedding space.



## Why Existing Metrics Fall Short

```yaml
TODO:
```

$\mathcal{N}\left(\boldsymbol{\mu}_{\mathrm{r}}, \boldsymbol{\Sigma}_{\mathrm{r}}\right)$


## Experiments

To validate their metrics, the authors designed four experiments covering model evaluation and auditing.

*Experiment 1:*

For the *evaluation setting*, they setup a small *ranking challenge*. The generate 4 synthetic, tabular datasets of COVID patient data using Generative Adversarial Networks (GANs), Variational Auto-Encoder (VAEs), and Wasserstein-GANs. Then, they fit a simple logistic regression model and evaluate the performance and derive a ground truth ranking.

Then they try to recover the ground truth ranking through estimating the similarity in terms of Fréchet inception distance (FID), Precision/Recall ($P_1$/$R_1$), Parzen window likelihood, density/coverage ($D$/$C$), as well as their own. In this experiment their integrated $\alpha$-precision and $\beta$-recall is among best to recover the true ranking and achieves among the highest $AUC-ROC$ scores on the real test set.

In a second sub-experiment, they demonstrate the performance of their approach as a criterion for finding a weighting-hyperparameter of a privacy-preserving loss function of ADS-GAN. 

Their third related sub-experiment is concerned about *model auditing*. Their results show that the ADS-GAN achieves a marginally larger AUC-ROC score on audited/pre-filtered samples.

```yaml

TODO:  maybe short? 
The first experiment tests if the metrics can correctly rank the quality of synthetic data. The authors generated four synthetic COVID-19 patient datasets and used them to train simple models. The real-world performance of these models established a "ground truth" ranking.

They found that their proposed metrics, α-precision (IP 
α
​
 ) and β-recall (IR 
β
​
 ), successfully reproduced this ground truth ranking, outperforming most standard metrics like FID and Precision/Recall. The metrics were also effective for hyperparameter tuning and model auditing in two smaller sub-experiments.
```

*Experiment 2:*

This experiment tackles mode dropping, a common failure where a generative model misses entire categories of data (e.g., a model trained on digits 0-9 fails to generate any '8's). Using a modified MNIST dataset, the authors showed that their $IR_{\beta}$ metric was significantly more sensitive to this problem than baseline approaches  like FID, Precision, and Recall.

*Experiment 3:*

Here, the authors re-evaluated models from a "Hide-and-Seek" challenge focused on generating private synthetic patient data. The original winner -- a simple model that just added noise to real data -- scored well on standard metrics but offered poor privacy.

The authors demonstrate that their authenticity metric, would have correctly flagged this model as low-quality, thereby exposing the privacy risk that other metrics missed.

*Experiment 4:*

In the last experiment they evaluate the performance of a StyleGAN and diffusion probabilistic models (DDPM) pre-trained on the CIFAR-10 dataset, generate 10,000 samples each, and compare against real samples  by $FID$ and $IP_{\alpha}$ and $IR_{\beta}$. 


## My Thoughts

The paper is a fresh and novel take on assessing the quality of synthetic data. I particularly like, there's finally a solution for sample-level evaluation and can be applied universally as long as we can input data into evaluation embeddings. 

My main concern is about practical applicability. The setup requires multiple parameters like $r$, $\nu$, $\Phi$ for the one-class classifiers, that require tuning and hyper-parameters like $k$ for Mini-Batch $k$-means for constructing the hyperspheres.

Ultimately, I remain sceptical about their experiments. The experiments demonstrate the applicability for various modalities (image, tabular etc.),  but the selection seem superficial, the models are date, and partly lacks quantitative evaluation e.g., their final experiment would have benefitted from an arena-like human eval compared to the metrics. A view that is shared by some reviewers on [openreview.net](https://openreview.net/forum?id=8qWazUd8Jm). What are your thoughts?  

## Useful links

![paper-summary](https://www.youtube.com/watch?v=zH1RVLHFr_M)

---


[^1]: see https://arxiv.org/abs/2102.08921

[^2]: Seems like nobody has coined a name for the metric yet. Feel free to propose

[^3]: Conceptually, this reminded me to [DBSCAN](https://en.wikipedia.org/wiki/DBSCAN) and its hyperparameter $\epsilon$.

[^4]: Post-hoc means here, that we leave our generative model as-is.

[^5]: Explanation adapted from here: https://youtu.be/_EEH9HU2EE0?feature=shared&t=2755