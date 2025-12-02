---
title: My thoughts on Be like a Goldfish, Don’t Memorize!🐠
date: 2025-12-02T07:52:00+02:00
description: My thoughts on the paper "Be like a Goldfish, Don’t Memorize!".
Tags: [causal-language-modelling, llm, memorization, paper]
Categories: [ai]
DisableComments: false
# thumbnail: images/thumbnail_cat_hypersphere.png
# images:
#   - images/thumbnail_cat_hypersphere.png
---

# Goals
- mitigate memorization by modifying the pretraining objective for next token prediction.
- Memorization is linked to both privacy and copyright risks.
	- For copyright risks cp. recent lawsuit by Helene Fischer vs. Openai.
	- For copyright risks of providers cp. recent lawsuit against meta, who trained on Annas Archive.
	- For privacy concerns i.e., leakage of personal identifiable information see emails etc. and real email footers in early chatpgt / or privacy book.
- *Note:* my own experience. European regulators stress measures against overfitting and memorization.
- *Note:* Explain how memorization and overfitting is different.


##  🐠 Contribution
- Introduce the *goldfish loss* (GL). During training a randomly sampled subset of tokens are excluded from the loss computation. Should prevent reproduction of complete token sequences in pre-training corpus.

> We present the goldfish loss, a strikingly simple technique that leverages properties of the next-token prediction objective to mitigate verbatim generation of memorized training data.

## Steps
- 1. Forward pass on all tokens in a batch.
	- NOTE: Interestingly the forward pass is calculated on the entire unaltered text sample to achieve high efficiency. Guess no masking/skipping of tokens whatsoever. (see Sec. 2.3)
- 2. next-token prediction loss is then calculated on a pseudo-random subset of (e.g., 25 %) of training tokens. $1/k$ of tokens are dropped, with $k$ being a hyper parameter.
	- NOTE: *pseudo* or *pseudo-random token* is important here. Here, pseudo doesnt refer to pseudo-number generators (which are practically all random number generators used in modern computers).  *pseudo-random token mask* means instead that a passage is always masked in the same manner. This prevents peeking for duplicates in the dataset or if a model is trained for several epochs I guess. (see also their motivation for hashed masks Sec. 3.1. They describe at the example of syndicated news articles)
	- NOTE: For true random number generators, link to lava lamp video by Tom Scott. Lava lamps generate truly random patterns at cloudflare, which are used in random number generators.
	- For construction of the goldfish mask multiple variants are possible:
		1. static mask: drop every $k$ th token in a sequence.
			- NOTE: Wondering if sequences are padded and not somewhat continues, so that leakage could happen again?
		2. *hashed mask:* used in goldfish loss. Its determinstic (:tada:)
		3. (only baseline) random mask, randomly drop tokens with probability ($1/k$)
- *Note:* research how $k$ should be set. In the paper they use $k=4$. Generally, its $k$-GL.
- 3. backward pass
	- Only information for the backward pass is masked.
	- NOTE: Explain how this is different from other regularization techniques, such as dropout. Would with dropout also be the forward pass be masked?

![[goldfish-loss.png]]
## Rationale
At inference time model must make an unsupervised guess each time it tries to predict a dropped token, making it diverge from training data/impeding verbatim reproductions.

## Definition

## Goldfish loss and comparison to cross entropy loss


LLMs are commonly trained using a causal language modeling (CLM) objective that represents the average log-probability of a token, conditioned on all previous tokens. For a sequence $x=\left\{x_i\right\}$ of $L$ training tokens, this is written as:

$$
\mathcal{L}(\theta)=-\frac{1}{L} \sum_{i=1}^L \log P\left(x_i \mid x_{<i} ; \theta\right)
$$


This objective is minimized when the model correctly predicts the sequence $\left\{x_i\right\}$ with high confidence.

- NOTE: Guess this the cross-entropy loss. See the following graphs for inspiration of visualizations.
- NOTE: see blog post. https://lena-voita.github.io/nlp_course/language_modeling.html
![[cross-entropy-loss-voita.png]]

![[topk-sampling-language-modellign.png]]
- NOTE: how sentences are completed. See this blogpost https://thegradient.pub/understanding-evaluation-metrics-for-language-models/ ![[viz-probability-sentence-modelling.png]]
-

For this reason, models trained by next token prediction can be prone to memorization. However, successful regeneration of a token $x_j$ at test time depends on the correct conditioning of the complete preceding sequence $x_{<j}$ being provided as input.
The goldfish loss is only computed on a subset of the tokens, and thus prevents the model from learning the entire token sequence. For a choosen a goldfish mask $G \in\{0,1\}^L$ and goldfish loss is defined as:

$$
\mathcal{L}_{\text {goldfish }}(\theta)=-\frac{1}{|G|} \sum_{i=1}^L G_i\left(x_i\right) \log P\left(x_i \mid x_{<i} ; \theta\right) .
$$


In plain English, we ignore the loss on the $i$ th token if its mask value is $G_i=0$, and include the token if $G_i=1$. Most importantly, the outputs $x_i$ are still conditioned on all prior tokens $x_{<i}$, allowing the model to learn the full distribution of natural language over the course of training. Yet, for a given passage, the model does not learn to predict the $i$ th token, and so is never conditioned on the exact sequence $x_{<i}$ at test time. Note that the goldfish mask will be chosen independently for each training sample, based on local context using a hash mask (described in detail in Section 3.1).

- NOTE: Need to rewrite the text for my own blogpost.
## Hashed mask for goldfish loss

Goal is to mask texts uniformly that might have different headers, attributions etc. They propose a localized *hashed mask*. The idea is equally simple and based around a hash function that compacts a text (sequence of tokens) to a real value $f:|V| \rightarrow \mathbb{R}$. For, the goldfish loss only the $h$ preceding tokens, the so called context width, is incorporated in the the calculation of hash value, resulting in a more consistent masking. The authors mask token $x_i$ if and only if the outputs of a hash function $f:|V|^h \rightarrow \mathbb{R}$ applied to the $h$ preceding tokens is less than $\frac{1}{k}$.
- NOTE: I'm wondering how minor differences in text affect hashing e.g. punctation like semicolon vs. periods or em-dashes vs plain dashes, linespaces, encodings?. Might require a sold preprocessing. They discuss under limitations that "We also remark that our technique is potentially vulnerable to leakage under near-duplicated (but different) text segments that get masked differently, especially if a proper hash based implementation is not used."
- NOTE: add some simple python pseudo code for masking.
- NOTE: there is a trade off between large and small $h$. If $h$ is too small the model may fail to memorize important n grams like institutions. Large values for $h$, e.g., $h=13$ are even used to check contamination of train/test sets.
- NOTE: figure out how n-grams relate to tokens. Are $n$-grams here equal to $n$ tokens?
- NOTE: reminds me to loses used in pretraining objectives of tabular transformers e.g., FT Transformer.
- NOTE: reminds me to monte carlo dropout to avoid overfitting (and also memorization)?
	- The authors reference papers that draw a link between memorization and overfitting. They discuss weight decay and dropout.
## Experiments

- They measure by 'extractable memorization' (see Sec. 2.1 +4). Simply, if the model model is prompted with a prefix of length $p$ and it memorizes the string, it will complete the remaining string as in the training corpus.
	- NOTE: see Carlini et. al. to better understand how it is defined. Add some python pseudo code. What acts as a boundary for the sentence? Punctuation?
	- It gives them percentage of correctly predicted sequences compared to ground truth.
	- NOTE: probably averaged.
- They measure by RougeL score. The score quantifies the length of the longest common
(non-consecutive) subsequence. NOTE: we need to check for the longest ordered set of tokens that appear in both sequences (not necessarily consecutively).
- For visualization see: https://community.deeplearning.ai/t/rouge-l-calculation-in-the-lecture-model-evaluation-of-week-2/423507
![[rouge-l.png]]
- They compare two setups:
	- extreme setup with many epochs that promotes memorization
	- standard training setup
- They construct a test set of training sequences by chopping them into fixed size prefix (for sequence completion) and the suffix of length $n$.

 ## Extreme setup
 - pretraining LLaMA-2-7B model for 100 epochs on 100 English Wikpedia articles.
 - temperature=0.
	 - Results:
		 - verbatim memorization of 84/100 articles
		 - with goldfish loss with $k=4$ none.
		 - RougeL metrics indicate that the model trained with goldfish loss repeats non-consecutive n-gram sub-sequences that are roughly twice as long as a model that never saw the data.  NOTE: I don't understand what is meant by that.
## standard training setup
- pretrain TinyLLaMA-1.1B on a vocab of 32k.
- they very $k$ (NOTE: can be interpred as aggressivness of masking) and benchmark against the standard causal language modeling loss.
- single epoch, RedPajama datadataset + token sequences from wikipedia of token length 1024 and 2048.
- To mimic the case of quasi-duplicates they duplicate sequences within the target set 50 times.
- NOTE: Personally, I would have preferred if they didn't duplicate but rather slightly augment the sequences, which would have led to partly failures of the hashing function.
- Result is the same:
	- goldfish loss hinders models ability to produce target sequences.
	- For lower values of $k$ the extractable memorization is closer to the control model trained using the standard language model objective. This is expected as fewer tokens are masked.
- NOTE: research setup, sampling, temperature etc.
![[goldfish-loss-standard-loss.png]]

## Divergence Position vs. Drop Positions

- tokens are not memorized when they are dropped by the goldfish loss. From the extreme example (experiment with 50 epochs).
- Observation the majority of index positions where the token was masked coincides with the position where the sequences start to diverge. This is both true for the static mask (every nth token) and the hashed mask (hash-based indexing).
- NOTE: I wonder if there are negative implications for facts that we wish to learn. e.g., founding date of the united states etc? -> This is answered under Section "Can LLMs Swallow the Goldfish Loss? Testing Impacts on Model Performance".

## possible degradations in language modelling and reasoning

- To check if the goldfish loss impacts evaluation benchmark performance, they evaluate the model trained with goldfish loss against a model with standard CLM objective on across an array of popular tasks from the Hugging Face Open LLM Leaderboard
- They find now systematic differences between the goldfish loss model and standard CLM model besides the BoolQ benchmark.
## impact on language modelling
- goldfish models are partically trained on fewer tokens (as a fraction of $1/k$ is masked). To investigate the effect they track validation losses and measure semantic coherence.
	- validation losses:
		- validation loss in terms of total number of supervised tokens (aka unmasked tokens).
		- For 12M total tokens of RedpajamaV2 data, the validation loss decreases slower for the goldfish model than the standard CLM model.
		- If however only unmasked tokens/supervised tokens equals the number of input tokens in the standard loss, the models end up with an approximately equal validation loss or use larger batches.
		- NOTE: implications for training will require more (or more capable) compute depending on the $k$ to achieve the same validation loss.
		- Some simple math for estimating *supervised tokens*:
		> Since the net number of supervised tokens is fewer with goldfish loss than with standard loss, we plot the number of supervised tokens (i.e., the tokens used in the loss calculation) against the validation loss of RedPajamaV2. For all models, we train with 20 billion supervised tokens. This corresponds to 20 billion input tokens for the standard loss and 26.7 billion input tokens for the goldfish loss. The calculation is based on the formula: $\left(1-\frac{1}{k}\right) \times$ Input Tokens $=$ Supervised Tokens, where $k=4$.
![[supervised-vs-input-tokens.png]]
- they also estimate Mauve scores to see if the generated text is both fluent and related to the original test.  The Mauve metric used to evaluate the quality of generated text against real text by measuring similarity in terms of diversity and naturalness. This metric also noted to be highly correlated with human text.
- On the Slimpajama dataset, models with a goldfish loss and small $k$ the output seems less natural and diverse with greedy sampling (aka temperature=0, sampling only the token with highest probablity), leading to inferior Mauve scores. For larger temperatures (temperature=0.7) and larger values of $k$ the goldfish model anneals the performance of standard CLM models. As $k$ becomes larger, the performance the of the CLM model, as fewer and fewer tokens are masked.
## Adverserial Extraction Methods

- they perform membership attacks in the loss and $zlib$ criertia on dataset of training and hold-out Wikipedia sequences.
- NOTE: I didn't read the reamining section, as it is not important for my work.

## Adaptive Attack

- do a beam search with large number of beams (e.g., $k=30$) to find different candidates for missing tokens to find a sequence with low perplexity.
- Expectedly for small values of $k$ the goldfish model withstands the attack, but for larger $k$s the extractability expectedly increases.
- NOTE: I feel like this scenario is somewhat artificial but rather risky, as the attacker would need access to prefix sequences, the beam search (and implementation). In a practical scenario how might have access to the raw data as well, if the remaining infrastructure has been intruded already?

## Findings
- We train a 7B parameter model on a small number of articles for 100 epochs, finding that the models trained with goldfish loss resist memorization while standard training memorizes most of the training data (see Figure 1).
- We then turn to more standard training regimen, where we observe that the memorization metrics of goldfish models closely resemble models that never saw the training data at all (Section 4.2).
- Improved privacy (aka lower accuracy) in membership attacks. In an experiment they used an aggressive beam search decoder. Note: read what is about.

## Downsides/Limitations
- According to the authors models can still learn effectively on training data, but require to train for longer. (yes some remarks above about supervised tokens)
- We also remark that our technique is potentially vulnerable to leakage under near-duplicated (but different) text segments that get masked differently, especially if a proper hash based implementation is not used (see Section 6.3).

## When to use
- interesting for documents from specfic high risk sources. Useful for late phases of the training curriculum.
- NOTE: I quite like that it only requires minimal changes to training setup, that is useful for practicioners with limited resources that might have to run seeveral experiemnts.
>  We hope that goldfish loss paves the way for aiding copyright compliance rather than serving as a means to misuse private data maliciously.
-  NOTE: I'm skeptical about their conclusion about copyright compliance. Couldn't goldfish loss become a vehicle of disguising that copyrighted content was used? Reference latest writers strikes in us.
- Goldfish loss is particularly interesting if model is trained over multiple epochs, as the mask remains the same. NOTE: I feel like this idea is universally applicable to other use cases as well.
- NOTE: It would have been interesting to see how the standard CLM model with different temperatures would have compared against the goldfish models.

## Citation

```latex
% NOTE: add bibtex citation for my own blogpost. Author is me Markus Bilz, year is 2025, blog is https://blog.markusbilz.com
```

## interesting resources
- evaluation metrics for language modeling by Chip Huyen https://thegradient.pub/understanding-evaluation-metrics-for-language-models/
- writers strikes https://www.latimes.com/business/technology/story/2023-09-25/column-sag-aftra-strike-writers-victory-humans-over-ai
- course on language modelling https://lena-voita.github.io/nlp_course/language_modeling.html
