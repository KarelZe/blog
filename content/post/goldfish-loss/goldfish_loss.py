# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

"""Utility functions for training and inference."""
import inspect
import math
import pickle
import shutil
import sys
from dataclasses import asdict, is_dataclass
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, Mapping, Optional, TypeVar, Union, Literal, Tuple
from functools import partial

import lightning as L
import torch
import torch.nn as nn
import torch.utils._device
import yaml
from lightning.fabric.loggers import CSVLogger, TensorBoardLogger
from lightning.fabric.strategies import FSDPStrategy
from lightning.fabric.utilities.load import _lazy_load as lazy_load
from lightning.pytorch.loggers import WandbLogger
from torch.serialization import normalize_storage_type
from typing_extensions import Self

global hash_table
hash_table = None
table_size = 1_000_003


def _load_hash_table(device):
    global hash_table
    rng = torch.Generator(device=device)
    rng.manual_seed(2971215073)  # fib47 is prime
    hash_table = torch.rand(table_size, device=device, generator=rng)

def apply_goldfish(
    targets: torch.Tensor,
    strategy: str,
    k: int,
    goldfish_start_position: int,
    ignore_index,
    goldfish_context_width: int = 4,  # context width for hash based drops
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply a mask to a tensor to ignore every k-th token.
    `targets` is NOT updated in-place so apply_goldfish can be indepdently called for analysis/debugging/logging.

    Args:
        target: The target to apply the goldfish mask to.
        strategy: The strategy to use for goldfish.
            options implemented:
                - "static": Ignore every k-th token starting from `goldfish_start_position`.
                - "seeded_random": Ignore tokens with a probability of 1/k.
                - "hash-legacy": Ignore tokens based on a hash of the context. For debugging purposes only.
                - "hash-table": Ignore tokens based on a hash of the context using a precomputed table.
                - "hash-avalanche": Ignore tokens based on a hash of the context using a hash function.
        k: The frequency with which tokens are ignored?
        goldfish_start_position: The position to start ignoring tokens from.
        context_width: Context width for hash-based approaches.

    Returns:
        The target with the mask applied and the indices of the dropped tokens.
    """
    device = targets.device
    mbs, block_size = targets.shape
    masked_targets = targets.clone()

    if strategy == "static":
        dropped_token_indices = torch.arange(block_size, device=device)[goldfish_start_position::k].long()
        masked_targets[:, dropped_token_indices] = ignore_index
    elif strategy == "seeded_random":
        random_tensor = torch.randint(1, k + 1, size=targets.size())
        dropped_token_indices = (random_tensor == k).int() # probability of dropping a token is 1/k
        masked_targets[dropped_token_indices] = ignore_index
    elif strategy == "hash-legacy":
        # Old hash for sanity checks, do not use
        dropped_token_indices = torch.zeros_like(targets)
        rng = torch.Generator(device=device)
        for b in range(mbs):
            for s in range(goldfish_context_width, block_size):
                prf_key = targets[b, s - goldfish_context_width : s].prod()
                rng.manual_seed(prf_key.item() % (2**64 - 1))
                dropped_token_indices[b, s] = torch.rand((1,), device=device) < 1 / k
        masked_targets[dropped_token_indices] = ignore_index
    elif strategy == "hash-table":
        global hash_table
        if hash_table is None:
            _load_hash_table(device)
        hashed_keys = hash_table[targets.unfold(1, goldfish_context_width, 1).prod(dim=-1) % table_size]
        dropped_token_indices = (hashed_keys < 1 / k)
        masked_targets[:, goldfish_context_width-1:][dropped_token_indices] = ignore_index
        dropped_token_indices = dropped_token_indices.int()
    else:
        raise NotImplementedError(f"{strategy} goldfish strategy is not implemented. Try 'static' instead.")

    return masked_targets, dropped_token_indices

def dropped_token_loss(labels, loss, loss_all_token, tokenizer, cfg):
    goldfish_masked_targets, _ = apply_goldfish(
        targets=labels,
        strategy=cfg.goldfish_strategy,
        k=cfg.k_goldish,
        goldfish_start_position=cfg.goldfish_start_position,
        goldfish_context_width=cfg.goldfish_context_width,
        ignore_index=tokenizer.pad_id,
    )
    post_goldfish_token_count = (goldfish_masked_targets != tokenizer.pad_id if tokenizer else -1).sum().item()
    no_goldfish_token_count = (labels != tokenizer.pad_id if tokenizer else -1).sum().item()
    total_loss_difference = (loss_all_token * no_goldfish_token_count) - (loss * post_goldfish_token_count)
    dropped_token_loss = total_loss_difference / (no_goldfish_token_count - post_goldfish_token_count)
    return dropped_token_loss

@torch.compile  #
def hashint(key: torch.Tensor, width: int = 32):
    """
    For any 1<k<=64, let mask=(1<<k)-1. hash_64() is a bijection on [0,1<<k), which means
    hash_64(x, mask)==hash_64(y, mask) if and only if x==y. hash_64i() is the inversion of
    hash_64(): hash_64i(hash_64(x, mask), mask) == hash_64(hash_64i(x, mask), mask) == x.

    Source: http://burtleburtle.net/bob/hash/integer.html, runs in base python, caches based on access
    """
    mask = (1 << width) - 1
    key = (~key + (key << 21)) & mask
    key = (key << 21) - key - 1
    key = key ^ key >> 24
    key = ((key + (key << 3)) + (key << 8)) & mask
    key = key * 265
    key = key ^ key >> 14
    key = ((key + (key << 2)) + (key << 4)) & mask
    key = key * 21
    key = key ^ key >> 28
    key = (key + (key << 31)) & mask
    return key


def chunked_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    chunk_size: int = 128,
    ignore_indices: list[int] = [-1],
    label_smoothing: float = 0.0,
    k_goldish: Union[None, int] = 3,
    reduction: str = None,
    training=True,
    goldfish_strategy: Union[str, None] = None,
    goldfish_start_position: int = 0,
    goldfish_context_width: int = 4,
) -> torch.Tensor:
    # with large max_sequence_lengths, the beginning of `backward` allocates a large memory chunk which can dominate
    # the memory usage in fine-tuning settings with low number of parameters.
    # as a workaround hack, the cross entropy computation is chunked to force it to deallocate on the go, reducing
    # the memory spike's magnitude
    ignore_index = ignore_indices[0]
    for additional_ignore in ignore_indices[1:]:
        if additional_ignore is not None and additional_ignore != ignore_index:
            targets[targets == additional_ignore] = ignore_index

    # ignore every k-th token using ignore_index for goldfish
    if goldfish_strategy is not None and training:
        targets, _ = apply_goldfish(
            targets,
            goldfish_strategy,
            k_goldish,
            goldfish_start_position,
            ignore_index,
            goldfish_context_width
        )

    cross_entropy_fn = partial(
        torch.nn.functional.cross_entropy,
        ignore_index=ignore_index,
        label_smoothing=label_smoothing if training else 0.0,
    )

    # no chunking at all
    logits = logits.reshape(-1, logits.size(-1))
    targets = targets.reshape(-1)
    if chunk_size == 0:
        if reduction is not None:
            return cross_entropy_fn(input=logits, target=targets, reduction=reduction)
        else:
            return cross_entropy_fn(input=logits, target=targets)

    # chunk cross entropy
    logit_chunks = logits.split(chunk_size)
    target_chunks = targets.split(chunk_size)
    losses = torch.zeros_like(targets, dtype=logits.dtype, device=logits.device)  # prealloc required for compile

    for idx, (logit_chunk, target_chunk) in enumerate(zip(logit_chunks, target_chunks)):
        loss_chunk = cross_entropy_fn(input=logit_chunk, target=target_chunk, reduction="none")
        losses[idx * chunk_size : (idx + 1) * chunk_size] = loss_chunk

    non_masked_elems = (targets != ignore_index).sum().clamp(min=1.0)
    return losses.sum() / non_masked_elems
