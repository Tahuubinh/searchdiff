"""Deterministic random helpers for batch-size-invariant generation."""

import random

import torch


def sample_seed(base_seed, sample_id, step, stream):
    """Return a stable seed keyed by a sample rather than its batch position."""
    # Keep the result in the range accepted by both Python and torch generators.
    value = int(base_seed) & ((1 << 63) - 1)
    for component in (sample_id, step, stream):
        value ^= int(component) + 0x9E3779B97F4A7C15 + (value << 6) + (value >> 2)
        value &= (1 << 63) - 1
    return value


def python_rng(base_seed, sample_id, step, stream):
    return random.Random(sample_seed(base_seed, sample_id, step, stream))


def rand_by_sample(sample_ids, sample_shape, *, device, dtype, base_seed,
                   step, stream):
    """Draw one tensor per global sample ID, independently of batch shape."""
    draws = []
    for sample_id in sample_ids:
        generator = torch.Generator(device=device)
        generator.manual_seed(sample_seed(base_seed, sample_id, step, stream))
        draws.append(torch.rand(
            sample_shape, device=device, dtype=dtype, generator=generator))
    return torch.stack(draws, dim=0)
