import torch

from sampling_rng import rand_by_sample


def test_rand_by_sample_is_invariant_to_batch_partition():
    sample_ids = list(range(11))
    full = rand_by_sample(
        sample_ids, (3, 4), device=torch.device("cpu"),
        dtype=torch.float32, base_seed=17, step=5, stream=2)

    partitioned = torch.cat([
        rand_by_sample(
            sample_ids[start:end], (3, 4), device=torch.device("cpu"),
            dtype=torch.float32, base_seed=17, step=5, stream=2)
        for start, end in ((0, 4), (4, 7), (7, 11))
    ])

    assert torch.equal(full, partitioned)


def test_rand_by_sample_is_distinct_across_samples_and_streams():
    kwargs = {
        "sample_shape": (16,),
        "device": torch.device("cpu"),
        "dtype": torch.float32,
        "base_seed": 23,
        "step": 4,
    }
    draws = rand_by_sample([8, 9], stream=1, **kwargs)
    other_stream = rand_by_sample([8], stream=2, **kwargs)

    assert not torch.equal(draws[0], draws[1])
    assert not torch.equal(draws[0], other_stream[0])
