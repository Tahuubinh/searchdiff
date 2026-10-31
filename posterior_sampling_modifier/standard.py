import torch
from .base import Sampler
from .shared_ops import *
from sampling_rng import rand_by_sample

class GumbelSampler(Sampler):
    """Gumbel sampler for sampling from categorical distributions."""
    
    def __init__(self, *args, **kwargs):
        """Initialize the Gumbel sampler."""
        super().__init__(*args, **kwargs)

    def get_sampling_method(self):
        """Return the sampling method function."""
        def _sample_categorical(categorical_probs, xt, step, sample_ids=None):
            """Sample from categorical distribution using Gumbel trick.
            
            Args:
                categorical_probs: Probability distribution tensor of shape (..., vocab_size)
                
            Returns:
                Sampled indices of shape (...,)
            """

            if self.nucleus:
                categorical_probs = nucleus_sampling_filter(categorical_probs, self.nucleus)

            if self.top_k_categorical:
                categorical_probs = filter_top_k_with_mask(categorical_probs, xt, self.mask_index, self.top_k_categorical)
            
            if sample_ids is None:
                random_values = torch.rand_like(categorical_probs)
            else:
                random_values = rand_by_sample(
                    sample_ids, categorical_probs.shape[1:],
                    device=categorical_probs.device,
                    dtype=categorical_probs.dtype,
                    base_seed=self.diffusion_model.config.seed, step=step,
                    stream=2)
            gumbel_norm = (
                1e-10 - (random_values + 1e-10).log()).to(
                    categorical_probs.dtype)
            # Temperature: divide log probs by temperature (higher temp = more uniform)
            return (categorical_probs.log() / self.temperature + gumbel_norm).argmax(dim=-1)
        
        return _sample_categorical
