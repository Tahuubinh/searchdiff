"""Posterior sampling utilities for discrete diffusion models."""

class Sampler:
    """Base class for samplers with common initialization and abstract sampling method."""

    def __init__(self, top_k_categorical, nucleus, diffusion_model, temperature):
        """Initialize the sampler with common parameters.
        
        Args:
            top_k_categorical: Top-k filtering parameter
            nucleus: Nucleus sampling parameter
            diffusion_model: Diffusion model used for sampling
            temperature: Sampling temperature
        """
        self.top_k_categorical = top_k_categorical
        self.nucleus = nucleus
        self.diffusion_model = diffusion_model
        self.mask_index = diffusion_model.mask_index
        self.temperature = temperature

    def get_sampling_method(self):
        """Abstract method to be implemented by child classes."""
        raise NotImplementedError("Child classes must implement the get_sampling_method function.")
