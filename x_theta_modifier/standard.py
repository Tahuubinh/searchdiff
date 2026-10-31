from x_theta_modifier.local_search_utils import *
from .base import XThetaModifier

class StandardXThetaModifier(XThetaModifier):
    
    def __init__(self, *args, **kwargs):
        """Initialize the Gumbel sampler."""
        super().__init__(*args, **kwargs)

    def get_x_theta_method(self):
        def _no_modification(x_theta, xt, step, best_clean_samples,
                             sample_ids=None):
            return x_theta, xt
        
        return _no_modification
