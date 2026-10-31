import torch
from x_theta_modifier.local_search_utils import *
from sampling_rng import rand_by_sample

def gumbel_sample(x_theta, num_samples, temperature=1.0, sample_ids=None,
                  base_seed=0, step=0):
    """Sample from categorical distribution using Gumbel-Softmax trick with temperature.
    
    Args:
        x_theta: Probability distribution tensor
        num_samples: Number of samples to draw
        temperature: Temperature parameter (default: 1.0)
                    - temperature > 1: more uniform (softer) distribution
                    - temperature < 1: more peaked (sharper) distribution
                    - temperature = 1: original distribution
    """
    if sample_ids is None:
        random_values = torch.rand(
            (num_samples, *x_theta.shape), device=x_theta.device,
            dtype=x_theta.dtype)
    else:
        # rand_by_sample returns [B, N, L, V]; transpose to the existing
        # candidate-major layout [N, B, L, V].
        random_values = rand_by_sample(
            sample_ids, (num_samples, *x_theta.shape[1:]),
            device=x_theta.device, dtype=x_theta.dtype, base_seed=base_seed,
            step=step, stream=1).transpose(0, 1)
    gumbel_noise = -torch.log(-torch.log(random_values + 1e-10) + 1e-10)
    # Apply temperature: divide log probabilities by temperature
    all_samples = (x_theta.unsqueeze(0).log() / temperature + gumbel_noise).argmax(dim=-1)
    return all_samples

def keep_nonmask_values(all_samples, xt, mask_index):
    non_mask = (xt != mask_index).unsqueeze(0)  # Shape: [1, batch_size, length]
    all_samples = torch.where(non_mask, xt.unsqueeze(0).expand_as(all_samples), all_samples)
    return all_samples, non_mask

def compute_properties(smiles_list, property_calcs, distance_funcs):
    property_values = [
        calc(smiles_list, len(smiles_list), smiles_list[0].device)
        for calc in property_calcs
    ]
    distances = [
        distance_func(property_value)
        for distance_func, property_value in zip(distance_funcs, property_values)
    ]
    return distances

def compute_combined_distances(smiles_list, batch_size, num_x_theta_samples, property_calcs_parallel, distance_to_bounds_parallel, device):
    property_size = batch_size * num_x_theta_samples
    property_values = [
        calc(smiles_list, property_size, device)
        for calc in property_calcs_parallel
    ]
    # Apply distance_to_bounds_parallel to property_values
    distances = [
        distance_func(property_value)
        for distance_func, property_value in zip(distance_to_bounds_parallel, property_values)
    ]

    # Reshape each distance in distances to [batch_size, self.num_x_theta_samples]
    reshaped_distances = [
        distance.reshape(batch_size, num_x_theta_samples)
        for distance in distances
    ]
    return torch.stack(reshaped_distances, dim=-1)  # Shape: [batch_size, self.num_x_theta_samples, num_properties]


def find_best_tokens(all_samples, device, seq_len, tokenizer, batch_size, num_x_theta_samples_keepbest, property_calcs_parallel, distance_to_bounds_parallel):
    # For each batch element, we have num_x_theta_samples_keepbest molecules to compare
    # Goal: Find the best molecule for each batch element independently
    # "Best" means: lowest distance on property 0; if tied, lowest on property 1; etc. (hierarchical/lexicographical)
    
    # Reshape samples for score calculation: (batch_size * num_x_theta_samples_keepbest, seq_len)
    reshaped_samples = all_samples.transpose(0, 1).reshape(-1, seq_len)
    # Decode SMILES strings from token IDs
    smiles_list = tokenizer.batch_decode(reshaped_samples.cpu().numpy())
    
    num_properties = len(property_calcs_parallel)
    property_size = batch_size * num_x_theta_samples_keepbest
    
    # Lazy evaluation: compute properties on-demand
    all_distances = [None] * num_properties
    
    def get_distances_for_property(prop_idx):
        """Compute distances for a property if not already computed."""
        if all_distances[prop_idx] is None:
            # Compute for ALL samples at once (parallel) - this is correct and efficient
            prop_value = property_calcs_parallel[prop_idx](smiles_list, property_size, device)
            distance = distance_to_bounds_parallel[prop_idx](prop_value)
            # Reshape to [batch_size, num_samples]
            all_distances[prop_idx] = distance.reshape(batch_size, num_x_theta_samples_keepbest)
        return all_distances[prop_idx]
    
    # For each batch element, track which molecules are still candidates
    # candidates[b] = list of molecule indices still in contention for batch element b
    candidates = [list(range(num_x_theta_samples_keepbest)) for _ in range(batch_size)]
    
    # Process properties in hierarchical order (property 0 is most important)
    for prop_idx in range(num_properties):
        # Early termination: if all batch elements have found their unique best molecule
        if all(len(c) == 1 for c in candidates):
            break
        
        # Compute this property (lazy - only if at least one batch element needs it)
        distances = get_distances_for_property(prop_idx)
        
        # Filter candidates for each batch element independently
        for b in range(batch_size):
            if len(candidates[b]) == 1:
                # This batch element already has a unique best molecule
                continue
            
            # Among remaining candidates for this batch element, find minimum distance
            min_dist = min(distances[b, idx] for idx in candidates[b])
            
            # Keep only molecules with minimum distance (filter out worse ones)
            # Using == for comparison (handles inf correctly, we verified distances only have inf and numbers)
            candidates[b] = [idx for idx in candidates[b] if distances[b, idx] == min_dist]
    
    # Extract best molecule index for each batch element
    # If multiple molecules tied on all properties, take the first one
    best_indices = torch.tensor([c[0] for c in candidates], device=device)
    
    # Gather the best tokens for each batch element
    best_tokens = all_samples[best_indices, torch.arange(batch_size, device=device)]
    return best_tokens

def find_best_tokens_sort(all_samples, device, seq_len, tokenizer, batch_size, num_x_theta_samples_keepbest, property_calcs_parallel, distance_to_bounds_parallel):
    # Reshape samples for score calculation: (batch_size * self.num_x_theta_samples_keepbest, seq_len)
    reshaped_samples = all_samples.transpose(0, 1).reshape(-1, seq_len)
    # Decode SMILES strings from token IDs
    smiles_list = tokenizer.batch_decode(reshaped_samples.cpu().numpy())
    # Calculate property values for each function in self.property_calcs_parallel

    combined_distances = compute_combined_distances(smiles_list, batch_size, num_x_theta_samples_keepbest, property_calcs_parallel, distance_to_bounds_parallel, device)  # Shape: [batch_size, self.num_x_theta_samples_keepbest, num_properties]

    # combined_distances: [B, N, P] (batch_size, num_samples, num_properties)
    B, N, P = combined_distances.shape

    # Initialize indices for each sample in the batch
    sorted_indices = torch.arange(N, device=combined_distances.device).repeat(B, 1)

    # Iterate over columns from right to left for lexicographical ordering
    for col in reversed(range(P)):
        # Sort samples by current column with stable sort to preserve previous order
        _, perm = combined_distances[:, :, col].sort(dim=1, stable=True)
        
        # Update sorted indices according to the current permutation
        sorted_indices = sorted_indices.gather(1, perm)

    # Select the best sample (first in sorted order) for each batch element
    best_indices = sorted_indices[:, 0]  # Shape: [B]
    # Use advanced indexing to get the corresponding tokens
    best_tokens = all_samples[best_indices, torch.arange(B, device=device)]
    return best_tokens

def modify_x_theta_on_valid_best_tokens(best_tokens, tokenizer, x_theta, batch_size, device):
    decoded_best_tokens = tokenizer.batch_decode(best_tokens.cpu().numpy(), skip_special_tokens=True)
    validity_scores = calc_validity_parallel(decoded_best_tokens, batch_size, device)
    
    # Create a new tensor for x_theta with the same shape
    new_x_theta = torch.zeros_like(x_theta)

    # Create a mask for validity scores equal to 1
    validity_mask = validity_scores.unsqueeze(1).expand(-1, best_tokens.shape[1])  # Shape: [batch_size, seq_len]

    # Ensure the data type of new_x_theta matches validity_mask
    validity_mask = validity_mask.to(new_x_theta.dtype)

    # Use scatter_add to set weights in parallel
    new_x_theta.scatter_add_(
        2,
        best_tokens.unsqueeze(-1),
        validity_mask.unsqueeze(-1)
    )

    # If validity score is smaller than 0.5, copy the corresponding categorical distribution from x_theta
    invalid_mask = (validity_scores < 0.5).unsqueeze(1).unsqueeze(2).expand(-1, new_x_theta.shape[1], new_x_theta.shape[2])
    new_x_theta[invalid_mask] = x_theta[invalid_mask]
    return new_x_theta, validity_mask

def modify_x_theta_no_condition(best_tokens, tokenizer, x_theta, batch_size, device):
    # Create a new tensor for x_theta with the same shape
    new_x_theta = torch.zeros_like(x_theta)

    # Set the weight of the corresponding token to 1 for each categorical distribution
    new_x_theta.scatter_(2, best_tokens.unsqueeze(-1), 1.0)

    return new_x_theta, None
