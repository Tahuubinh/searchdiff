import random
import time  # Import time module for timing functionality
import math  # Import math module for infinity
from sampling_rng import python_rng
# Add RDKit SA_Score to path for sascorer import
from properties.peptides import *
from properties.molecules import *
from properties.trna import *
from properties.property_util import *

# Load the tokenizer
# tokenizer = AutoTokenizer.from_pretrained("yairschiff/qm9-tokenizer", trust_remote_code=True)

special_tokens = ['<bos>', '<eos>', '<mask>', '<pad>', '<unk>']

def local_search_qm9(tokens, distance_to_bounds, property_calcs, num_local_search_iters, max_num_local_search_iters, max_candidate_tokens, rng=random):
    # Initialize best molecule
    tokens = fix_basic_syntax_tokens(tokens[:])
    best_tokens = tokens[:]
    best_sequence = ''.join(token for token in best_tokens if token not in special_tokens)
    best_prop_values = [calc(best_sequence) for calc in property_calcs]
    best_distances = [dist(value) for dist, value in zip(distance_to_bounds, best_prop_values)]
    is_found = all(distance == 0 for distance in best_distances)
    operator = "Keep"
    has_updated = False  # Track if best_tokens has been updated

    # Shuffle indices to randomize traversal order
    indices = list(range(len(tokens)))
    rng.shuffle(indices)

    for i in indices:
        token = tokens[i]
        if token in atom_candidates:
            best_tokens, best_prop_values, best_distances, num_local_search_iters, operator, is_found, is_max_num_local_search_iters, has_updated = substitution_operator(best_tokens, best_prop_values, best_distances, tokens, i, token, num_local_search_iters, operator, atom_candidates_with_pad, property_calcs, distance_to_bounds, max_num_local_search_iters, max_candidate_tokens, 'atom substitution', has_updated, rng)
            if is_found or is_max_num_local_search_iters:
                return best_tokens, best_prop_values, num_local_search_iters, operator, is_found
                
        elif token in special_tokens and (i == 0 or tokens[i - 1] not in special_tokens):
        # elif token in special_tokens and tokens[i - 1] not in special_tokens:
            best_tokens, best_prop_values, best_distances, num_local_search_iters, operator, is_found, is_max_num_local_search_iters, has_updated = substitution_operator(best_tokens, best_prop_values, best_distances, tokens, i, token, num_local_search_iters, operator, atom_candidates, property_calcs, distance_to_bounds, max_num_local_search_iters, max_candidate_tokens, 'atom insertion', has_updated, rng)
            if is_found or is_max_num_local_search_iters:
                return best_tokens, best_prop_values, num_local_search_iters, operator, is_found

    return best_tokens, best_prop_values, num_local_search_iters, operator, is_found

def local_search_grampa(tokens, distance_to_bounds, property_calcs, num_local_search_iters, max_num_local_search_iters, max_candidate_tokens, rng=random):
    best_tokens = tokens[:]
    best_sequence = ''.join(token for token in best_tokens if token not in special_tokens)
    best_prop_values = [calc(best_sequence) for calc in property_calcs]
    best_distances = [dist(value) for dist, value in zip(distance_to_bounds, best_prop_values)]
    is_found = all(distance == 0 for distance in best_distances)
    operator = "Keep"
    has_updated = False  # Track if best_tokens has been updated

    # Shuffle indices to randomize traversal order
    indices = list(range(len(tokens)))
    rng.shuffle(indices)

    for i in indices:
        token = tokens[i]
        if token in aa_candidates:
            best_tokens, best_prop_values, best_distances, num_local_search_iters, operator, is_found, is_max_num_local_search_iters, has_updated = substitution_operator(best_tokens, best_prop_values, best_distances, tokens, i, token, num_local_search_iters, operator, aa_candidates_with_pad, property_calcs, distance_to_bounds, max_num_local_search_iters, max_candidate_tokens, 'amino acid substitution', has_updated, rng)
            if is_found or is_max_num_local_search_iters:
                return best_tokens, best_prop_values, num_local_search_iters, operator, is_found
                
        elif token in special_tokens and (i == 0 or tokens[i - 1] not in special_tokens):
        # elif token in special_tokens and tokens[i - 1] not in special_tokens:
            best_tokens, best_prop_values, best_distances, num_local_search_iters, operator, is_found, is_max_num_local_search_iters, has_updated = substitution_operator(best_tokens, best_prop_values, best_distances, tokens, i, token, num_local_search_iters, operator, aa_candidates, property_calcs, distance_to_bounds, max_num_local_search_iters, max_candidate_tokens, 'amino acid insertion', has_updated, rng)
            if is_found or is_max_num_local_search_iters:
                return best_tokens, best_prop_values, num_local_search_iters, operator, is_found


    return best_tokens, best_prop_values, num_local_search_iters, operator, is_found

def local_search_trna(tokens, distance_to_bounds, property_calcs, num_local_search_iters, max_num_local_search_iters, max_candidate_tokens, rng=random):
    best_tokens = tokens[:]
    best_sequence = ''.join(token for token in best_tokens if token not in special_tokens)
    best_prop_values = [calc(best_sequence) for calc in property_calcs]
    best_distances = [dist(value) for dist, value in zip(distance_to_bounds, best_prop_values)]
    is_found = all(distance == 0 for distance in best_distances)
    operator = "Keep"
    has_updated = False  # Track if best_tokens has been updated
    if is_found:
        return best_tokens, best_prop_values, num_local_search_iters, operator, is_found
    
    # Shuffle indices to randomize traversal order
    indices = list(range(len(tokens)))
    rng.shuffle(indices)

    for i in indices:
        token = tokens[i]
        if token in nucleotide_candidates:
            best_tokens, best_prop_values, best_distances, num_local_search_iters, operator, is_found, is_max_num_local_search_iters, has_updated = substitution_operator(best_tokens, best_prop_values, best_distances, tokens, i, token, num_local_search_iters, operator, nucleotide_candidates_with_pad, property_calcs, distance_to_bounds, max_num_local_search_iters, max_candidate_tokens, 'nucleotide substitution', has_updated, rng)
            if is_found or is_max_num_local_search_iters:
                return best_tokens, best_prop_values, num_local_search_iters, operator, is_found
                
        elif token in special_tokens and (i == 0 or tokens[i - 1] not in special_tokens):
        # elif token in special_tokens and tokens[i - 1] not in special_tokens:
            best_tokens, best_prop_values, best_distances, num_local_search_iters, operator, is_found, is_max_num_local_search_iters, has_updated = substitution_operator(best_tokens, best_prop_values, best_distances, tokens, i, token, num_local_search_iters, operator, nucleotide_candidates, property_calcs, distance_to_bounds, max_num_local_search_iters, max_candidate_tokens, 'nucleotide insertion', has_updated, rng)
            if is_found or is_max_num_local_search_iters:
                return best_tokens, best_prop_values, num_local_search_iters, operator, is_found


    return best_tokens, best_prop_values, num_local_search_iters, operator, is_found

def substitution_operator(best_tokens, best_prop_values, best_distances, tokens, i, token, num_local_search_iters, operator, candidates, property_calcs, distance_to_bounds, max_num_local_search_iters, max_candidate_tokens, operator_type, has_updated, rng=random):
    is_max_num_local_search_iters = False
    is_found = False
    num_candidate_tokens = 0
    num_local_search_iters += 1
    # Never mutate the module-level candidate lists shared by other samples.
    candidates = list(candidates)
    rng.shuffle(candidates)

    for replacement in candidates:
        if token == replacement: # Skip replacing a token with itself
            continue
        num_candidate_tokens += 1
        # Replace the token with the candidate
        new_tokens = tokens[:]
        new_tokens[i] = replacement
        new_sequence = ''.join(token for token in new_tokens if token not in special_tokens)
        
        # Compute properties and distances lazily with early termination
        new_prop_values = []
        new_distances = []
        is_better = False
        is_worse = False
        
        for prop_idx, (calc, dist_func) in enumerate(zip(property_calcs, distance_to_bounds)):
            # Compute property and distance for this index
            prop_value = calc(new_sequence)
            distance = dist_func(prop_value)
            new_prop_values.append(prop_value)
            new_distances.append(distance)
            
            # Compare with best distance at this index
            if distance < best_distances[prop_idx]:
                is_better = True
                break  # New is better, no need to compute remaining properties
            elif distance > best_distances[prop_idx]:
                is_worse = True
                break  # New is worse, no need to compute remaining properties
            # If equal, continue to next property
        
        # If we found a better solution, OR if best hasn't been updated and new is not worse
        accept_new = is_better or (not has_updated and not is_worse)
        
        if accept_new:
            # Compute remaining properties for completeness
            for prop_idx in range(len(new_prop_values), len(property_calcs)):
                prop_value = property_calcs[prop_idx](new_sequence)
                distance = distance_to_bounds[prop_idx](prop_value)
                new_prop_values.append(prop_value)
                new_distances.append(distance)
            
            operator = operator_type
            best_tokens = new_tokens
            best_prop_values = new_prop_values
            best_distances = new_distances
            has_updated = True  # Mark as updated
            if all(distance == 0 for distance in best_distances):
                is_found = True
                return best_tokens, best_prop_values, best_distances, num_local_search_iters, operator, is_found, is_max_num_local_search_iters, has_updated
        
        # Break if maximum iterations are reached
        if num_local_search_iters >= max_num_local_search_iters:
            is_max_num_local_search_iters = True
            return best_tokens, best_prop_values, best_distances, num_local_search_iters, operator, is_found, is_max_num_local_search_iters, has_updated
        if num_candidate_tokens >= max_candidate_tokens:
            break # Only try `max_candidate_tokens` replacements per position

    return best_tokens, best_prop_values, best_distances, num_local_search_iters, operator, is_found, is_max_num_local_search_iters, has_updated

def local_search_on_best_tokens(best_tokens, tokenizer, num_local_searches, distance_to_bounds, property_calcs, local_search, device, max_candidate_tokens, sample_ids=None, base_seed=0, step=0):
    # Iterate through each best_token in best_tokens
    modified_tokens = []
    for best_token_index, best_token in enumerate(best_tokens):
        rng = (random if sample_ids is None else python_rng(
            base_seed, sample_ids[best_token_index], step, 3))
        # Move best_token to CPU for RDKit and tokenizer operations
        best_token_cpu = best_token.cpu()
        # shape = best_token_cpu.shape
        # best_token_cpu = torch.arange(1, shape[0]+1, dtype=best_token_cpu.dtype, device=best_token_cpu.device)

        # Convert token IDs to tokens directly to preserve length
        sequence_string = tokenizer.decode(best_token_cpu, skip_special_tokens=False)
        modified_token = tokenizer.tokenize(sequence_string)
        # modified_token = tokenizer.convert_ids_to_tokens(best_token_cpu.tolist())

        max_num_local_search_iters = 1e9
        index_local_search = 0
        num_local_search_iters = 0
        is_found = False

        while index_local_search < num_local_searches and num_local_search_iters < max_num_local_search_iters and not is_found:
            new_best_tokens, best_prop_values, num_local_search_iters, operator, is_found = local_search(
                tokens=modified_token,
                distance_to_bounds=distance_to_bounds,
                property_calcs=property_calcs,
                num_local_search_iters=num_local_search_iters,
                max_num_local_search_iters=max_num_local_search_iters,
                max_candidate_tokens=max_candidate_tokens,
                rng=rng
            )
            index_local_search += 1
            if new_best_tokens == modified_token:
                break
            else:
                modified_token = new_best_tokens[:]

        modified_token_ids = tokenizer.convert_tokens_to_ids(modified_token)
        # --- Append to the list of modified tokens ---
        modified_tokens.append(modified_token_ids)

    # Stack into a tensor with proper device
    best_tokens = torch.tensor(modified_tokens, dtype=torch.long, device=device)
    return best_tokens

if __name__ == "__main__":
    smiles = "C1=CC=CC=)C1O"
    import ipdb; ipdb.set_trace()
