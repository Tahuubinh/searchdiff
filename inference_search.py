#!/usr/bin/env python3
"""Generate sequences from pretrained MDLM or AR checkpoints."""

import torch
import os
import argparse
import lightning as L
from inference_utils import create_x_theta_config, create_posterior_sampling_config, generate_molecules_no_guidance
import time


def parse_arguments():
    """Parse command-line arguments for sequence generation."""
    parser = argparse.ArgumentParser(
        description="Generate sequences with optional x_theta modifications")
    parser.add_argument('--data', type=str, required=True,
                           choices=['qm9', 'grampa', 'trna'],
                           help='Type of dataset to use')
    parser.add_argument('--model_type', type=str, default='mdlm',
                       choices=['ar', 'mdlm'],
                       help='Generative model type (default: mdlm)')
    # Basic generation parameters
    parser.add_argument('--num_samples', type=int, default=1024,
                       help='Number of sequences to generate (default: 1024)')
    parser.add_argument('--batch_size', type=int, default=256,
                       help='Batch size for generation')
    parser.add_argument('--version', type=str, default='1',
                       help='Version label used in the output path (default: 1)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    
    # Sampling parameters
    parser.add_argument('--posterior_sampling_method', type=str, default='standard',
                       choices=['standard'])
    parser.add_argument('--top_k_categorical', type=int, default=None,
                       help='Restrict categorical sampling to the top K tokens')
    parser.add_argument("--property_type", type=str, nargs='+', default=["qed"], help="List of property types")
    parser.add_argument("--lower_bound", type=float, nargs='+', default=None, help="List of lower bounds for the property values")
    parser.add_argument("--upper_bound", type=float, nargs='+', default=None, help="List of upper bounds for the property values")
    parser.add_argument('--argmax_mode', type=str, default='none',
                       choices=['none', 'last_step', 'all_steps'],
                       help='Argmax mode stored in the sampling configuration; currently unused')
    parser.add_argument('--nucleus', type=float, default=None,
                       help='Nucleus sampling probability (default: None)')
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='Temperature for sampling (default: 1.0). Higher values make distribution more uniform, lower values make it more peaked.')

    # X-theta modification parameters
    parser.add_argument('--x_theta_type', type=str, default='standard',
                       choices=['standard', 'bon', 'bon_localsearch', 'bon_localsearch_laststep'],
                       help='Type of x_theta modification to apply')
    parser.add_argument('--num_x_theta_samples', type=int, default=32,
                       help='Number of samples for x_theta sampling method')
    parser.add_argument("--x_theta_max_num_local_search_iters", type=int, default=10000000, help="Reserved x_theta local-search iteration limit; currently unused")
    parser.add_argument("--x_theta_num_local_searches", type=int, default=30, help="Number of local searches to perform for x_theta")
    parser.add_argument("--max_candidate_tokens", type=int, default=1000, help="Number of candidate tokens for local search")
    
    return parser.parse_args()

def main():
    """Generate and save sequences using the requested sampling configuration."""
    
    # Parse command-line arguments
    args = parse_arguments()
    
    # Configure deterministic generation.
    L.seed_everything(args.seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    
    # Print all arguments
    print("Arguments:")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    data = args.data
    posterior_sampling_method = args.posterior_sampling_method
    property_type = args.property_type  
    lower_bound = args.lower_bound
    upper_bound = args.upper_bound
    top_k_categorical = args.top_k_categorical
    x_theta_type = args.x_theta_type
    nucleus = args.nucleus
    temperature = args.temperature
    max_candidate_tokens = args.max_candidate_tokens
    if data == 'qm9':
        checkpoint_path = f'outputs/QM9/{args.model_type}/best.ckpt'
        tokenizer_name_or_path = 'yairschiff/qm9-tokenizer'
        sequence_length = 32
        diffusion_steps = 32
    elif data == 'grampa':
        checkpoint_path = 'outputs/grampa/mdlm_no-guidance_padTrue/checkpoints/best.ckpt'
        tokenizer_name_or_path = 'grampa'
        sequence_length = 50
        diffusion_steps = 32
    elif data == 'trna':
        checkpoint_path = 'outputs/trna/mdlm_no-guidance_full/checkpoints/best.ckpt'
        tokenizer_name_or_path = 'trna'
        sequence_length = 110
        diffusion_steps = 32
    # Generate sequences without classifier guidance.
    print("Generating sequences without classifier guidance...")
    
    # Create the posterior and x_theta sampling configurations.
    posterior_sampling_config = create_posterior_sampling_config(args)
    x_theta_config = create_x_theta_config(args)
    start_time = time.time()
    molecules, raw_molecules = generate_molecules_no_guidance(
        data=data,
        sequence_length=sequence_length,
        diffusion_steps=diffusion_steps,
        num_samples=args.num_samples, 
        batch_size=args.batch_size,
        checkpoint_path=checkpoint_path,
        tokenizer_name_or_path=tokenizer_name_or_path,
        argmax_mode=args.argmax_mode,
        posterior_sampling_config=posterior_sampling_config,
        x_theta_config=x_theta_config,
        seed=args.seed,
        model_type=args.model_type,
    )
    end_time = time.time()
    
    if not top_k_categorical:
        top_k_categorical = 'all'
    if not nucleus:
        nucleus = 'all'
    # Determine the output folder.
    output_folder = f"sample_results/{data}/topk_{top_k_categorical}_nucleus_{nucleus}"

    folder_name_suffix = f"{posterior_sampling_method}/{x_theta_type}/{property_type}_lb{lower_bound}_ub{upper_bound}"
    if "localsearch" in x_theta_type:
        folder_name_suffix = f"{posterior_sampling_method}/{x_theta_type}_{args.x_theta_num_local_searches}_{max_candidate_tokens}/{property_type}_lb{lower_bound}_ub{upper_bound}"

    mol_folder = f"{output_folder}/{folder_name_suffix}/{args.version}/seed_{args.seed}"

    mol_file = f"{mol_folder}/molecules.txt"
    os.makedirs(os.path.dirname(mol_file), exist_ok=True)
    with open(mol_file, 'w') as mol_file:
        for mol in molecules:
            mol_file.write(f"{mol}\n")
    print(f"\nGenerated sequences saved to: {mol_file.name}")

    raw_mol_file = f"{mol_folder}/raw_molecules.txt"
    os.makedirs(os.path.dirname(raw_mol_file), exist_ok=True)
    with open(raw_mol_file, 'w') as raw_mol_file:
        for mol in raw_molecules:
            raw_mol_file.write(f"{mol}\n")
    print(f"\nGenerated raw sequences saved to: {raw_mol_file.name}")

    time_file = f"{mol_folder}/generation_time.txt"
    os.makedirs(os.path.dirname(time_file), exist_ok=True)
    with open(time_file, 'w') as time_file:
       time_file.write(f"Generation time: {end_time - start_time} seconds\n")

    print(f"\nGeneration time: {end_time - start_time} seconds")
    
    # Save all arguments to a file
    params_file = f"{mol_folder}/params.txt"
    os.makedirs(os.path.dirname(params_file), exist_ok=True)
    with open(params_file, 'w') as f:
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")

    print(f"\nParameters saved to: {params_file}")

if __name__ == "__main__":
    main()
