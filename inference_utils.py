import torch
from omegaconf import OmegaConf
import diffusion_search
import dataloader

def setup_config_no_guidance(data, sequence_length, diffusion_steps, checkpoint_path, tokenizer_name_or_path, argmax_mode='none', posterior_sampling_config=None, x_theta_config=None, seed=42, model_type='mdlm'):
    """Build an inference configuration for a local checkpoint.
    
    Args:
        data: Dataset or sequence domain.
        sequence_length: Maximum generated sequence length.
        diffusion_steps: Number of diffusion sampling steps.
        checkpoint_path: Path to the local checkpoint file
        tokenizer_name_or_path: Tokenizer name or local path.
        argmax_mode: Argmax mode stored in the sampling configuration.
        posterior_sampling_config: Posterior sampling configuration.
        x_theta_config: x_theta modification configuration.
        seed: Random seed for reproducibility.
        model_type: Generative model type, either ``mdlm`` or ``ar``.
    """
    if model_type == 'ar':
        backbone = 'dit'
        parameterization = 'ar'
        time_conditioning = False
        zero_recon_loss = False
        model_config = {
            'name': 'small',
            'type': 'ddit',
            'hidden_size': 768,
            'cond_dim': 128,
            'length': sequence_length,
            'n_blocks': 12,
            'n_heads': 12,
            'scale_by_sigma': True,
            'dropout': 0.1,
            'tie_word_embeddings': False,
        }
    else:
        backbone = 'dit'
        parameterization = 'subs'
        time_conditioning = True
        zero_recon_loss = True
        model_config = {
            'name': 'small',
            'type': 'ddit',
            'hidden_size': 768,
            'cond_dim': 128,
            'length': sequence_length,
            'n_blocks': 12,
            'n_heads': 12,
            'scale_by_sigma': True,
            'dropout': 0.1,
            'tie_word_embeddings': False,
        }

    # Create a complete inference config matching the selected checkpoint.
    config = OmegaConf.create({
        # Model and backbone settings
        'mode': 'train',
        'backbone': backbone,
        'classifier_backbone': None,
        'diffusion': 'absorbing_state',
        'parameterization': parameterization,
        'time_conditioning': time_conditioning,
        'subs_masking': False,
        'zero_recon_loss': zero_recon_loss,
        'T': 0,
        'is_vision': False,
        'seed': seed,
        
        'model': model_config,
        
        # Dataset and tokenizer configuration
        'data': {
            'train': data,
            'valid': data,
            'tokenizer_name_or_path': tokenizer_name_or_path,
            'cache_dir': '/tmp/huggingface_cache',
            'wrap': False,
            'streaming': False,
            'override_cache': False,
            'add_special_tokens': True,
            'label_col': None,
            'label_col_pctile': None,
            'num_classes': None
        },
        
        # Noise schedule
        'noise': {
            'type': 'loglinear'
        },
        
        # Guidance (None for no guidance)
        'guidance': None,
        
        
        # Sampling config
        'sampling': {
            'use_cache': False,
            'steps': diffusion_steps,
            'batch_size': 16,  # Overridden before generation
            'num_sample_batches': 64,  # Overridden before generation
            'use_float64': False,
            'argmax_mode': argmax_mode,
            # Posterior and x_theta sampling configuration
            'posterior_sampling_method': posterior_sampling_config.get('method'),
            'top_k_categorical': posterior_sampling_config.get('top_k_categorical'),
            'property_type': posterior_sampling_config.get('property_type'),
            'nucleus': posterior_sampling_config.get('nucleus'),
            'lower_bound': posterior_sampling_config.get('lower_bound'),
            'upper_bound': posterior_sampling_config.get('upper_bound'),
            'x_theta_modifier_method': x_theta_config.get('method'),
            'num_x_theta_samples': x_theta_config.get('num_x_theta_samples'),
            'x_theta_num_local_searches': x_theta_config.get('num_local_searches'),
            'max_candidate_tokens': x_theta_config.get('max_candidate_tokens'),
            'temperature': posterior_sampling_config.get('temperature', 1.0),
        },
        
        # Training config (needed even for inference)
        'training': {
            'ema': 0.9999,
            'antithetic_sampling': True,
            'importance_sampling': False,
            'sampling_eps': 1e-3,
            'change_of_variables': False,
            'compute_loss_on_pad_tokens': False,
            'use_simple_ce_loss': False,
            'guidance': None
        },
        
        # Loader config
        'loader': {
            'global_batch_size': 512,  # Unused during direct inference
            'eval_global_batch_size': 512,  # Unused during direct inference
            'batch_size': 16,  # Overridden before generation
            'eval_batch_size': 16,  # Overridden before generation
            'num_workers': 4,
            'pin_memory': True,
            'persistent_workers': True
        },
        
        # Checkpoint and generation settings
        'eval': {
            'checkpoint_path': checkpoint_path,
            'disable_ema': False,
            'generate_samples': True,
            'generated_samples_path': '',
            'max_samples': 50000
        },
        
        # Optimizer config required by Diffusion initialization
        'optim': {
            'weight_decay': 0,
            'lr': 3e-4,
            'beta1': 0.9,
            'beta2': 0.999,
            'eps': 1e-8
        },
        
        # LR scheduler config required by Diffusion initialization
        'lr_scheduler': {
            '_target_': 'constant_warmup',
            'warmup_steps': 1000,
            'constant_lr': 3e-4
        },
        
        # Trainer config (minimal for inference)
        'trainer': {
            'accelerator': 'gpu' if torch.cuda.is_available() else 'cpu',
            'devices': 1 if torch.cuda.is_available() else 'auto',
            'precision': 'bf16' if torch.cuda.is_available() else '32'
        }
    })
    return config

def create_x_theta_config(args):
    """Create x_theta configuration from parsed arguments."""
    return {
            'method': args.x_theta_type,
            'num_x_theta_samples': args.num_x_theta_samples,
            'num_local_searches': args.x_theta_num_local_searches,
            'top_k_categorical': args.top_k_categorical,
            'max_num_local_search_iters': args.x_theta_max_num_local_search_iters,
            'property_type': args.property_type,
            'lower_bound': args.lower_bound,
            'upper_bound': args.upper_bound,
            'max_candidate_tokens': args.max_candidate_tokens,
            'temperature': args.temperature
        }

def create_posterior_sampling_config(args):
    """Create the standard posterior sampling configuration."""
    return {
            'method': args.posterior_sampling_method,
            'top_k_categorical': args.top_k_categorical,
            'nucleus': args.nucleus,
            'property_type': args.property_type,
            'lower_bound': args.lower_bound,
            'upper_bound': args.upper_bound,
            'temperature': args.temperature
        }

def generate_molecules_no_guidance(data, sequence_length, diffusion_steps, num_samples=100, batch_size=16, checkpoint_path=None, tokenizer_name_or_path=None, argmax_mode='none', posterior_sampling_config=None, x_theta_config=None, seed=42, model_type='mdlm'):
    """Generate sequences from a local MDLM or AR checkpoint without guidance.
    
    Args:
        data: Dataset or sequence domain.
        sequence_length: Maximum generated sequence length.
        diffusion_steps: Number of diffusion sampling steps.
        num_samples: Number of sequences to generate.
        batch_size: Batch size for generation.
        checkpoint_path: Path to the local checkpoint file.
        tokenizer_name_or_path: Tokenizer name or local path.
        argmax_mode: Argmax mode stored in the sampling configuration.
        posterior_sampling_config: Posterior sampling configuration.
        x_theta_config: x_theta modification configuration.
        seed: Random seed for reproducibility.
        model_type: Generative model type, either ``mdlm`` or ``ar``.

    Returns:
        A pair containing cleaned sequences and their original decoded forms.
    """
    
    # Select the best available device.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Loading model from checkpoint: {checkpoint_path}")
    print(f"Using batch size: {batch_size}")
    
    # Build the inference configuration.
    config = setup_config_no_guidance(
        data, sequence_length, diffusion_steps, checkpoint_path,
        tokenizer_name_or_path, argmax_mode, posterior_sampling_config,
        x_theta_config, seed, model_type)
    config.sampling.batch_size = min(batch_size, num_samples)  # Use provided batch_size
    config.sampling.num_sample_batches = (num_samples + config.sampling.batch_size - 1) // config.sampling.batch_size
    
    # Also update loader config to match
    config.loader.batch_size = config.sampling.batch_size
    config.loader.eval_batch_size = config.sampling.batch_size
    
    # Print posterior sampling info
    if posterior_sampling_config:
        method = posterior_sampling_config.get('method')
        print(f"Using posterior sampling method: {method}")
    else:
        print("Using standard posterior sampling")
    
    # Load tokenizer
    tokenizer = dataloader.get_tokenizer(config)
    
    # Load model from checkpoint
    model = diffusion_search.Diffusion.load_from_checkpoint(
        checkpoint_path,
        tokenizer=tokenizer,
        config=config, 
        logger=False,
        weights_only=False
    )
    
    # Move the model to the selected device.
    model = model.to(device)
    model.eval()
    
    all_samples = []
    
    for i in range(config.sampling.num_sample_batches):
        print(f"Generating batch {i+1}/{config.sampling.num_sample_batches}")
        sample_start = len(all_samples)

        # Generate one batch.
        with torch.no_grad():
            sample_ids = list(range(
                sample_start,
                sample_start + config.sampling.batch_size))
            samples = model.sample(sample_ids=sample_ids)
            
        # Decode samples
        decoded_samples = tokenizer.batch_decode(samples)
        all_samples.extend(decoded_samples)
        
        if len(all_samples) >= num_samples:
            break
    
    # Remove tokenizer special tokens from decoded sequences.
    cleaned_samples = []
    for sample in all_samples[:num_samples]:
        cleaned = sample.replace('<bos>', '').replace('<eos>', '').replace('<pad>', '').replace('<mask>', '').replace('<unk>', '').replace('<cls>', '').replace('<sep>', '').replace('<reserved>', '').strip()
        
        if cleaned:
            cleaned_samples.append(cleaned)
    
    # Return both cleaned and original decoded sequences.
    return cleaned_samples, all_samples[:num_samples]
