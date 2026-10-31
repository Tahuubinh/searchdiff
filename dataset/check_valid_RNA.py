import RNA

def check_structural_stability(sequence: str, mfe_threshold: float = -20.0) -> (bool, float, str):
    """
    Checks if an RNA sequence is thermodynamically stable (Valid).
    The threshold is based on general stability for functional ncRNAs.
    
    Args:
        sequence: The generated RNA sequence.
        mfe_threshold: The threshold for minimum stability (must be less than this value).
                       Value -20.0 kcal/mol is a general safe lower bound for functional RNA.
                       
    Returns:
        (is_stable, mfe_value, structure)
    """
    seq = sequence.upper()
    
    # 1. Syntax Check (Must be A, C, G, U)
    if any(c not in 'ACGU' for c in seq):
        return False, 999.0, "Invalid Syntax"
        
    # 2. Physics Check (MFE Calculation)
    try:
        # RNA.fold is the core physical oracle
        (structure, mfe) = RNA.fold(seq)
        
        # Rule: Stability is directly related to MFE (Lower/More negative = More Stable)
        is_stable = mfe < mfe_threshold
        
        return is_stable, mfe, structure
    except Exception:
        # Catch errors if ViennaRNA fails on an unexpected character or sequence length
        return False, 999.0, "Tool Error"


# --- Examples ---
# 1. 5S rRNA example
valid_5s = "UUUGCAGUUCCAAAGUGCUCACAGUGCAGGUAGUACACACUGAUCUACUGCAGUUUGUGCACUUCAGGUGCUGCUGG"
# 2. Random RNA example
random_seq = "AUGUCAAAGUGCUAACAGUGCAGGUAGCUUUUUGAGUUCUACUGCAGUGCCAGCACUUCUUACAU"

print("--- STABILITY CHECK ---")
stable, mfe_val, struct = check_structural_stability(valid_5s)
print(f"5S rRNA: Valid={stable} | MFE={mfe_val:.2f} kcal/mol (Target: < -20.0)")

stable_rand, mfe_rand, struct_rand = check_structural_stability(random_seq)
print(f"Random : Valid={stable_rand} | MFE={mfe_rand:.2f} kcal/mol")
