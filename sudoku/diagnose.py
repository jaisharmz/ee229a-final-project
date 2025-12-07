"""
Diagnostic script to check data and training for NaN issues.
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import sys

sys.path.append('.')
from sudoku import SudokuDataset, SudokuMDM, MaskSchedule

def diagnose_data():
    """Check dataset for potential NaN sources."""
    print("=" * 60)
    print("DIAGNOSING DATA")
    print("=" * 60)
    
    # Load or generate small dataset
    data_dir = Path('data')
    train_path = data_dir / 'train_sudoku.npz'
    
    if train_path.exists():
        print(f"Loading dataset from {train_path}...")
        dataset = SudokuDataset(data_path=str(train_path))
    else:
        print("Generating small test dataset...")
        dataset = SudokuDataset(num_samples=100, difficulty='mixed')
    
    print(f"Dataset size: {len(dataset)}")
    
    # Check samples
    print("\nChecking first 5 samples:")
    for i in range(min(5, len(dataset))):
        puzzle, solution = dataset[i]
        
        print(f"\nSample {i}:")
        print(f"  Puzzle: min={puzzle.min()}, max={puzzle.max()}, unique={len(torch.unique(puzzle))}")
        print(f"  Solution: min={solution.min()}, max={solution.max()}, unique={len(torch.unique(solution))}")
        
        # Check for 0s in solution
        if (solution == 0).any():
            print(f"  ⚠️  WARNING: Solution contains 0s! Count: {(solution == 0).sum().item()}")
        
        # Check if solution is valid
        if not all(torch.unique(solution) <= 9) or not all(torch.unique(solution) >= 1):
            print(f"  ⚠️  WARNING: Solution values out of range [1-9]")
    
    print("\n" + "=" * 60)
    print("DATA CHECK COMPLETE")
    print("=" * 60)


def diagnose_masking():
    """Check masking strategy."""
    print("\n" + "=" * 60)
    print("DIAGNOSING MASKING")
    print("=" * 60)
    
    schedule = MaskSchedule('linear', 10)
    
    # Create dummy solution
    solution = torch.randint(1, 10, (1, 81)).long()  # Valid 1-9
    
    print(f"Original solution: min={solution.min()}, max={solution.max()}")
    
    for t in [0, 5, 10]:
        mask_ratio = schedule.get_mask_ratio(t)
        masked, mask = schedule.apply_mask(solution.clone(), mask_ratio)
        
        print(f"\nStep {t}: mask_ratio={mask_ratio:.2f}")
        print(f"  Masked positions: {mask.sum().item()} / {mask.numel()}")
        print(f"  Masked values: min={masked[mask].min()}, max={masked[mask].max()}")
        print(f"  Unmasked values: min={masked[~mask].min()}, max={masked[~mask].max()}")
        
        if (masked[mask] != 0).any():
            print(f"  ⚠️  WARNING: Masked positions don't all have 0!")


def diagnose_loss():
    """Check loss computation."""
    print("\n" + "=" * 60)
    print("DIAGNOSING LOSS COMPUTATION")
    print("=" * 60)
    
    vocab_size = 10
    batch_size = 4
    seq_len = 81
    
    # Create dummy data
    logits = torch.randn(batch_size, seq_len, vocab_size)  # [B, L, V]
    solution = torch.randint(1, 10, (batch_size, seq_len)).long()  # Valid 1-9
    
    # Create masking schedule and mask
    schedule = MaskSchedule('linear', 10)
    t = 5
    mask_ratio = schedule.get_mask_ratio(t)
    masked_solution, mask = schedule.apply_mask(solution.clone(), mask_ratio)
    
    print(f"Logits: shape={logits.shape}, min={logits.min():.4f}, max={logits.max():.4f}")
    print(f"Solution: shape={solution.shape}, min={solution.min()}, max={solution.max()}")
    print(f"Masked positions: {mask.sum().item()} / {mask.numel()}")
    
    # Try loss computation
    try:
        # Get logits and targets for masked positions
        logits_masked = logits[mask].view(-1, vocab_size)  # [num_masked, V]
        targets_masked = solution[mask].view(-1)  # [num_masked]
        
        print(f"\nLogits for masked: shape={logits_masked.shape}")
        print(f"Targets for masked: shape={targets_masked.shape}, min={targets_masked.min()}, max={targets_masked.max()}")
        
        # Check if targets are in valid range
        if (targets_masked < 0).any() or (targets_masked >= vocab_size).any():
            print(f"⚠️  WARNING: Target values out of vocab range [0, {vocab_size})")
            print(f"   Min: {targets_masked.min()}, Max: {targets_masked.max()}")
        
        # Compute loss
        loss = F.cross_entropy(logits_masked, targets_masked, reduction='mean')
        
        print(f"\nLoss: {loss.item():.6f}")
        
        if torch.isnan(loss):
            print("⚠️  LOSS IS NAN!")
        else:
            print("✓ Loss is valid")
    
    except Exception as e:
        print(f"⚠️  ERROR during loss computation: {e}")
    
    print("\n" + "=" * 60)
    print("LOSS CHECK COMPLETE")
    print("=" * 60)


def diagnose_model():
    """Check model output."""
    print("\n" + "=" * 60)
    print("DIAGNOSING MODEL OUTPUT")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create model
    model = SudokuMDM(
        vocab_size=10,
        d_model=256,
        nhead=8,
        num_layers=2,  # Smaller for testing
        dim_feedforward=512,
        dropout=0.1
    ).to(device)
    
    print(f"Model created on {device}")
    
    # Create dummy input
    x = torch.randint(0, 10, (2, 81)).to(device)  # [B, L]
    
    print(f"Input: shape={x.shape}, min={x.min()}, max={x.max()}")
    
    try:
        with torch.no_grad():
            logits = model(x)
        
        print(f"Output logits: shape={logits.shape}, min={logits.min():.4f}, max={logits.max():.4f}")
        
        # Check for NaN/Inf
        if torch.isnan(logits).any():
            print("⚠️  Logits contain NaN!")
        if torch.isinf(logits).any():
            print("⚠️  Logits contain Inf!")
        
        # Check marginals
        marginals = torch.softmax(logits, dim=-1)
        print(f"Marginals: shape={marginals.shape}, min={marginals.min():.6f}, max={marginals.max():.6f}")
        print(f"Marginals sum: {marginals.sum(dim=-1).mean():.6f} (should be ~1.0)")
        
        if torch.isnan(marginals).any():
            print("⚠️  Marginals contain NaN!")
        
        print("\n✓ Model output looks valid")
    
    except Exception as e:
        print(f"⚠️  ERROR during model forward pass: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("MODEL CHECK COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    diagnose_data()
    diagnose_masking()
    diagnose_loss()
    diagnose_model()
    
    print("\n" + "=" * 60)
    print("ALL DIAGNOSTICS COMPLETE")
    print("=" * 60)
    print("\nIf you see any ⚠️  warnings, that's the likely source of NaN!")
