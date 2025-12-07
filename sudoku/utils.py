"""
Utility functions for working with the Sudoku MDM.
"""

import torch
import numpy as np
from pathlib import Path
from sudoku import SudokuMDM, MDMSampler, check_sudoku_valid


def load_trained_model(checkpoint_path='checkpoints/sudoku_mdm_best.pt', device='cuda'):
    """
    Load a trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
        
    Returns:
        model: Loaded SudokuMDM
        sampler: MDMSampler instance
        checkpoint_info: Dictionary with training info
    """
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create model (assuming default config)
    model = SudokuMDM(
        vocab_size=10,
        d_model=256,
        nhead=8,
        num_layers=6,
        dim_feedforward=1024,
        dropout=0.1
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Create sampler
    sampler = MDMSampler(model, device=device)
    
    # Extract info
    info = {
        'epoch': checkpoint.get('epoch', 'unknown'),
        'train_loss': checkpoint.get('train_loss', 'unknown'),
        'val_loss': checkpoint.get('val_loss', 'unknown'),
    }
    
    return model, sampler, info


def generate_and_save_samples(sampler, num_samples=100, output_path='samples.npz'):
    """
    Generate samples and save to file.
    
    Args:
        sampler: MDMSampler instance
        num_samples: Number of samples to generate
        output_path: Path to save samples
    """
    print(f"Generating {num_samples} samples...")
    samples = sampler.sample(num_samples=num_samples, num_steps=20)
    
    # Evaluate
    samples_np = samples.cpu().numpy()
    num_valid = sum(check_sudoku_valid(s) for s in samples_np)
    validity_rate = num_valid / num_samples
    
    print(f"Validity rate: {validity_rate*100:.2f}%")
    
    # Save
    np.savez_compressed(
        output_path,
        samples=samples_np,
        num_valid=num_valid,
        validity_rate=validity_rate
    )
    print(f"Saved to {output_path}")
    
    return samples, validity_rate


def visualize_sudoku_ascii(board):
    """
    Print Sudoku board in ASCII art.
    
    Args:
        board: [81] tensor or numpy array
    """
    board = board.reshape(9, 9)
    if isinstance(board, torch.Tensor):
        board = board.cpu().numpy()
    
    print("┌───────┬───────┬───────┐")
    for i in range(9):
        if i > 0 and i % 3 == 0:
            print("├───────┼───────┼───────┤")
        
        row_str = "│"
        for j in range(9):
            val = board[i, j]
            if val == 0:
                row_str += " ."
            else:
                row_str += f" {int(val)}"
            
            if (j + 1) % 3 == 0:
                row_str += " │"
        
        print(row_str)
    
    print("└───────┴───────┴───────┘")


def batch_compute_mi(sampler, contexts, position_pairs, num_mc_samples=1000):
    """
    Compute MI for multiple position pairs across multiple contexts.
    
    Args:
        sampler: MDMSampler instance
        contexts: [batch_size, 81] tensor of puzzle configurations
        position_pairs: List of (i, j) tuples
        num_mc_samples: Number of MC samples per pair
        
    Returns:
        mi_matrix: [num_contexts, num_pairs] numpy array of MI estimates
    """
    num_contexts = len(contexts)
    num_pairs = len(position_pairs)
    
    mi_matrix = np.zeros((num_contexts, num_pairs))
    
    for ctx_idx, context in enumerate(contexts):
        print(f"Processing context {ctx_idx+1}/{num_contexts}...")
        
        for pair_idx, (pos_i, pos_j) in enumerate(position_pairs):
            # Sample from model
            samples = sampler.sample(
                num_samples=num_mc_samples,
                num_steps=20,
                given_puzzle=context.unsqueeze(0).repeat(num_mc_samples, 1)
            )
            
            # Extract values
            values_i = samples[:, pos_i].cpu().numpy()
            values_j = samples[:, pos_j].cpu().numpy()
            
            # Estimate joint
            joint = np.zeros((9, 9))
            for vi, vj in zip(values_i, values_j):
                if vi > 0 and vj > 0:
                    joint[vi-1, vj-1] += 1
            
            if joint.sum() > 0:
                joint = joint / joint.sum()
                
                # Compute MI
                marginal_i = joint.sum(axis=1)
                marginal_j = joint.sum(axis=0)
                
                mi = 0.0
                for i in range(9):
                    for j in range(9):
                        if joint[i, j] > 0:
                            mi += joint[i, j] * np.log(
                                joint[i, j] / (marginal_i[i] * marginal_j[j] + 1e-10)
                            )
                
                mi_matrix[ctx_idx, pair_idx] = mi
    
    return mi_matrix


def get_sudoku_constraints():
    """
    Get all constraint relationships in Sudoku.
    
    Returns:
        row_groups: List of 9 lists, each containing positions in that row
        col_groups: List of 9 lists, each containing positions in that column
        box_groups: List of 9 lists, each containing positions in that 3x3 box
    """
    # Rows
    row_groups = [[i*9 + j for j in range(9)] for i in range(9)]
    
    # Columns
    col_groups = [[i*9 + j for i in range(9)] for j in range(9)]
    
    # Boxes
    box_groups = []
    for box_i in range(3):
        for box_j in range(3):
            box = []
            for i in range(3):
                for j in range(3):
                    pos = (box_i*3 + i)*9 + (box_j*3 + j)
                    box.append(pos)
            box_groups.append(box)
    
    return row_groups, col_groups, box_groups


def get_constrained_pairs():
    """
    Get all pairs of positions that share a constraint.
    
    Returns:
        constrained_pairs: List of (i, j) tuples that share row/col/box
        unconstrained_pairs: Sample of pairs that don't share constraints
    """
    row_groups, col_groups, box_groups = get_sudoku_constraints()
    
    constrained_pairs = set()
    
    # Add all pairs within each constraint group
    for groups in [row_groups, col_groups, box_groups]:
        for group in groups:
            for i in range(len(group)):
                for j in range(i+1, len(group)):
                    pair = tuple(sorted([group[i], group[j]]))
                    constrained_pairs.add(pair)
    
    # Sample unconstrained pairs
    all_pairs = set()
    for i in range(81):
        for j in range(i+1, 81):
            all_pairs.add((i, j))
    
    unconstrained_pairs = list(all_pairs - constrained_pairs)
    
    return list(constrained_pairs), unconstrained_pairs


if __name__ == '__main__':
    # Demo
    print("Sudoku MDM Utilities")
    print("=" * 50)
    
    # Show constraint structure
    row_groups, col_groups, box_groups = get_sudoku_constraints()
    print(f"\nNumber of rows: {len(row_groups)}")
    print(f"Number of columns: {len(col_groups)}")
    print(f"Number of boxes: {len(box_groups)}")
    
    # Show constraint pairs
    constrained, unconstrained = get_constrained_pairs()
    print(f"\nConstrained pairs: {len(constrained)}")
    print(f"Unconstrained pairs: {len(unconstrained)}")
    
    # Example board
    print("\nExample empty board:")
    empty_board = torch.zeros(81)
    visualize_sudoku_ascii(empty_board)
