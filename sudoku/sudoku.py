"""
Masked Diffusion Model for Sudoku

This implements a discrete diffusion model that learns to generate valid Sudoku solutions
by iteratively unmasking positions. The model outputs per-position marginal distributions
over the vocabulary {0,1,2,...,9} where 0 represents MASK.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple, List
import math
from tqdm import tqdm
import requests
import zipfile
import os


# ============================================================================
# Sudoku Data Generation & Loading
# ============================================================================

class SudokuDataset(Dataset):
    """Dataset of Sudoku puzzles with solutions.

    Supports loading from an NPZ file (puzzles/solutions arrays) or from a CSV
    with columns 'quizzes' and 'solutions' where each entry is an 81-character
    digit string.
    """

    def __init__(self, data_path: Optional[str] = None, num_samples: int = 10000,
                 difficulty: str = "mixed", is_csv: bool = False, start_idx: int = 0):
        """Initialize the dataset.

        Args:
            data_path: Optional path to dataset (NPZ or CSV).
            num_samples: Number of samples to load (or generate if no data_path).
            difficulty: Used when generating synthetic puzzles.
            is_csv: If True, treat `data_path` as CSV even if extension differs.
            start_idx: Row offset into CSV (useful to split train/val/test).
        """
        self.puzzles = []
        self.solutions = []

        if data_path and os.path.exists(data_path):
            # Prefer CSV loader when requested or when filename ends with .csv
            if is_csv or data_path.lower().endswith('.csv'):
                self._load_from_csv(data_path, num_samples=num_samples, start_idx=start_idx)
            else:
                self._load_from_file(data_path)
        else:
            print(f"Generating {num_samples} Sudoku puzzles...")
            self._generate_data(num_samples, difficulty)

    def _load_from_file(self, path: str):
        """Load puzzles from an NPZ file containing 'puzzles' and 'solutions'."""
        data = np.load(path)
        self.puzzles = torch.from_numpy(data['puzzles']).long()
        self.solutions = torch.from_numpy(data['solutions']).long()
        print(f"Loaded {len(self.puzzles)} puzzles from {path}")

    def _load_from_csv(self, path: str, num_samples: Optional[int] = None, start_idx: int = 0):
        """Load puzzles/solutions from a CSV file.

        The CSV is expected to contain at least two columns: the first (or
        'quizzes') is the puzzle string and the second (or 'solutions') is the
        solution string. Each string should be 81 characters long (digits).
        """
        try:
            import pandas as pd
        except Exception as e:
            raise ImportError("pandas is required to load CSV datasets: install pandas") from e

        print(f"Loading Sudoku from CSV: {path} (start={start_idx}, n={num_samples})")

        # Use pandas to read only the needed rows for efficiency
        if start_idx > 0:
            # skiprows expects an int or list of row indices to skip; easiest is
            # to use header=0 and then skiprows=range(1, start_idx+1)
            skiprows = list(range(1, start_idx + 1))
            df = pd.read_csv(path, skiprows=skiprows, nrows=num_samples)
        else:
            df = pd.read_csv(path, nrows=num_samples)

        # Detect columns
        puzzle_col = 'quizzes' if 'quizzes' in df.columns else df.columns[0]
        solution_col = 'solutions' if 'solutions' in df.columns else (df.columns[1] if len(df.columns) > 1 else None)
        if solution_col is None:
            raise ValueError("Could not find a solution column in CSV")

        puzzles = []
        solutions = []

        for _, row in df.iterrows():
            p = str(row[puzzle_col]).strip()
            s = str(row[solution_col]).strip()
            if len(p) != 81 or len(s) != 81:
                # skip malformed rows
                continue
            puzzles.append([int(ch) for ch in p])
            solutions.append([int(ch) for ch in s])

        if len(puzzles) == 0:
            raise ValueError(f"No valid puzzles loaded from CSV: {path}")

        self.puzzles = torch.from_numpy(np.array(puzzles, dtype=np.int32)).long()
        self.solutions = torch.from_numpy(np.array(solutions, dtype=np.int32)).long()
        print(f"✓ Loaded {len(self.puzzles)} puzzles from CSV: {path}")

    def _generate_data(self, num_samples: int, difficulty: str):
        """Generate Sudoku puzzles using the built-in generator."""
        puzzles, solutions = generate_sudoku_batch(num_samples, difficulty)
        self.puzzles = torch.from_numpy(puzzles).long()
        self.solutions = torch.from_numpy(solutions).long()

    def __len__(self):
        return len(self.solutions)

    def __getitem__(self, idx):
        """Return (puzzle, solution). Puzzle may contain 0s for unknowns."""
        return self.puzzles[idx], self.solutions[idx]

    def save(self, path: str):
        """Save dataset to an NPZ file for later reuse."""
        np.savez_compressed(path, puzzles=self.puzzles.numpy(), solutions=self.solutions.numpy())


# ============================================================================
# Sudoku Generator (Simple bactracking-based)
# ============================================================================

class SudokuGenerator:
    """Generate valid Sudoku puzzles."""
    
    @staticmethod
    def is_valid(board, row, col, num):
        """Check if placing num at (row, col) is valid."""
        # Check row
        if num in board[row]:
            return False
        
        # Check column
        if num in board[:, col]:
            return False
        
        # Check 3x3 box
        box_row, box_col = 3 * (row // 3), 3 * (col // 3)
        if num in board[box_row:box_row+3, box_col:box_col+3]:
            return False
        
        return True
    
    @staticmethod
    def solve(board):
        """Solve Sudoku using backtracking."""
        for i in range(9):
            for j in range(9):
                if board[i, j] == 0:
                    # Try digits 1-9
                    nums = np.random.permutation(9) + 1
                    for num in nums:
                        if SudokuGenerator.is_valid(board, i, j, num):
                            board[i, j] = num
                            if SudokuGenerator.solve(board):
                                return True
                            board[i, j] = 0
                    return False
        return True
    
    @staticmethod
    def generate_solution():
        """Generate a valid Sudoku solution."""
        board = np.zeros((9, 9), dtype=np.int32)
        SudokuGenerator.solve(board)
        return board
    
    @staticmethod
    def create_puzzle(solution, difficulty='medium'):
        """Create puzzle by removing numbers from solution."""
        puzzle = solution.copy()
        
        # Difficulty determines how many cells to remove
        if difficulty == 'easy':
            num_remove = np.random.randint(35, 45)
        elif difficulty == 'medium':
            num_remove = np.random.randint(45, 55)
        elif difficulty == 'hard':
            num_remove = np.random.randint(55, 65)
        else:  # mixed
            num_remove = np.random.randint(35, 65)
        
        positions = [(i, j) for i in range(9) for j in range(9)]
        np.random.shuffle(positions)
        
        for i in range(num_remove):
            row, col = positions[i]
            puzzle[row, col] = 0
        
        return puzzle


def generate_sudoku_batch(num_samples: int, difficulty: str = 'mixed'):
    """Generate a batch of Sudoku puzzles."""
    puzzles = []
    solutions = []
    
    for _ in tqdm(range(num_samples), desc="Generating Sudoku"):
        solution = SudokuGenerator.generate_solution()
        puzzle = SudokuGenerator.create_puzzle(solution, difficulty)
        
        puzzles.append(puzzle.flatten())
        solutions.append(solution.flatten())
    
    return np.array(puzzles), np.array(solutions)


# ============================================================================
# Masked Diffusion Model Architecture
# ============================================================================

class Sudoku2DPositionalEncoding(nn.Module):
    """
    2D Positional encoding that captures Sudoku structure.
    
    Encodes three types of position information:
    - Row position (0-8)
    - Column position (0-8)  
    - Box position (0-8, which 3x3 box)
    
    This gives the model explicit knowledge of Sudoku constraints.
    """
    
    def __init__(self, d_model: int):
        super().__init__()
        
        # Learnable embeddings for each structural component
        self.row_embedding = nn.Embedding(9, d_model // 3)
        self.col_embedding = nn.Embedding(9, d_model // 3)
        self.box_embedding = nn.Embedding(9, d_model - 2 * (d_model // 3))  # Remaining dims
        
        # Precompute position indices for a 9x9 grid flattened to 81
        rows = torch.arange(81) // 9  # [0,0,0,0,0,0,0,0,0,1,1,1,...]
        cols = torch.arange(81) % 9   # [0,1,2,3,4,5,6,7,8,0,1,2,...]
        boxes = (rows // 3) * 3 + (cols // 3)  # [0,0,0,1,1,1,2,2,2,0,0,0,...]
        
        self.register_buffer('rows', rows)
        self.register_buffer('cols', cols)
        self.register_buffer('boxes', boxes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add 2D positional encoding to input.
        
        Args:
            x: [batch_size, 81, d_model] - embedded tokens
            
        Returns:
            [batch_size, 81, d_model] - tokens with positional encoding added
        """
        batch_size = x.size(0)
        
        # Get positional embeddings
        row_pe = self.row_embedding(self.rows)  # [81, d_model//3]
        col_pe = self.col_embedding(self.cols)  # [81, d_model//3]
        box_pe = self.box_embedding(self.boxes)  # [81, remaining]
        
        # Concatenate to form full positional encoding
        pos_encoding = torch.cat([row_pe, col_pe, box_pe], dim=-1)  # [81, d_model]
        
        # Add to input (broadcast over batch)
        return x + pos_encoding.unsqueeze(0)


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer (legacy, 1D sinusoidal)."""
    
    def __init__(self, d_model: int, max_len: int = 81):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """Add positional encoding to input."""
        return x + self.pe[:, :x.size(1)]


class SudokuMDM(nn.Module):
    """
    Masked Diffusion Model for Sudoku.
    
    The model takes a partially masked Sudoku board and predicts marginal
    distributions for each position. Vocabulary: {0=MASK, 1-9=digits}.
    """
    
    def __init__(
        self,
        vocab_size: int = 10,  # 0 (mask) + 1-9 (digits)
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_seq_len: int = 81,  # 9x9 Sudoku
        use_2d_pos: bool = True  # Use 2D Sudoku-aware positional encoding
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Token embedding
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding - use 2D Sudoku-aware encoding by default
        if use_2d_pos:
            self.pos_encoder = Sudoku2DPositionalEncoding(d_model)
        else:
            self.pos_encoder = PositionalEncoding(d_model, max_seq_len)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output head - predicts logits for each position
        self.output_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, vocab_size)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: [batch_size, seq_len] - token indices
            
        Returns:
            logits: [batch_size, seq_len, vocab_size] - logits for each position
        """
        # Embed tokens
        x = self.embedding(x)  # [B, L, D]
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transform
        x = self.transformer(x)  # [B, L, D]
        
        # Predict logits
        logits = self.output_head(x)  # [B, L, V]
        
        return logits
    
    def get_marginals(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get marginal distributions for each position.
        
        Args:
            x: [batch_size, seq_len] - token indices
            
        Returns:
            marginals: [batch_size, seq_len, vocab_size] - probability distributions
        """
        logits = self.forward(x)
        return F.softmax(logits, dim=-1)


# ============================================================================
# Masking Schedule
# ============================================================================

class MaskSchedule:
    """Handles masking schedule for diffusion process."""
    
    def __init__(self, schedule_type: str = 'linear', total_steps: int = 10):
        self.schedule_type = schedule_type
        self.total_steps = total_steps
    
    def get_mask_ratio(self, step: int) -> float:
        """Get the masking ratio at a given step."""
        if self.schedule_type == 'linear':
            return 1.0 - (step / self.total_steps)
        elif self.schedule_type == 'cosine':
            return 0.5 * (1 + math.cos(math.pi * step / self.total_steps))
        elif self.schedule_type == 'sqrt':
            return math.sqrt(1.0 - step / self.total_steps)
        else:
            raise ValueError(f"Unknown schedule: {self.schedule_type}")
    
    def apply_mask(self, x: torch.Tensor, mask_ratio: float, 
                   given_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply masking to input.
        
        Args:
            x: [batch_size, seq_len] - original tokens (1-9)
            mask_ratio: fraction of positions to mask
            given_mask: optional fixed mask (e.g., from puzzle)
            
        Returns:
            masked_x: masked version with 0s
            mask: boolean mask indicating which positions were masked
        """
        batch_size, seq_len = x.shape
        
        if given_mask is not None:
            # Use provided mask (e.g., original puzzle)
            mask = given_mask.bool()
        else:
            # Random masking
            num_mask = int(seq_len * mask_ratio)
            mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=x.device)
            
            for i in range(batch_size):
                indices = torch.randperm(seq_len, device=x.device)[:num_mask]
                mask[i, indices] = True
        
        masked_x = x.clone()
        masked_x[mask] = 0  # 0 is the MASK token
        
        return masked_x, mask


# ============================================================================
# Training
# ============================================================================

class MDMTrainer:
    """Trainer for Masked Diffusion Model."""
    
    def __init__(
        self,
        model: SudokuMDM,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        mask_schedule: str = 'linear',
        num_diffusion_steps: int = 10
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=len(train_loader) * 300  # Assume 300 epochs
        )
        
        self.mask_schedule = MaskSchedule(mask_schedule, num_diffusion_steps)
        self.num_diffusion_steps = num_diffusion_steps
        
        self.train_losses = []
        self.val_losses = []
    
    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for puzzle, solution in pbar:
            puzzle = puzzle.to(self.device)
            solution = solution.to(self.device)
            
            # Sample random diffusion step
            t = np.random.randint(0, self.num_diffusion_steps + 1)
            mask_ratio = self.mask_schedule.get_mask_ratio(t)
            
            # Apply masking
            masked_solution, mask = self.mask_schedule.apply_mask(solution, mask_ratio)
            
            # Skip if no masked positions (shouldn't happen, but be safe)
            if not mask.any():
                continue
            
            # Forward pass
            logits = self.model(masked_solution)
            
            # Compute loss only on masked positions
            # Targets are 1-9, predictions are over vocab 0-9
            # Use label smoothing for better numerical stability
            loss = F.cross_entropy(
                logits[mask].view(-1, self.model.vocab_size),
                solution[mask].view(-1),
                reduction='mean',
                label_smoothing=0.01
            )
            
            # Skip if loss is NaN (numerical instability)
            if torch.isnan(loss) or torch.isinf(loss):
                if not hasattr(self, '_nan_warned'):
                    print(f"  Warning: NaN/Inf loss detected, skipping batch")
                    self._nan_warned = True
                continue
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / num_batches if num_batches > 0 else float('nan')
        self.train_losses.append(avg_loss)
        return avg_loss
    
    @torch.no_grad()
    def validate(self) -> float:
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        for puzzle, solution in self.val_loader:
            puzzle = puzzle.to(self.device)
            solution = solution.to(self.device)
            
            # Use random mask ratio for validation
            t = np.random.randint(0, self.num_diffusion_steps + 1)
            mask_ratio = self.mask_schedule.get_mask_ratio(t)
            masked_solution, mask = self.mask_schedule.apply_mask(solution, mask_ratio)
            
            # Skip if no masked positions
            if not mask.any():
                continue
            
            logits = self.model(masked_solution)
            
            loss = F.cross_entropy(
                logits[mask].view(-1, self.model.vocab_size),
                solution[mask].view(-1),
                reduction='mean',
                label_smoothing=0.01
            )
            
            # Skip if loss is NaN
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else float('nan')
        self.val_losses.append(avg_loss)
        return avg_loss
    
    def train(self, num_epochs: int, save_path: Optional[str] = None):
        """Full training loop."""
        best_val_loss = float('inf')
        
        for epoch in range(1, num_epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate()
            
            print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
            
            # Save best model
            if save_path and val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }, save_path)
                print(f"Saved best model to {save_path}")


# ============================================================================
# Sampling / Generation
# ============================================================================

class MDMSampler:
    """Sample from a trained MDM."""
    
    def __init__(self, model: SudokuMDM, device: str = 'cuda'):
        self.model = model.to(device)
        self.device = device
        self.model.eval()
    
    @torch.no_grad()
    def sample(
        self,
        num_samples: int = 1,
        num_steps: int = 10,
        temperature: float = 1.0,
        given_puzzle: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Generate Sudoku solutions using iterative unmasking.
        
        Args:
            num_samples: number of samples to generate
            num_steps: number of denoising steps
            temperature: sampling temperature
            given_puzzle: optional starting puzzle [num_samples, 81]
            
        Returns:
            samples: [num_samples, 81] - generated solutions
        """
        seq_len = 81
        
        # Start with all masked
        if given_puzzle is not None:
            x = given_puzzle.to(self.device)
            fixed_mask = (x != 0)  # Keep given numbers fixed
        else:
            x = torch.zeros(num_samples, seq_len, dtype=torch.long, device=self.device)
            fixed_mask = torch.zeros(num_samples, seq_len, dtype=torch.bool, device=self.device)
        
        # Iteratively unmask
        for step in range(num_steps):
            # Get marginals
            logits = self.model(x) / temperature
            probs = F.softmax(logits, dim=-1)
            
            # Find masked positions (excluding fixed positions)
            mask = (x == 0) & (~fixed_mask)
            
            if not mask.any():
                break
            
            # Number of positions to unmask this step
            num_masked = mask.sum(dim=1)
            #num_unmask = (num_masked * (1.0 / (num_steps - step))).long()
            num_unmask = 1

            # For each sample, unmask highest confidence predictions
            for i in range(num_samples):
                if num_unmask[i] == 0:
                    continue
                
                # Get confidence scores for masked positions
                masked_positions = mask[i].nonzero(as_tuple=True)[0]
                if len(masked_positions) == 0:
                    continue
                
                # Confidence = max probability (excluding MASK token)
                confidence = probs[i, masked_positions, 1:].max(dim=-1)[0]
                
                # Select top-k confident positions
                k = min(num_unmask[i].item(), len(masked_positions))
                top_k = confidence.topk(k).indices
                positions_to_unmask = masked_positions[top_k]
                
                # Sample tokens for these positions
                sampled_tokens = torch.multinomial(
                    probs[i, positions_to_unmask, 1:],  # Exclude MASK token
                    num_samples=1
                ).squeeze(-1) + 1  # +1 because we excluded 0
                
                x[i, positions_to_unmask] = sampled_tokens
        
        return x
    
    @torch.no_grad()
    def get_marginals(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get marginal distributions for a given configuration.
        
        Args:
            x: [batch_size, 81] - current state
            
        Returns:
            marginals: [batch_size, 81, 10] - probability distributions
        """
        x = x.to(self.device)
        return self.model.get_marginals(x)


# ============================================================================
# Evaluation
# ============================================================================

def check_sudoku_valid(board: np.ndarray) -> bool:
    """Check if a Sudoku solution is valid."""
    board = board.reshape(9, 9)
    
    # Check rows
    for i in range(9):
        if len(set(board[i])) != 9 or 0 in board[i]:
            return False
    
    # Check columns
    for j in range(9):
        if len(set(board[:, j])) != 9 or 0 in board[:, j]:
            return False
    
    # Check 3x3 boxes
    for box_i in range(3):
        for box_j in range(3):
            box = board[box_i*3:(box_i+1)*3, box_j*3:(box_j+1)*3].flatten()
            if len(set(box)) != 9 or 0 in box:
                return False
    
    return True


def evaluate_samples(samples: torch.Tensor) -> dict:
    """Evaluate generated samples."""
    samples = samples.cpu().numpy()
    num_samples = len(samples)
    
    num_valid = sum(check_sudoku_valid(s) for s in samples)
    validity_rate = num_valid / num_samples
    
    return {
        'validity_rate': validity_rate,
        'num_valid': num_valid,
        'num_samples': num_samples
    }


# ============================================================================
# Main training script
# ============================================================================

def main():
    """Main training function."""
    
    # Configuration
    config = {
        'num_train': 50000,
        'num_val': 5000,
        'batch_size': 64,
        'num_epochs': 50,
        'd_model': 256,
        'nhead': 8,
        'num_layers': 6,
        'dim_feedforward': 1024,
        'dropout': 0.1,
        'lr': 1e-4,
        'weight_decay': 0.01,
        'num_diffusion_steps': 10,
        'mask_schedule': 'linear',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    print("Configuration:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    
    # Create datasets
    print("\n" + "="*50)
    print("Creating datasets...")
    print("="*50)
    
    train_dataset = SudokuDataset(
        num_samples=config['num_train'],
        difficulty='mixed'
    )
    
    val_dataset = SudokuDataset(
        num_samples=config['num_val'],
        difficulty='mixed'
    )
    
    # Save datasets
    os.makedirs('data', exist_ok=True)
    train_dataset.save('data/train_sudoku.npz')
    val_dataset.save('data/val_sudoku.npz')
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Create model
    print("\n" + "="*50)
    print("Creating model...")
    print("="*50)
    
    model = SudokuMDM(
        vocab_size=10,
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        dim_feedforward=config['dim_feedforward'],
        dropout=config['dropout']
    )
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Create trainer
    trainer = MDMTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=config['device'],
        lr=config['lr'],
        weight_decay=config['weight_decay'],
        mask_schedule=config['mask_schedule'],
        num_diffusion_steps=config['num_diffusion_steps']
    )
    
    # Train
    print("\n" + "="*50)
    print("Training...")
    print("="*50)
    
    os.makedirs('checkpoints', exist_ok=True)
    trainer.train(
        num_epochs=config['num_epochs'],
        save_path='checkpoints/sudoku_mdm_best.pt'
    )
    
    # Test sampling
    print("\n" + "="*50)
    print("Testing sampling...")
    print("="*50)
    
    sampler = MDMSampler(model, device=config['device'])
    samples = sampler.sample(num_samples=100, num_steps=20)
    
    results = evaluate_samples(samples)
    print(f"\nSampling results:")
    print(f"  Validity rate: {results['validity_rate']*100:.2f}%")
    print(f"  Valid samples: {results['num_valid']}/{results['num_samples']}")
    
    print("\nTraining complete!")


if __name__ == '__main__':
    main()
