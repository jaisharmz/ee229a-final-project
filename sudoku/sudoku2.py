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
import os

# ============================================================================
# Sudoku Data Generation & Loading
# ============================================================================

class SudokuDataset(Dataset):
    """Dataset of Sudoku puzzles with solutions."""

    def __init__(self, data_path: Optional[str] = None, num_samples: int = 10000,
                 difficulty: str = "mixed", is_csv: bool = False, start_idx: int = 0):
        self.puzzles = []
        self.solutions = []

        if data_path and os.path.exists(data_path):
            if is_csv or data_path.lower().endswith('.csv'):
                self._load_from_csv(data_path, num_samples=num_samples, start_idx=start_idx)
            else:
                self._load_from_file(data_path)
        else:
            print(f"Generating {num_samples} Sudoku puzzles...")
            self._generate_data(num_samples, difficulty)

    def _load_from_file(self, path: str):
        data = np.load(path)
        self.puzzles = torch.from_numpy(data['puzzles']).long()
        self.solutions = torch.from_numpy(data['solutions']).long()
        print(f"Loaded {len(self.puzzles)} puzzles from {path}")

    def _load_from_csv(self, path: str, num_samples: Optional[int] = None, start_idx: int = 0):
        try:
            import pandas as pd
        except Exception as e:
            raise ImportError("pandas is required to load CSV datasets") from e

        print(f"Loading Sudoku from CSV: {path}")
        
        # Determine rows to read
        skiprows = range(1, start_idx + 1) if start_idx > 0 else None
        df = pd.read_csv(path, skiprows=skiprows, nrows=num_samples)

        puzzles = []
        solutions = []

        # Adjust column names based on Kaggle dataset standard or generic
        puzzle_col = 'quizzes' if 'quizzes' in df.columns else df.columns[0]
        solution_col = 'solutions' if 'solutions' in df.columns else df.columns[1]

        for _, row in df.iterrows():
            p = str(row[puzzle_col])
            s = str(row[solution_col])
            if len(p) == 81 and len(s) == 81:
                puzzles.append([int(ch) for ch in p])
                solutions.append([int(ch) for ch in s])

        self.puzzles = torch.from_numpy(np.array(puzzles, dtype=np.int32)).long()
        self.solutions = torch.from_numpy(np.array(solutions, dtype=np.int32)).long()
        print(f"Loaded {len(self.puzzles)} puzzles from CSV")

    def _generate_data(self, num_samples: int, difficulty: str):
        puzzles, solutions = generate_sudoku_batch(num_samples, difficulty)
        self.puzzles = torch.from_numpy(puzzles).long()
        self.solutions = torch.from_numpy(solutions).long()

    def __len__(self):
        return len(self.solutions)

    def __getitem__(self, idx):
        return self.puzzles[idx], self.solutions[idx]

    def save(self, path: str):
        np.savez_compressed(path, puzzles=self.puzzles.numpy(), solutions=self.solutions.numpy())


# ============================================================================
# Sudoku Generator (Backtracking)
# ============================================================================

class SudokuGenerator:
    @staticmethod
    def is_valid(board, row, col, num):
        if num in board[row]: return False
        if num in board[:, col]: return False
        box_row, box_col = 3 * (row // 3), 3 * (col // 3)
        if num in board[box_row:box_row+3, box_col:box_col+3]: return False
        return True
    
    @staticmethod
    def solve(board):
        for i in range(9):
            for j in range(9):
                if board[i, j] == 0:
                    nums = np.random.permutation(9) + 1
                    for num in nums:
                        if SudokuGenerator.is_valid(board, i, j, num):
                            board[i, j] = num
                            if SudokuGenerator.solve(board): return True
                            board[i, j] = 0
                    return False
        return True
    
    @staticmethod
    def generate_solution():
        board = np.zeros((9, 9), dtype=np.int32)
        SudokuGenerator.solve(board)
        return board
    
    @staticmethod
    def create_puzzle(solution, difficulty='medium'):
        puzzle = solution.copy()
        if difficulty == 'easy': num_remove = np.random.randint(35, 45)
        elif difficulty == 'medium': num_remove = np.random.randint(45, 55)
        elif difficulty == 'hard': num_remove = np.random.randint(55, 65)
        else: num_remove = np.random.randint(35, 65)
        
        positions = [(i, j) for i in range(9) for j in range(9)]
        np.random.shuffle(positions)
        for i in range(num_remove):
            row, col = positions[i]
            puzzle[row, col] = 0
        return puzzle

def generate_sudoku_batch(num_samples: int, difficulty: str = 'mixed'):
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
    def __init__(self, d_model: int):
        super().__init__()
        self.row_embedding = nn.Embedding(9, d_model // 3)
        self.col_embedding = nn.Embedding(9, d_model // 3)
        self.box_embedding = nn.Embedding(9, d_model - 2 * (d_model // 3))
        
        rows = torch.arange(81) // 9
        cols = torch.arange(81) % 9
        boxes = (rows // 3) * 3 + (cols // 3)
        
        self.register_buffer('rows', rows)
        self.register_buffer('cols', cols)
        self.register_buffer('boxes', boxes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        row_pe = self.row_embedding(self.rows)
        col_pe = self.col_embedding(self.cols)
        box_pe = self.box_embedding(self.boxes)
        pos_encoding = torch.cat([row_pe, col_pe, box_pe], dim=-1)
        return x + pos_encoding.unsqueeze(0)

class SudokuMDM(nn.Module):
    def __init__(
        self,
        vocab_size: int = 10,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_seq_len: int = 81
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = Sudoku2DPositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.output_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, vocab_size)
        )
        
        self._init_weights()
    
    # --- FIX: Use @property instead of self.fc_out = ... in __init__ ---
    @property
    def fc_out(self):
        return self.output_head

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1: nn.init.xavier_uniform_(p)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        logits = self.output_head(x)
        return logits
    
    def get_marginals(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        return F.softmax(logits, dim=-1)


# ============================================================================
# Utilities
# ============================================================================

class MaskSchedule:
    def __init__(self, schedule_type: str = 'linear', total_steps: int = 10):
        self.schedule_type = schedule_type
        self.total_steps = total_steps
    
    def get_mask_ratio(self, step: int) -> float:
        if self.schedule_type == 'linear': return 1.0 - (step / self.total_steps)
        elif self.schedule_type == 'cosine': return 0.5 * (1 + math.cos(math.pi * step / self.total_steps))
        else: return math.sqrt(1.0 - step / self.total_steps)
    
    def apply_mask(self, x: torch.Tensor, mask_ratio: float, given_mask=None):
        batch_size, seq_len = x.shape
        if given_mask is not None:
            mask = given_mask.bool()
        else:
            num_mask = int(seq_len * mask_ratio)
            mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=x.device)
            for i in range(batch_size):
                indices = torch.randperm(seq_len, device=x.device)[:num_mask]
                mask[i, indices] = True
        
        masked_x = x.clone()
        masked_x[mask] = 0
        return masked_x, mask

def check_sudoku_valid(board: np.ndarray) -> bool:
    board = board.reshape(9, 9)
    # Check rows, cols, boxes
    for i in range(9):
        if len(set(board[i])) != 9 or 0 in board[i]: return False
        if len(set(board[:, i])) != 9 or 0 in board[:, i]: return False
    for r in range(0, 9, 3):
        for c in range(0, 9, 3):
            box = board[r:r+3, c:c+3].flatten()
            if len(set(box)) != 9 or 0 in box: return False
    return True

def evaluate_samples(samples: torch.Tensor) -> dict:
    samples = samples.cpu().numpy()
    num_valid = sum(check_sudoku_valid(s) for s in samples)
    return {'validity_rate': num_valid / len(samples), 'num_valid': num_valid}

class MDMTrainer:
    """Trainer for Masked Diffusion Model."""
    def __init__(self, model, train_loader, val_loader, device='cuda', lr=1e-4, 
                 weight_decay=0.01, mask_schedule='linear', num_diffusion_steps=10):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=len(train_loader)*50)
        self.mask_schedule = MaskSchedule(mask_schedule, num_diffusion_steps)
        self.num_steps = num_diffusion_steps

    def train(self, num_epochs, save_path=None):
        best_loss = float('inf')
        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0
            for puzzle, solution in tqdm(self.train_loader, desc=f"Epoch {epoch+1}"):
                puzzle, solution = puzzle.to(self.device), solution.to(self.device)
                
                # Random Masking
                t = np.random.randint(0, self.num_steps + 1)
                ratio = self.mask_schedule.get_mask_ratio(t)
                masked_sol, mask = self.mask_schedule.apply_mask(solution, ratio)
                
                if not mask.any(): continue
                
                logits = self.model(masked_sol)
                loss = F.cross_entropy(
                    logits[mask].view(-1, 10), solution[mask].view(-1), label_smoothing=0.01
                )
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.scheduler.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(self.train_loader)
            print(f"Epoch {epoch+1}: Loss {avg_loss:.4f}")
            if save_path and avg_loss < best_loss:
                best_loss = avg_loss
                torch.save({'model_state_dict': self.model.state_dict()}, save_path)

class MDMSampler:
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device
        self.model.eval()

    @torch.no_grad()
    def sample(self, num_samples=1, num_steps=10):
        x = torch.zeros(num_samples, 81, dtype=torch.long, device=self.device)
        for _ in range(num_steps):
            logits = self.model(x)
            probs = F.softmax(logits, dim=-1)
            
            mask = (x == 0)
            if not mask.any(): break
            
            # Simple greedy sampling for demo
            # In real usage, unmask top-k confident positions per step
            preds = torch.argmax(probs, dim=-1)
            
            # Unmask 1/num_steps percent of tokens
            for i in range(num_samples):
                m_indices = mask[i].nonzero(as_tuple=True)[0]
                if len(m_indices) == 0: continue
                
                # Pick highest confidence
                confs = probs[i, m_indices].max(dim=-1)[0]
                k = max(1, len(m_indices) // (num_steps // 2 + 1))
                topk = torch.topk(confs, k).indices
                to_unmask = m_indices[topk]
                
                x[i, to_unmask] = preds[i, to_unmask]
        return x