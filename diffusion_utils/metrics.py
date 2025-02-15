import os, glob
from typing import List
import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
import pickle

class AttentionMetricsLogger:
    def __init__(self, block_type: str, save_dir: str):
        """
        Parameters:
            block_type (str): The block type (e.g., 'down', 'mid', or 'up').
            save_dir (str): Directory where metric files will be saved.
        """
        self.save_dir: str = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.curr_time_stamp: int = 0
        self.block_type: str = block_type
        self.layer_idx: int = 0
        self.metric_tracker = defaultdict(list)

    def compute_l2_difference(Mat1: torch.Tensor, Mat2: torch.Tensor) -> float:
        """
        Compute the L2 norm difference between two tensors.

        Args:
            prev (Tensor): The previous state tensor.
            curr (Tensor): The current state tensor.

        Returns:
            float: The Euclidean (L2) norm of (curr - prev) as a Python float.
        """
        return torch.norm(Mat1- Mat2).item()
        

    def log_metrics(self, Q1:torch.Tensor, Q2:torch.Tensor) -> None:
        """
        Compute row-level metrics for the provided raw unnormalized attention tensor and save as a NumPy file.
        The filename is based on the block type, the current layer index (incremented each time),
        and the current time stamp.

        Parameters:
            attn (torch.Tensor): Raw unnormalized attention tensor of shape (batch, attn_heads, 1024, 1024).
        """
        # For demonstration, only log if current timestamp mod 3 == 0 and layer_idx <= 5.
        # (This is from your provided logic; you can adjust as needed.)
        self.layer_idx += 1
        if self.curr_time_stamp % 3 > 0 or self.layer_idx > 3:
            return

        l2_norm = self.compute_l2_difference(Q1, Q2)
        self.metric_tracker[self.layer_idx].append(l2_norm)
        return
    
    def save_metrics(self):
        """
        Save the metrics to disk as pickle files.
        """
        with open(os.path.join(self.save_dir, f"{self.block_type}_block.pkl"), "wb") as f:
            pickle.dump(self.metric_tracker, f)
            print(f"Metrics saved to {self.save_dir}/{self.block_type}_block.pkl")
