import os, glob
import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
import pickle

class AttentionMetricsLogger:
    def __init__(self, block_type: str, save_dir: str, metric_file_name:str=""):
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
        self.metric_file_name = metric_file_name
        self.metric_tracker = defaultdict(list)

    def update_time_stamp(self, time_stamp: int) -> None:
            """
            Update the current time stamp.

            Args:
                time_stamp (int): The new time stamp.
            """
            self.curr_time_stamp = time_stamp
            self.layer_idx = 0   # Reset layer index for new time stamp.

    def compute_l2_difference(self, Mat1: torch.Tensor, Mat2: torch.Tensor) -> float:
        """
        Compute the L2 norm difference between two tensors.

        Args:
            prev (Tensor): The previous state tensor.
            curr (Tensor): The current state tensor.

        Returns:
            float: The Euclidean (L2) norm of (curr - prev) as a Python float.
        """
        return torch.mean(torch.norm(Mat1- Mat2, dtype=torch.float32, dim=[1,2])).item()


    def laplacian_variance(self, tensor):
        """Calculate average variance of Laplacian across channels."""
        variances = []
        tensor = tensor.to(torch.float64)
        tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())  # Normalize to [0, 1]
        tensor = (tensor * 512).to(torch.uint16)

        laplacian_kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float64, device=tensor.device)
        laplacian_kernel = laplacian_kernel.view(1, 1, 3, 3)

        tensor = tensor.unsqueeze(1).to(torch.float64)  # Shape (C, 1, H, W)
        laplacian = F.conv2d(tensor, laplacian_kernel, padding=1)
        variances = laplacian.var(dim=[2, 3])

        return variances.mean().item()

    def gradient_entropy(self, tensor):
        """Calculate average entropy of the gradient magnitude across channels."""
        tensor = tensor.to(torch.float64)
        tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())  # Normalize to [0, 1]
        tensor = (tensor * 512).to(torch.uint16)

        sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32, device=tensor.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32, device=tensor.device).view(1, 1, 3, 3)

        entropies = []
        for c in range(tensor.shape[0]):
            channel = tensor[c].unsqueeze(0).unsqueeze(0).to(torch.float32)  # Shape (1, 1, H, W)
            grad_x = F.conv2d(channel, sobel_x, padding=1)
            grad_y = F.conv2d(channel, sobel_y, padding=1)
            magnitude = torch.sqrt(grad_x**2 + grad_y**2)

            hist = torch.histc(magnitude, bins=512, min=0, max=512)
            hist = hist / hist.sum()
            entropy = -torch.sum(hist * torch.log2(hist + 1e-10)).item()
            entropies.append(entropy)
        return sum(entropies) / len(entropies)


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

        if self.layer_idx > 3:
            return

        l2_norm = self.compute_l2_difference(Q1, Q2)
        laplacian_variance = self.laplacian_variance(Q2 - Q1)
        gradient_entropy = self.gradient_entropy(Q2 - Q1)
        metrics = {
            "l2_norm": l2_norm,
            "laplacian_variance": laplacian_variance,
            "gradient_entropy": gradient_entropy
        }
        self.metric_tracker[self.layer_idx].append(metrics)
        return

    def save_metrics(self):
        """
        Save the metrics to disk as pickle files.
        """
        with open(os.path.join(self.save_dir, f"{self.metric_file_name}___block-{self.block_type}.pkl"), "wb") as f:
            pickle.dump(self.metric_tracker, f)
            print(f"Metrics saved to {self.save_dir}/{self.block_type}_block.pkl")
