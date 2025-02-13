import os, glob
from typing import List
import torch
import torch.nn.functional as F
import numpy as np


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
    
    def update_time_stamp(self, time_stamp: int) -> None:
        """Update the time stamp for the current run and reset the layer index to 0."""
        self.curr_time_stamp = time_stamp
        self.layer_idx = 0
        
    @staticmethod
    def shannon_entropy(p: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
        """
        Compute Shannon entropy for a 1D probability vector p.
        p is assumed to be already L1-normalized via softmax.
        """
        return -torch.sum(p * torch.log(p + eps))
    
    @staticmethod
    def gini_coeff(p: torch.Tensor) -> torch.Tensor:
        """
        Compute the Gini coefficient for a 1D probability vector p.
        p is assumed to be nonnegative and to sum to 1.
        """
        n = p.numel()
        diff_matrix = torch.abs(p.unsqueeze(0) - p.unsqueeze(1))
        return torch.sum(diff_matrix) / (2.0 * n)
    
    @staticmethod
    def peak_to_average_ratio(p: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
        """
        Compute the ratio of the maximum value in p to its mean.
        p is assumed to be a 1D probability vector.
        """
        return torch.max(p) / (torch.mean(p) + eps)
    
    @staticmethod
    def row_std(p: torch.Tensor) -> torch.Tensor:
        """Compute the standard deviation of a 1D tensor p."""
        return torch.std(p)
    
    @staticmethod
    def row_mean(p: torch.Tensor) -> torch.Tensor:
        """Compute the mean of a 1D tensor p."""
        return torch.mean(p)
    
    @staticmethod
    def lse_raw(z: torch.Tensor) -> torch.Tensor:
        """
        Compute the log-sum-exp of the raw (unnormalized) 1D tensor z using torch.logsumexp.
        """
        return torch.logsumexp(z, dim=-1)
    
    @staticmethod
    def energy_ratio(z: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
        """
        Compute the energy ratio for a 1D raw vector z.
        Defined as the ratio of the sum of absolute values (total energy)
        to the L2 norm (Frobenius norm for a vector) of z.
        """
        total_energy = torch.sum(torch.abs(z))
        fro_norm = torch.norm(z, p=2)
        return total_energy / (fro_norm + eps)
    
    @staticmethod
    def gradient_norm(z: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
        """
        Compute the gradient norm of the energy function E = logsumexp(z) with respect to z.
        Since dE/dz = softmax(z), this is the L2 norm of the softmax of z.
        """
        p = F.softmax(z, dim=-1)
        return torch.norm(p, p=2)
    
    def compute_metrics_for_matrix(self, attn_matrix: torch.Tensor) -> torch.Tensor:
        """
        Compute metrics for each row of a given unnormalized attention matrix.
        
        Parameters:
            attn_matrix (torch.Tensor): 2D tensor of shape (num_rows, num_keys) with raw attention scores.
        
        Returns:
            torch.Tensor: Tensor of shape (num_rows, 8) containing metrics for each row:
                          [Shannon Entropy, Gini Coefficient, Peak-to-Average Ratio, Std, Mean, LSE,
                           Energy Ratio, Gradient Norm]
        """
        num_rows = attn_matrix.shape[0]
        metrics = torch.zeros((num_rows, 8), dtype=torch.float32, device=attn_matrix.device)
        eps = 1e-10
        
        for i in range(num_rows):
            raw_row: torch.Tensor = attn_matrix[i]  # shape: (num_keys,)
            # Normalize with softmax along the last dimension.
            p: torch.Tensor = F.softmax(raw_row, dim=-1)
            
            metrics[i, 0] = self.shannon_entropy(p, eps)
            metrics[i, 1] = self.gini_coeff(p)
            metrics[i, 2] = self.peak_to_average_ratio(p, eps)
            metrics[i, 3] = self.row_std(p)
            metrics[i, 4] = self.row_mean(p)
            metrics[i, 5] = self.lse_raw(raw_row)
            metrics[i, 6] = self.energy_ratio(raw_row, eps)
            metrics[i, 7] = self.gradient_norm(raw_row, eps)
        return metrics
    
    def compute_metrics(self, attn: torch.Tensor) -> torch.Tensor:
        """
        Compute metrics for each row for all samples and attention heads.
        
        Parameters:
            attn (torch.Tensor): Raw unnormalized attention tensor of shape 
                (batch, attn_heads, num_rows, num_keys).
        
        Returns:
            torch.Tensor: Tensor of shape (batch, attn_heads, num_rows, 8) containing metrics for each row.
        """
        batch, n_heads, n_rows, n_keys = attn.shape
        all_metrics = torch.zeros((batch, n_heads, n_rows, 8), dtype=torch.float32, device=attn.device)
        for b in range(batch):
            for h in range(n_heads):
                head_attn: torch.Tensor = attn[b, h, :, :]  # shape: (n_rows, n_keys)
                all_metrics[b, h] = self.compute_metrics_for_matrix(head_attn)
        return all_metrics
    
    def log_metrics(self, attn: torch.Tensor) -> None:
        """
        Compute row-level metrics for the provided raw unnormalized attention tensor and save as a NumPy file.
        The filename is based on the block type, the current layer index (which is incremented each time),
        and the current time stamp.
        
        Parameters:
            attn (torch.Tensor): Raw unnormalized attention tensor of shape (batch, attn_heads, 1024, 1024).
        """
        self.layer_idx += 1
        # Compute metrics; shape: (batch, attn_heads, 1024, 8)
        metrics: torch.Tensor = self.compute_metrics(attn)
        # Convert to a NumPy array with float16 precision.
        metrics_np: np.ndarray = metrics.cpu().numpy().astype(np.float16)
        filename: str = f"{self.block_type}_block-{self.layer_idx}_layer-{self.curr_time_stamp}_timestamp.npy"
        file_path: str = os.path.join(self.save_dir, filename)
        np.save(file_path, metrics_np)
        print(f"Saved metrics for {self.block_type} block, layer {self.layer_idx} at {file_path}")




def concatenate_metric_files(src_dir: str, dest_dir: str, block_type: str, layer_idx: int) -> None:
    """
    Searches for metric files in src_dir matching a given block type and layer index,
    concatenates the numpy arrays along a new time dimension, and saves the result as an .npy file in dest_dir.
    
    Parameters:
        src_dir (str): Directory containing the individual metric files.
        dest_dir (str): Directory where the concatenated file will be saved.
        block_type (str): Block type (e.g., 'down', 'mid', 'up').
        layer_idx (int): Layer index within the block.
    
    The function expects file names to have the format:
        "{block_type}_block-{layer_idx}_layer-{time_stamp}_time_stamp.npy"
    where {time_stamp} is a string representing the time stamp.
    """
    
    # Ensure destination directory exists.
    os.makedirs(dest_dir, exist_ok=True)
    
    # Construct the glob pattern to match files.
    pattern = os.path.join(src_dir, f"{block_type}_block-{layer_idx}_layer-*_time_stamp.npy")
    file_list: List[str] = glob.glob(pattern)
    
    if not file_list:
        print(f"No files found for {block_type} block, layer {layer_idx} in {src_dir}.")
        return
    
    # Extract timestamps from filenames and sort files by timestamp (lexicographically).
    # Assuming the file format: "{block_type}_block-{layer_idx}_layer-{time_stamp}_time_stamp.npy"
    def extract_timestamp(filename: str) -> str:
        base = os.path.basename(filename)
        # Split based on known separators.
        # Example: "mid_block-3_layer-2025-02-13T19-30-00_time_stamp.npy"
        try:
            parts = base.split('_')
            # parts[2] should be "layer-{time_stamp}" so remove "layer-" prefix.
            time_str = parts[2]
            if time_str.startswith("layer-"):
                return time_str[len("layer-"):]
            return time_str
        except Exception as e:
            print(f"Error extracting timestamp from {base}: {e}")
            return base  # fallback

    file_list.sort(key=extract_timestamp)
    
    # List to hold the metric arrays with a new time dimension.
    metrics_list: List[np.ndarray] = []
    
    for file in file_list:
        # Load the NumPy array
        arr = np.load(file)
        # Check if the array has shape (batch, attn_heads, num_rows, k)
        if arr.ndim != 4:
            print(f"File {file} has unexpected shape {arr.shape}. Skipping.")
            continue
        # Add a new axis at position 0 for time, resulting in shape (1, batch, attn_heads, num_rows, k)
        arr_time = np.expand_dims(arr, axis=0)
        metrics_list.append(arr_time)
    
    if not metrics_list:
        print("No valid metric files to concatenate.")
        return
    
    # Concatenate along the new time dimension (axis=0)
    concatenated: np.ndarray = np.concatenate(metrics_list, axis=0)
    # Ensure the data type is float16.
    concatenated = concatenated.astype(np.float16)
    
    # Define destination file name.
    dest_filename = f"{block_type}_block-{layer_idx}_concatenated.npy"
    dest_path = os.path.join(dest_dir, dest_filename)
    
    # Save the concatenated array.
    np.save(dest_path, concatenated)
    print(f"Saved concatenated metrics for {block_type} block, layer {layer_idx} at {dest_path}")
