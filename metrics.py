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

    @staticmethod
    def compute_metrics_for_matrix(attn_matrix: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
        """
        Vectorized computation of metrics for each row of a given unnormalized attention matrix.

        Parameters:
            attn_matrix (torch.Tensor): 2D tensor of shape (num_rows, num_keys) with raw attention scores.
            eps (float): Small constant for numerical stability.

        Returns:
            torch.Tensor: Tensor of shape (num_rows, 8) containing metrics for each row:
                [Shannon Entropy, Gini Coefficient, Peak-to-Average Ratio, Std, Mean, LSE,
                 Energy Ratio, Gradient Norm]
        """
        # attn_matrix: shape (num_rows, num_keys)
        num_keys = attn_matrix.shape[-1]

        # Compute softmax along keys for each row (shape: num_rows x num_keys)
        p = F.softmax(attn_matrix, dim=-1)

        # 1. Shannon entropy: -sum_i p_i log(p_i)
        entropy = -torch.sum(p * torch.log(p + eps), dim=-1)  # shape: (num_rows,)

        # 2. Gini coefficient: for each row, compute pairwise differences.
        # p has shape (num_rows, num_keys); unsqueeze to get (num_rows, num_keys, 1) and (num_rows, 1, num_keys)
        diff = torch.abs(p.unsqueeze(2) - p.unsqueeze(1))  # shape: (num_rows, num_keys, num_keys)
        gini = torch.sum(diff, dim=(-1, -2)) / (2.0 * num_keys)  # shape: (num_rows,)

        # 3. Peak-to-Average Ratio: max(p)/mean(p)
        p_max = torch.max(p, dim=-1)[0]
        p_mean = torch.mean(p, dim=-1)
        peak_avg = p_max / (p_mean + eps)

        # 4. Standard deviation of p per row
        std_p = torch.std(p, dim=-1)

        # 5. Mean of p per row
        mean_p = torch.mean(p, dim=-1)

        # 6. Log-sum-exp of raw row (without normalization)
        lse = torch.logsumexp(attn_matrix, dim=-1)

        # 7. Energy Ratio: sum(abs(raw_row)) / (L2 norm of raw_row)
        total_energy = torch.sum(torch.abs(attn_matrix), dim=-1)
        fro_norm = torch.norm(attn_matrix, p=2, dim=-1)
        energy_ratio = total_energy / (fro_norm + eps)

        # 8. Gradient norm: L2 norm of softmax vector (p)
        grad_norm = torch.norm(p, p=2, dim=-1)

        # Stack all metrics: resulting shape (num_rows, 8)
        metrics = torch.stack([entropy, gini, peak_avg, std_p, mean_p, lse, energy_ratio, grad_norm], dim=-1)
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
        # Vectorized over batch and heads: reshape to (batch * n_heads, num_rows, num_keys)
        attn_reshaped = attn.view(-1, n_rows, n_keys)
        metrics_list = []
        # Process each sample-head combination in a vectorized way.
        # Here, compute_metrics_for_matrix is fully vectorized over the row dimension.
        for i in range(attn_reshaped.shape[0]):
            metrics_list.append(self.compute_metrics_for_matrix(attn_reshaped[i]))
        # Stack along the first dimension and then reshape back to (batch, n_heads, n_rows, 8)
        all_metrics = torch.stack(metrics_list, dim=0)
        all_metrics = all_metrics.view(batch, n_heads, n_rows, 8)
        return all_metrics

    def log_metrics(self, attn: torch.Tensor) -> None:
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

        # Compute metrics; shape: (batch, attn_heads, 1024, 8)
        metrics: torch.Tensor = self.compute_metrics(attn)
        # Convert to numpy with float16 precision
        metrics_np: np.ndarray = metrics.cpu().numpy().astype(np.float16)
        filename: str = f"{self.block_type}_block-{self.layer_idx}_layer-{self.curr_time_stamp}_timestamp.npy"
        file_path: str = os.path.join(self.save_dir, filename)
        np.save(file_path, metrics_np)
        # print(f"Saved metrics for {self.block_type} block, layer {self.layer_idx} at {file_path}")




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
    pattern = os.path.join(src_dir, f"{block_type}_block-{layer_idx}_layer-*_timestamp.npy")
    print("pattern:", pattern)
    file_list: List[str] = glob.glob(pattern)
    print("files: ", file_list)

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


# iterate over all the files in directory results/metrics/seed-97 with their full path relative to results/metrics/seed-97
def stitch_time_stamps(src_dir, dest_dir):
    for folder1 in glob.glob(src_dir+"/*"):
        folder1_name = folder1.split("/")[-1]
        for folder2 in glob.glob(folder1 + "/*"):
            folder2_name = folder2.split("/")[-1]
            for folder3 in glob.glob(folder2 + "/*"):
                folder3_name = folder3.split("/")[-1]
                
                dir1 , dir2 = folder3 , os.path.join(dest_dir, folder1_name, folder2_name, folder3_name)
                for block_type in ["down", "mid", "up"]:
                    for layer_idx in range(1, 4):
                        concatenate_metric_files(dir1, dir2, block_type, layer_idx)
       
if __name__=="__main__":     
    stitch_time_stamps("results/metrics/seed-97", "results/metrics/seed-97-time-stitched")