import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from typing import Tuple, Optional

<<<<<<< HEAD
<<<<<<< HEAD
=======
class CNTK:
    """
    Convolutional Neural Tangent Kernel implementation.
    
    Implements exact CNTK computation for ReLU networks with:
    - Vanilla CNTK (CNTK-V): Sum over all spatial locations
    - Global Average Pooling CNTK (CNTK-GAP): Mean over spatial locations
    
    Based on Arora et al., NeurIPS 2019.
    """
    
    def __init__(self, depth: int, use_gap: bool = True, filter_size: int = 3):
        """
        Initialize CNTK.
        
        Args:
            depth: Number of convolutional layers
            use_gap: If True, use Global Average Pooling (CNTK-GAP)
                    If False, use Vanilla CNTK (CNTK-V)
            filter_size: Convolutional filter size (default 3x3)
        """
        self.depth = depth
        self.use_gap = use_gap
        self.filter_size = filter_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def _kappa0(self, rho: torch.Tensor) -> torch.Tensor:
        """
        Compute kappa_0 (derivative covariance) for ReLU.
        E[sigma'(u) * sigma'(v)] = (1/2pi) * (pi - arccos(rho))
        
        This is the "dot-product" kernel for ReLU derivatives.
        """
        # Clamp rho to valid range for numerical stability
        rho = torch.clamp(rho, -1.0 + 1e-7, 1.0 - 1e-7)
        return (1.0 / (2 * np.pi)) * (np.pi - torch.arccos(rho))
    
    def _kappa1(self, rho: torch.Tensor) -> torch.Tensor:
        """
        Compute kappa_1 (activation covariance) for ReLU.
        E[sigma(u) * sigma(v)] = (1/2pi) * (rho * (pi - arccos(rho)) + sqrt(1 - rho^2))
        
        Note: This returns the normalized version (assuming unit variance inputs).
        The actual covariance is sqrt(Sigma_11 * Sigma_22) * kappa1(rho).
        """
        rho = torch.clamp(rho, -1.0 + 1e-7, 1.0 - 1e-7)
        angle = torch.arccos(rho)
        return (1.0 / (2 * np.pi)) * (rho * (np.pi - angle) + torch.sqrt(1 - rho**2))
    
    def _compute_base_kernel(self, X1: torch.Tensor, X2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute base case kernel: Sigma^(0) and Theta^(0).
        
        For the base case (l=0), we compute the inner product of input patches.
        
        Args:
            X1: First set of images [N1, C, H, W]
            X2: Second set of images [N2, C, H, W]
        
        Returns:
            Sigma: Covariance tensor [N1, N2, H, W, H, W]
            Theta: NTK tensor [N1, N2, H, W, H, W] (initialized same as Sigma)
        """
        N1, C, H, W = X1.shape
        N2 = X2.shape[0]
        
        # Flatten spatial dimensions for inner product computation
        # X1: [N1, C, H*W], X2: [N2, C, H*W]
        X1_flat = X1.reshape(N1, C, H * W)
        X2_flat = X2.reshape(N2, C, H * W)
        
        # Compute patch-wise inner products
        # Result: [N1, N2, H*W, H*W] representing inner products between all patch pairs
        # For base case, patches are single pixels (1x1 receptive field)
        Sigma = torch.einsum('nci,mcj->nmij', X1_flat, X2_flat)  # [N1, N2, H*W, H*W]
        
        # Normalize by number of channels (variance = c_sigma^2 / C * sum)
        # Using c_sigma = 1 for simplicity
        Sigma = Sigma / C
        
        # Reshape to [N1, N2, H, W, H, W]
        Sigma = Sigma.reshape(N1, N2, H, W, H, W)
        
        # Initialize Theta = Sigma for base case
        Theta = Sigma.clone()
        
        return Sigma, Theta
    
    def _apply_conv_layer(self, Sigma: torch.Tensor, Theta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply one convolutional layer transformation.
        
        Implements the recursive formula:
        Sigma^(l+1)_{ij} = sum over (a,b) in receptive field of kappa1(rho^(l)_{i+a, j+b})
        Theta^(l+1) = Sigma^(l+1) + Theta^(l) * kappa0(rho^(l))
        
        Args:
            Sigma: Current covariance tensor [N1, N2, H, W, H, W]
            Theta: Current NTK tensor [N1, N2, H, W, H, W]
        
        Returns:
            New Sigma and Theta tensors
        """
        N1, N2, H, W, _, _ = Sigma.shape
        k = self.filter_size
        pad = k // 2  # Same padding to preserve spatial dimensions
        
        # Extract diagonal elements for normalization
        # Sigma_ii for X1: diagonal over last two spatial dims at i,j position
        # We need Sigma[n1, n1, i, j, i, j] but we're computing cross-kernel
        # For proper normalization, we need the self-kernels
        
        # Compute correlation coefficient rho = Sigma_12 / sqrt(Sigma_11 * Sigma_22)
        # We approximate Sigma_11 and Sigma_22 from the diagonal of the cross-kernel
        # In practice, for efficiency, we track variances separately
        
        # For simplicity, we compute variances from the kernel itself
        # Sigma_11[i,j] = Sigma[n1, n1, i, j, i, j] (self-kernel)
        # We'll use a different approach: maintain unit variance by rescaling
        
        # Get standard deviations for normalization
        # Sigma has shape [N1, N2, H, W, H, W]
        # For position (i1, j1) in image 1 and (i2, j2) in image 2:
        # We need sqrt(Sigma[n1,n1,i1,j1,i1,j1]) and sqrt(Sigma[n2,n2,i2,j2,i2,j2])
        
        # Since we're computing K(X,X) typically, we can extract diagonals
        # For now, use the diagonal of spatial dimensions as variance proxy
        
        # Reshape for easier manipulation
        # [N1, N2, H, W, H, W] -> work with 4D spatial kernel
        
        # Compute diagonal variances (same position in both images)
        # Shape: [N1, N2, H, W]
        diag_indices = torch.arange(H * W, device=Sigma.device)
        Sigma_flat = Sigma.reshape(N1, N2, H * W, H * W)
        
        # Extract diagonal: Sigma[..., i, i] for all i
        var1 = torch.diagonal(Sigma_flat, dim1=2, dim2=3)  # [N1, N2, H*W]
        var1 = var1.reshape(N1, N2, H, W)
        
        # For proper normalization, we need variance of each image at each position
        # var1_expanded[n1, n2, i1, j1, i2, j2] = var[n1, i1, j1]
        # var2_expanded[n1, n2, i1, j1, i2, j2] = var[n2, i2, j2]
        
        # Compute std from variance (add small epsilon for stability)
        eps = 1e-8
        
        # For correlation, we need Sigma_11 and Sigma_22
        # When computing K(X,Y), we'd need separate self-kernels
        # For simplicity in K(X,X), use the diagonal
        
        # Reshape var1 for broadcasting
        # var1: [N1, N2, H, W] represents variance at position (h,w)
        # We need std1[n1, n2, i, j, :, :] = sqrt(var1[n1, n1, i, j])
        # and std2[n1, n2, :, :, i, j] = sqrt(var1[n2, n2, i, j])
        
        # Extract self-kernel variances for proper normalization
        # std1[i,j] from image 1, std2[i,j] from image 2
        std1 = torch.sqrt(var1 + eps).unsqueeze(-1).unsqueeze(-1)  # [N1, N2, H, W, 1, 1]
        std2 = torch.sqrt(var1 + eps).unsqueeze(-3).unsqueeze(-3)  # [N1, N2, 1, 1, H, W]
        
        # Compute correlation
        rho = Sigma / (std1 * std2 + eps)
        rho = torch.clamp(rho, -1.0 + 1e-7, 1.0 - 1e-7)
        
        # Apply kappa functions
        kappa0_val = self._kappa0(rho)  # For NTK derivative term
        kappa1_val = self._kappa1(rho)  # For covariance
        
        # Scale kappa1 by standard deviations to get actual covariance
        Sigma_new = std1 * std2 * kappa1_val
        
        # Apply convolution (sum over filter window)
        # For each position (i,j), sum over neighboring positions (i+a, j+b)
        # This is a 4D convolution over the last 4 spatial dimensions
        
        # Reshape for 2D convolution: treat [H,W,H,W] as 2 sets of 2D
        # We need to convolve over both (i1,j1) and (i2,j2) dimensions
        
        # Create convolution kernel (all ones for sum)
        conv_kernel = torch.ones(1, 1, k, k, device=self.device) / (k * k)
        
        # Reshape Sigma_new for convolution
        # [N1, N2, H, W, H, W] -> [N1*N2*H*W, 1, H, W]
        Sigma_reshaped = Sigma_new.reshape(N1 * N2 * H * W, 1, H, W)
        Sigma_conv = F.conv2d(Sigma_reshaped, conv_kernel, padding=pad)
        Sigma_conv = Sigma_conv.reshape(N1, N2, H, W, H, W)
        
        # Convolve over the first spatial pair (i1, j1) as well
        Sigma_conv = Sigma_conv.permute(0, 1, 4, 5, 2, 3)  # [N1, N2, H, W, H, W]
        Sigma_reshaped = Sigma_conv.reshape(N1 * N2 * H * W, 1, H, W)
        Sigma_conv = F.conv2d(Sigma_reshaped, conv_kernel, padding=pad)
        Sigma_conv = Sigma_conv.reshape(N1, N2, H, W, H, W)
        Sigma_conv = Sigma_conv.permute(0, 1, 4, 5, 2, 3)  # Back to [N1, N2, H, W, H, W]
        
        # Scale by filter size (already normalized by k*k in kernel)
        Sigma_new = Sigma_conv * (k * k)
        
        # Update Theta (NTK)
        # Theta^(l+1) = Sigma^(l+1) + Theta^(l) * kappa0
        Theta_scaled = Theta * kappa0_val
        
        # Apply same convolution to Theta
        Theta_reshaped = Theta_scaled.reshape(N1 * N2 * H * W, 1, H, W)
        Theta_conv = F.conv2d(Theta_reshaped, conv_kernel, padding=pad)
        Theta_conv = Theta_conv.reshape(N1, N2, H, W, H, W)
        
        Theta_conv = Theta_conv.permute(0, 1, 4, 5, 2, 3)
        Theta_reshaped = Theta_conv.reshape(N1 * N2 * H * W, 1, H, W)
        Theta_conv = F.conv2d(Theta_reshaped, conv_kernel, padding=pad)
        Theta_conv = Theta_conv.reshape(N1, N2, H, W, H, W)
        Theta_conv = Theta_conv.permute(0, 1, 4, 5, 2, 3)
        
        Theta_new = Sigma_new + Theta_conv * (k * k)
        
        return Sigma_new, Theta_new
    
    def compute_kernel(self, X1: torch.Tensor, X2: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute the full CNTK kernel matrix.
        
        Args:
            X1: First set of images [N1, C, H, W]
            X2: Second set of images [N2, C, H, W] (default: X1)
        
        Returns:
            Kernel matrix [N1, N2]
        """
        if X2 is None:
            X2 = X1
        
        X1 = X1.to(self.device)
        X2 = X2.to(self.device)
        
        # Compute base kernel
        Sigma, Theta = self._compute_base_kernel(X1, X2)
        
        # Apply convolutional layers
        for l in range(self.depth):
            Sigma, Theta = self._apply_conv_layer(Sigma, Theta)
        
        # Final aggregation
        # Theta has shape [N1, N2, H, W, H, W]
        if self.use_gap:
            # Global Average Pooling: mean over all spatial dimensions
            K = Theta.mean(dim=(2, 3, 4, 5))  # [N1, N2]
        else:
            # Vanilla: sum over spatial dimensions (or just diagonal)
            # Sum over matching positions only
            N1, N2, H, W, _, _ = Theta.shape
            # Take diagonal: Theta[..., i, j, i, j]
            K = torch.zeros(N1, N2, device=self.device)
            for i in range(H):
                for j in range(W):
                    K += Theta[:, :, i, j, i, j]
        
        return K
    

class CNTKEfficient:
    """
    Efficient CNTK implementation using vectorized operations.
    
    This implementation follows the paper more closely and is more memory-efficient
    by computing kernels in a pairwise manner.
    """
    
    def __init__(self, depth: int, use_gap: bool = True):
        """
        Args:
            depth: Number of convolutional layers
            use_gap: Use Global Average Pooling (True) or Vanilla (False)
        """
        self.depth = depth
        self.use_gap = use_gap
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @staticmethod
    def _kappa0(rho: torch.Tensor) -> torch.Tensor:
        """E[sigma'(u) * sigma'(v)] for ReLU."""
        rho = torch.clamp(rho, -1.0 + 1e-7, 1.0 - 1e-7)
        return (np.pi - torch.arccos(rho)) / (2 * np.pi)
    
    @staticmethod
    def _kappa1(rho: torch.Tensor) -> torch.Tensor:
        """E[sigma(u) * sigma(v)] for ReLU (normalized)."""
        rho = torch.clamp(rho, -1.0 + 1e-7, 1.0 - 1e-7)
        angle = torch.arccos(rho)
        return (rho * (np.pi - angle) + torch.sqrt(1 - rho**2)) / (2 * np.pi)
    
    def _compute_kernel_pair(self, x1: torch.Tensor, x2: torch.Tensor) -> float:
        """
        Compute kernel between two images.
        
        Args:
            x1: First image [C, H, W]
            x2: Second image [C, H, W]
        
        Returns:
            Kernel value (scalar)
        """
        C, H, W = x1.shape
        P = H * W  # Number of patches/positions
        
        # Flatten to [C, P]
        x1_flat = x1.reshape(C, P)
        x2_flat = x2.reshape(C, P)
        
        # Base case: Sigma^(0) and Theta^(0)
        # Sigma[i,j] = <x1[:, i], x2[:, j]> / C
        Sigma = (x1_flat.T @ x2_flat) / C  # [P, P]
        
        # Self-variances for normalization
        var1 = (x1_flat ** 2).sum(dim=0) / C  # [P]
        var2 = (x2_flat ** 2).sum(dim=0) / C  # [P]
        
        Theta = Sigma.clone()
        
        # Layer-wise recursion
        for l in range(self.depth):
            # Compute standard deviations
            std1 = torch.sqrt(var1 + 1e-8)
            std2 = torch.sqrt(var2 + 1e-8)
            
            # Correlation matrix
            rho = Sigma / (std1.unsqueeze(1) * std2.unsqueeze(0) + 1e-8)
            rho = torch.clamp(rho, -1.0 + 1e-7, 1.0 - 1e-7)
            
            # Apply kappa functions
            k0 = self._kappa0(rho)
            k1 = self._kappa1(rho)
            
            # New covariance
            Sigma_new = std1.unsqueeze(1) * std2.unsqueeze(0) * k1
            
            # New NTK: Theta = Sigma + Theta * k0
            Theta_new = Sigma_new + Theta * k0
            
            # Update variances (diagonal of covariance)
            var1 = torch.diag(Sigma_new) if Sigma.shape[0] == Sigma.shape[1] else var1 * self._kappa1(torch.ones_like(var1))
            var2 = var1  # Same for self-kernel
            
            Sigma = Sigma_new
            Theta = Theta_new
        
        # Final aggregation
        if self.use_gap:
            # Mean over all patch pairs (GAP)
            return Theta.mean().item()
        else:
            # Sum of diagonal (Vanilla)
            return torch.diag(Theta).sum().item()
    
    def compute_kernel(self, X: torch.Tensor, Y: Optional[torch.Tensor] = None, 
                      batch_size: int = 100) -> torch.Tensor:
        """
        Compute full kernel matrix.
        
        Args:
            X: Images [N1, C, H, W]
            Y: Images [N2, C, H, W] or None for self-kernel
            batch_size: Not used in pairwise computation
        
        Returns:
            Kernel matrix [N1, N2]
        """
        if Y is None:
            Y = X
        
        X = X.to(self.device)
        Y = Y.to(self.device)
        
        N1, N2 = X.shape[0], Y.shape[0]
        K = torch.zeros(N1, N2, device=self.device)
        
        for i in tqdm(range(N1), desc="Computing kernel rows"):
            for j in range(N2):
                K[i, j] = self._compute_kernel_pair(X[i], Y[j])
        
        return K
    
>>>>>>> 22d1f47 (coded cntk + library, added implementation and analysis)
=======
>>>>>>> 37ce088 (Updated model)

class CNTKVectorized:
    """
    Memory-efficient CNTK implementation.
<<<<<<< HEAD
<<<<<<< HEAD
=======
    
    Instead of storing full [N1, N2, P, P] tensors, this implementation
    computes the kernel pairwise and aggregates directly.
>>>>>>> 22d1f47 (coded cntk + library, added implementation and analysis)
=======
>>>>>>> 37ce088 (Updated model)
    """
    
    def __init__(self, depth: int, use_gap: bool = True):
        self.depth = depth
        self.use_gap = use_gap
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @staticmethod
    def _kappa0(rho: torch.Tensor) -> torch.Tensor:
        """Derivative covariance for ReLU: E[sigma'(u) * sigma'(v)]."""
        rho = torch.clamp(rho, -1.0 + 1e-7, 1.0 - 1e-7)
        return (np.pi - torch.arccos(rho)) / (2 * np.pi)
    
    @staticmethod
    def _kappa1(rho: torch.Tensor) -> torch.Tensor:
        """Activation covariance for ReLU (normalized): E[sigma(u) * sigma(v)]."""
        rho = torch.clamp(rho, -1.0 + 1e-7, 1.0 - 1e-7)
        angle = torch.arccos(rho)
        return (rho * (np.pi - angle) + torch.sqrt(1 - rho**2)) / (2 * np.pi)
    
    def _compute_single_kernel(self, x1: torch.Tensor, x2: torch.Tensor) -> float:
        """
        Compute kernel value between two images.
        
        Args:
            x1: First image [C, H, W]
            x2: Second image [C, H, W]
        
        Returns:
            Kernel value (scalar)
        """
        C, H, W = x1.shape
        
        # Extract patches using unfold
        unfold = torch.nn.Unfold(kernel_size=3, padding=1)
        patches1 = unfold(x1.unsqueeze(0)).squeeze(0)
        patches2 = unfold(x2.unsqueeze(0)).squeeze(0)

        # Normalize patches
        dim_norm = patches1.shape[0]
        Sigma = (patches1.T @ patches2) / dim_norm

        # Self-variances
        var1 = (patches1 ** 2).sum(dim=0) / dim_norm
        var2 = (patches2 ** 2).sum(dim=0) / dim_norm

        Sigma = Sigma.reshape(H, W, H, W)

        if self.use_gap:
            Theta = torch.zeros_like(Sigma)
        else:
            Theta = Sigma.clone()
        
        pad = 1

        # Layer recursion
        for l in range(self.depth):
            v1_flat = var1.flatten()
            v2_flat = var2.flatten()
            std1 = torch.sqrt(v1_flat + 1e-8) 
            std2 = torch.sqrt(v2_flat + 1e-8) 

            Sigma_flat = Sigma.reshape(H * W, H * W)
            
            # Correlation
            std_outer = std1.unsqueeze(1) * std2.unsqueeze(0)
            rho = Sigma_flat / (std_outer + 1e-8)
            
            # Apply kappa functions
            k0 = self._kappa0(rho)
            k1 = self._kappa1(rho)
            
            # Calculate pointwise values
            Sigma_pointwise = (std_outer * k1 * 2.0).reshape(H, W, H, W)
            Theta_pointwise = (Theta.reshape(H*W, H*W) * k0).reshape(H, W, H, W)
            
            # Diagonal convolution
            Sigma_next = torch.zeros_like(Sigma_pointwise)
            Theta_conv = torch.zeros_like(Theta_pointwise)

            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    # Source slices
                    src_h_start = max(0, -di)
                    src_h_end = H + min(0, -di)
                    src_w_start = max(0, -dj)
                    src_w_end = W + min(0, -dj)

                    # Destination slices
                    dst_h_start = max(0, di)
                    dst_h_end = H + min(0, di)
                    dst_w_start = max(0, dj)
                    dst_w_end = W + min(0, dj)

                    Sigma_next[dst_h_start:dst_h_end, dst_w_start:dst_w_end,
                                 dst_h_start:dst_h_end, dst_w_start:dst_w_end] += \
                        Sigma_pointwise[src_h_start:src_h_end, src_w_start:src_w_end,
                                        src_h_start:src_h_end, src_w_start:src_w_end]
                    
                    Theta_conv[dst_h_start:dst_h_end, dst_w_start:dst_w_end,
                                dst_h_start:dst_h_end, dst_w_start:dst_w_end] += \
                        Theta_pointwise[src_h_start:src_h_end, src_w_start:src_w_end,
                                        src_h_start:src_h_end, src_w_start:src_w_end]
                
            # Normalize
            Sigma_next = Sigma_next / 9.0
            Theta_conv = Theta_conv / 9.0

            # Update Theta
            if self.use_gap and l == self.depth - 1:
                Theta = Theta_conv
            else:
                Theta= Sigma_next + Theta_conv
            
            Sigma = Sigma_next
            
            # Update variances
            var1_grid = var1.reshape(1, 1, H, W)
            var2_grid = var2.reshape(1, 1, H, W)

            pool = torch.nn.AvgPool2d(kernel_size=3, stride=1, padding=1)

            var1 = pool(var1_grid).flatten()
            var2 = pool(var2_grid).flatten()
        
        # Final aggregation
        if self.use_gap:
            # GAP: mean over all patches
            return Theta.mean().item()
        else:
            # Vanilla: sum of diagonal elements
            return torch.diag(Theta.reshape(H * W, H * W)).sum().item()
    
    def compute_kernel(self, X: torch.Tensor, Y: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute kernel matrix.
        
        Args:
            X: Images [N1, C, H, W]
            Y: Images [N2, C, H, W] or None for self-kernel
        
        Returns:
            Kernel matrix [N1, N2]
        """
        if Y is None:
            Y = X
            symmetric = True
        else:
            symmetric = False
        
        X = X.to(self.device)
        Y = Y.to(self.device)
        
        N1, N2 = X.shape[0], Y.shape[0]
        K = torch.zeros(N1, N2, device=self.device)
        
        total = N1 * N2 if not symmetric else N1 * (N1 + 1) // 2
        pbar = tqdm(total=total, desc="Computing kernel")
        
        for i in range(N1):
            j_start = i if symmetric else 0
            for j in range(j_start, N2):
                val = self._compute_single_kernel(X[i], Y[j])
                K[i, j] = val
                if symmetric and i != j:
                    K[j, i] = val
                pbar.update(1)
        
        pbar.close()
        return K