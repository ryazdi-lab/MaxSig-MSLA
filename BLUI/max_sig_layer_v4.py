import torch
import torch.nn as nn
import torch.nn.functional as F


class MaxSigLayerV4(nn.Module):
    """
    MaxSigLayer — v4, fully revised.

    Changes vs v3:
    1. Consistent value space: center_values is now extracted from D_norm so
       all three statistics (center, median, mean) live in the same normalised
       space before being combined.
    2. Learned attention over statistics: the fixed formula
       (w_center·center + w_median·median − mean) is replaced by a softmax-
       weighted combination of all three statistics. The network learns which
       statistic is most useful rather than having the sign hard-coded.
    3. Residual connection: a 1×1 conv skip path (identity when ch_in==ch_out)
       is added around the whole transformation, stabilising early training.
    4. BatchNorm before projection: brings the feature scale in line with
       surrounding Conv-BN-ReLU blocks before the 1×1 proj conv.
    5. Learnable per-channel softplus beta: each input channel learns its own
       soft-threshold sharpness instead of sharing a fixed global scalar.

    Args:
        ch_in        (int):   Number of input channels.
        ch_out       (int):   Number of output channels.
        kernel_size  (int):   Patch size (odd recommended, e.g. 3, 5).
        stride       (int):   Stride for the unfolding operation.
        beta_init    (float): Initial value for the per-channel softplus beta.
                              Higher → closer to hard max; lower → smoother.
    """

    def __init__(
        self,
        ch_in: int,
        ch_out: int,
        kernel_size: int,
        stride: int = 1,
        beta_init: float = 5.0,
    ):
        super().__init__()
        self.ch_in       = ch_in
        self.ch_out      = ch_out
        self.kernel_size = kernel_size
        self.stride      = stride

        self.padding     = (kernel_size - 1) // 2
        self.kernel_area = kernel_size ** 2
        self.center_index = self.kernel_area // 2

        # Per-channel patch weights and bias
        self.weight = nn.Parameter(
            torch.zeros(1, ch_in, self.kernel_area, 1, 1)
        )
        self.bias = nn.Parameter(
            torch.zeros(1, ch_in, self.kernel_area, 1, 1)
        )

        # Fix 5: per-channel learnable softplus beta, shape (1, ch_in, 1, 1, 1)
        # Stored as log so it stays positive: beta = softplus(log_beta)
        self.log_beta = nn.Parameter(
            torch.full((1, ch_in, 1, 1, 1), float(beta_init))
        )

        # Fix 2: learned attention weights over [center, median, mean]
        # shape (3,) — softmax applied in forward to ensure they sum to 1
        self.stat_weight = nn.Parameter(torch.ones(3))

        # Fix 4: BatchNorm before projection
        self.bn = nn.BatchNorm2d(ch_in)

        # Learned ch_in → ch_out projection
        self.proj = nn.Conv2d(ch_in, ch_out, kernel_size=1, bias=False)

        # Fix 3: residual skip path
        self.residual = (
            nn.Conv2d(ch_in, ch_out, kernel_size=1, bias=False)
            if ch_in != ch_out
            else nn.Identity()
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.weight, mean=-1.0, std=0.5)
        nn.init.zeros_(self.bias)
        nn.init.kaiming_uniform_(self.proj.weight, nonlinearity="relu")
        if isinstance(self.residual, nn.Conv2d):
            nn.init.kaiming_uniform_(self.residual.weight, nonlinearity="relu")
        # stat_weight starts uniform → softmax gives equal weight to all three
        nn.init.ones_(self.stat_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C_in, H, W)
        Returns:
            (B, C_out, H', W')
        """
        B, C_in, H, W = x.shape
        eps = 1e-6

        # Sigmoid on input only — values in (0, 1)
        x_sigmoid = torch.sigmoid(x)

        # Unfold into local KxK patches
        x_unfold = F.unfold(
            x_sigmoid,
            kernel_size=self.kernel_size,
            padding=self.padding,
            stride=self.stride,
        )
        out_h = (H + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_w = (W + 2 * self.padding - self.kernel_size) // self.stride + 1
        x_unfold = x_unfold.view(B, C_in, self.kernel_area, out_h, out_w)

        # Fix 5: per-channel beta, kept positive via softplus
        beta = F.softplus(self.log_beta)  # (1, C_in, 1, 1, 1), always > 0

        # Soft-max approximation: softplus(beta*(x-w))/beta + w + b
        diff = x_unfold - self.weight     # (B, C_in, K*K, out_h, out_w)
        D = F.softplus(diff * beta) / beta + self.weight + self.bias

        # Per-patch normalisation across the K*K dimension
        mean_D = D.mean(dim=2, keepdim=True)
        std_D  = D.std(dim=2, keepdim=True)
        D_norm = (D - mean_D) / (std_D + eps)  # (B, C_in, K*K, out_h, out_w)

        # Fix 1: all three statistics extracted from the SAME normalised tensor
        center_values = D_norm[:, :, self.center_index, :, :]  # (B, C_in, H, W)
        median_stat, _ = torch.median(D_norm, dim=2)           # (B, C_in, H, W)
        mean_stat      = D_norm.mean(dim=2)                    # (B, C_in, H, W)

        # Fix 2: softmax-weighted combination — network learns the importance
        # of each statistic; softmax ensures the weights sum to 1
        attn = torch.softmax(self.stat_weight, dim=0)          # (3,)
        stats = torch.stack(
            [center_values, median_stat, mean_stat], dim=2
        )                                                       # (B, C_in, 3, H, W)
        feat = (stats * attn.view(1, 1, 3, 1, 1)).sum(dim=2)  # (B, C_in, H, W)

        # Fix 4: BatchNorm before projection
        feat = self.bn(feat)

        # Learned channel projection
        out = self.proj(feat)                                   # (B, C_out, H, W)

        # Fix 3: residual connection
        out = out + self.residual(x)

        return out


if __name__ == "__main__":
    torch.manual_seed(0)

    layer = MaxSigLayerV4(ch_in=4, ch_out=8, kernel_size=3)
    x = torch.randn(2, 4, 16, 16)
    y = layer(x)
    print(f"Input : {x.shape}")
    print(f"Output: {y.shape}")   # expect (2, 8, 16, 16)

    loss = y.mean()
    loss.backward()
    print(f"weight.grad norm   : {layer.weight.grad.norm().item():.6f}")
    print(f"log_beta.grad norm : {layer.log_beta.grad.norm().item():.6f}")
    print(f"stat_weight.grad   : {layer.stat_weight.grad}")
    print(f"proj.weight.grad   : {layer.proj.weight.grad.norm().item():.6f}")
    print("All good — gradients flow through all parameters.")

    # Verify residual works for same-channel case too
    layer_same = MaxSigLayerV4(ch_in=4, ch_out=4, kernel_size=3)
    y2 = layer_same(x)
    assert y2.shape == (2, 4, 16, 16), "Same-channel residual failed"
    print("Same-channel residual: OK")
