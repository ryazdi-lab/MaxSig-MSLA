"""
Fixed / corrected version of MaxSigLayerNormalized.

Changes from the original:
1. BUG FIX: the first `out = weight_center*center + weight_median*median - mean_stat`
   was being silently overwritten by the morphological branch before it ever
   reached `proj`. weight, bias, weight_center, weight_median, mean_stat,
   median_stat were dead code (unused params + wasted compute). Now both
   branches are actually combined and contribute a gradient.
2. "Soft dilation / erosion" is now a real soft-max / soft-min via a
   learnable-temperature LogSumExp, instead of a fixed ~box-filter average
   masquerading as "soft max". This matches the morphological framing the
   comments claimed.
3. Added GroupNorm + activation after the 1x1 projection, since dropping a
   bare Conv2d(bias=False) with no normalization into a U-Net (where every
   other block has BN/GN + ReLU) tends to destabilize training.
4. Proper (Kaiming-style) init instead of torch.rand(...) for weight/bias.
5. A plain DepthwiseSeparableConvBlock baseline is included so you can run
   a clean ablation: does the "morphological" framing actually buy you
   anything over a standard learnable filter of the same kernel size?
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaxSigLayerFixed(nn.Module):
    """
    Local statistics + soft-morphology block with proper C_in -> C_out projection.

    Args:
        ch_in: input channels
        ch_out: output channels
        kernel_size: odd recommended
        stride: keep 1 to preserve H, W
        use_sigmoid_input: apply sigmoid(x) before unfolding
        norm_groups: number of GroupNorm groups applied after projection
    """

    def __init__(
        self,
        ch_in: int,
        ch_out: int,
        kernel_size: int,
        stride: int = 1,
        use_sigmoid_input: bool = False,
        norm_groups: int = 8,
    ):
        super().__init__()
        assert kernel_size >= 1
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.kernel_size = kernel_size
        self.stride = stride
        self.use_sigmoid_input = use_sigmoid_input
        self.padding = (kernel_size - 1) // 2
        self.kernel_area = kernel_size ** 2
        self.center_index = self.kernel_area // 2

        # patch weight / bias (shared across channels), now actually used
        self.weight = nn.Parameter(torch.zeros(1, 1, self.kernel_area, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, 1, self.kernel_area, 1, 1))

        # learnable temperature for soft-max / soft-min (per input channel)
        # softplus keeps it positive; init -> temperature ~= 1
        self.log_temp = nn.Parameter(torch.zeros(1, ch_in, 1, 1, 1))

        # learnable scalars combining the two statistic branches
        self.weight_center = nn.Parameter(torch.tensor(1.0))
        self.weight_median = nn.Parameter(torch.tensor(0.5))
        self.weight_mean = nn.Parameter(torch.tensor(0.5))
        self.weight_morph = nn.Parameter(torch.tensor(1.0))

        # projection + normalization + activation (fixes "bare conv" issue)
        self.proj = nn.Conv2d(ch_in, ch_out, kernel_size=1, bias=False)
        groups = norm_groups if ch_out % norm_groups == 0 else 1
        self.norm = nn.GroupNorm(groups, ch_out)
        self.act = nn.GELU()

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.trunc_normal_(self.weight, std=0.02)
        nn.init.zeros_(self.bias)
        nn.init.kaiming_normal_(
            self.proj.weight, mode="fan_out", nonlinearity="relu"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C_in, H, W = x.shape
        assert C_in == self.ch_in, f"Expected {self.ch_in} channels, got {C_in}"
        x_in = torch.sigmoid(x) if self.use_sigmoid_input else x

        x_unfold = F.unfold(
            x_in, kernel_size=self.kernel_size, padding=self.padding, stride=self.stride
        )  # (B, C_in*K*K, L)
        _, CK2, L = x_unfold.shape
        assert CK2 == C_in * self.kernel_area

        out_h = (H + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_w = (W + 2 * self.padding - self.kernel_size) // self.stride + 1
        x_unfold = x_unfold.view(B, C_in, self.kernel_area, out_h, out_w)

        center_values = x_unfold[:, :, self.center_index, :, :]  # (B,C,H,W)
        median_stat, _ = torch.median(x_unfold, dim=2)           # (B,C,H,W)

        w = torch.sigmoid(self.weight)
        b = torch.sigmoid(self.bias)
        D = torch.maximum(w, x_unfold) + b
        mean_stat = torch.mean(D, dim=2)                         # (B,C,H,W)

        # --- real soft-max / soft-min via temperature-scaled LogSumExp ---
        temp = F.softplus(self.log_temp) + 1e-3                  # (1,C,1,1,1), >0
        dilation = torch.logsumexp(x_unfold * temp, dim=2) / temp.squeeze(2)   # soft max
        erosion = -torch.logsumexp(-x_unfold * temp, dim=2) / temp.squeeze(2)  # soft min
        morph_grad = dilation - erosion                          # local contrast / edges

        # combine both branches (previously the first branch was discarded)
        stat_branch = (
            self.weight_center * center_values
            + self.weight_median * median_stat
            + self.weight_mean * mean_stat
        )
        out = stat_branch + self.weight_morph * morph_grad

        out = self.proj(out)
        out = self.norm(out)
        out = self.act(out)
        return out


class DepthwiseSeparableConvBlock(nn.Module):
    """
    Plain baseline for ablation: depthwise conv (same kernel size) + pointwise
    projection + GroupNorm + activation. If MaxSigLayerFixed doesn't clearly
    beat this on Dice/IoU, the morphological framing isn't earning its
    complexity/compute cost.
    """

    def __init__(self, ch_in: int, ch_out: int, kernel_size: int, norm_groups: int = 8):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.depthwise = nn.Conv2d(
            ch_in, ch_in, kernel_size, padding=padding, groups=ch_in, bias=False
        )
        self.pointwise = nn.Conv2d(ch_in, ch_out, kernel_size=1, bias=False)
        groups = norm_groups if ch_out % norm_groups == 0 else 1
        self.norm = nn.GroupNorm(groups, ch_out)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.norm(x)
        x = self.act(x)
        return x


if __name__ == "__main__":
    # quick sanity check: shapes match and gradients flow through every param
    x = torch.randn(2, 16, 32, 32, requires_grad=True)

    layer = MaxSigLayerFixed(ch_in=16, ch_out=32, kernel_size=3)
    out = layer(x)
    assert out.shape == (2, 32, 32, 32), out.shape
    out.sum().backward()

    dead = [n for n, p in layer.named_parameters() if p.grad is None]
    assert not dead, f"Parameters with no gradient (still dead code): {dead}"
    print("MaxSigLayerFixed: shape OK, all parameters receive gradients.")

    baseline = DepthwiseSeparableConvBlock(ch_in=16, ch_out=32, kernel_size=3)
    out_b = baseline(x)
    assert out_b.shape == (2, 32, 32, 32)
    print("DepthwiseSeparableConvBlock (ablation baseline): shape OK.")
