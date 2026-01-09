import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================
# 工具：Window 分块与合并
# =============================
def window_partition(x, window_size):
    # x: B, C, H, W
    B, C, H, W = x.shape
    x = x.reshape(B, C, H // window_size, window_size, W // window_size, window_size)
    # print(x.shape)
    x = x.permute(0, 2, 4, 3, 5, 1).reshape(-1, window_size * window_size, C)
    return x   # (num_windows*B, WS*WS, C)

def window_reverse(windows, window_size, H, W, C):
    B = windows.shape[0] // (H // window_size * W // window_size)
    x = windows.reshape(B, H // window_size, W // window_size, window_size, window_size, C)
    x = x.permute(0, 5, 1, 3, 2, 4).reshape(B, C, H, W)
    return x

# =============================
# Multi-Head Self Attention
# =============================
class WindowAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        Bwin, N, C = x.shape
        head_dim = C // self.num_heads
        scale = head_dim ** -0.5

        # (Bwin, N, 3*C) -> (3, Bwin, num_heads, N, head_dim)
        qkv = (
            self.qkv(x)
            .reshape(Bwin, N, 3, self.num_heads, head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]      # (Bwin, num_heads, N, head_dim)

        att = (q @ k.transpose(-2, -1)) * scale      # (Bwin, num_heads, N, N)
        att = att.softmax(dim=-1)

        x = (att @ v)                                # (Bwin, num_heads, N, head_dim)
        x = x.transpose(1, 2).reshape(Bwin, N, C)    # (Bwin, N, C)

        return self.proj(x)

class SwinBasicBlock(nn.Module):
    def __init__(self, dim, window_size=8, num_heads=4, shift=False, mlp_ratio=4):
        super().__init__()
        self.window_size = window_size
        self.shift = shift
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)

        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim)
        )

    def forward(self, x):
        """
        x: B,C,H,W
        """
        B, C, H, W = x.shape
        shortcut = x

        # ---- LayerNorm & BCHW → BHWC ----
        x = x.permute(0, 2, 3, 1)  # BHWC
        x = self.norm1(x).permute(0, 3, 1, 2)  # BCHW

        # ---- Optional Shift ----
        if self.shift:
            x = torch.roll(x, shifts=(-(self.window_size // 2),
                                      -(self.window_size // 2)), dims=(2, 3))

        # ---- Window Partition ----
        x_windows = window_partition(x, self.window_size)  # (B*nW, WS*WS, C)

        # ---- Attention ----
        x_windows = self.attn(x_windows)

        # ---- Window Reverse ----
        x = window_reverse(x_windows, self.window_size, H, W, C)

        # ---- Reverse Shift ----
        if self.shift:
            x = torch.roll(x, shifts=(self.window_size // 2,
                                      self.window_size // 2), dims=(2, 3))

        # ---- Residual ----
        x = x + shortcut

        # ---- MLP ----
        shortcut = x
        x2 = x.permute(0, 2, 3, 1)  # BCHW → BHWC
        x2 = self.norm2(x2)
        x2 = self.mlp(x2)
        x2 = x2.permute(0, 3, 1, 2)

        return x + shortcut

class SwinBlockPair(nn.Module):
    def __init__(self, dim, window_size=8, num_heads=4, mlp_ratio=4):
        super().__init__()

        self.block1 = SwinBasicBlock(
            dim, window_size, num_heads,
            shift=False,
            mlp_ratio=mlp_ratio
        )

        self.block2 = SwinBasicBlock(
            dim, window_size, num_heads,
            shift=True,
            mlp_ratio=mlp_ratio
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        return x

# =============================
# Swin-like Block for CNN
# =============================
class SwinBlockPairConv(nn.Module):
    def __init__(self, in_ch, out_ch, window_size=8, num_heads=4, mlp_ratio=4, downsample=False):
        super().__init__()

        # patch embedding（可以当成conv替代）
        self.proj = nn.Conv2d(in_ch, out_ch, 1)

        # Swin block pair (shift=False + shift=True)
        self.block1 = SwinBasicBlock(out_ch, window_size, num_heads, shift=False, mlp_ratio=mlp_ratio)
        self.block2 = SwinBasicBlock(out_ch, window_size, num_heads, shift=True,  mlp_ratio=mlp_ratio)

        # optional downsample (stride=2)
        self.down = nn.Conv2d(out_ch, out_ch, 2, stride=2) if downsample else None

    def forward(self, x):
        x = self.proj(x)      # channel projection (conv-based embedding)
        x = self.block1(x)    # window attention
        x = self.block2(x)    # shifted window attention
        if self.down is not None:
            x = self.down(x)  # optional downsample
        return x


class SwinUNetLike(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        # 编码器：只做通道变化 + SwinBlockPair，不在这里下采样
        self.enc1 = SwinBlockPairConv(in_channels, 64,  window_size=8, num_heads=4, downsample=False)
        self.enc2 = SwinBlockPairConv(64,         128, window_size=8, num_heads=4, downsample=False)
        self.enc3 = SwinBlockPairConv(128,        256, window_size=8, num_heads=8, downsample=False)

        # 仍然用原来的 pooling
        self.pool = nn.MaxPool2d(2)

        # 解码器：concat 之后再用 SwinBlockPairConv 压回目标通道数
        self.up2  = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = SwinBlockPairConv(256, 128, window_size=8, num_heads=4, downsample=False)

        self.up1  = nn.ConvTranspose2d(128, 64,  2, stride=2)
        self.dec1 = SwinBlockPairConv(128, 64,  window_size=8, num_heads=4, downsample=False)


        self.final = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        pad_h = (64 - H % 64) % 64
        pad_w = (64 - W % 64) % 64

        if pad_h != 0 or pad_w != 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))  # (left, right, top, bottom)

        e1 = self.enc1(x)               # (B, 64,  H,   W)
        e2 = self.enc2(self.pool(e1))   # (B, 128, H/2, W/2)
        e3 = self.enc3(self.pool(e2))   # (B, 256, H/4, W/4)

        d2 = self.up2(e3)               # (B, 128, H/2, W/2)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)               # (B, 64, H, W)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        out= self.final(d1)
        if pad_h != 0 or pad_w != 0:
            out = out[:, :, :H, :W]

        return out


