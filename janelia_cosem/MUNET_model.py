import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionGate(nn.Module):
    """
    g: gating（来自 decoder 的上采样特征）
    x: skip（来自 encoder 的特征）
    F_g: gating 通道数
    F_l: skip 通道数
    F_int: 中间通道数（压缩比例）
    """

    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, bias=False),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, bias=False),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, bias=True),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # 将 g 映射到中间通道；必要时插值到 x 的空间分辨率
        g1 = self.W_g(g)
        if g1.shape[-2:] != x.shape[-2:]:
            g1 = F.interpolate(g1, size=x.shape[-2:], mode='bilinear', align_corners=False)

        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)  # 注意力权重 (B,1,H,W)
        out = x * psi  # 对 skip 做加权
        return out

class MultiKernelConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=(3, 5, 7)):
        super().__init__()
        padding = [(k // 2) for k in kernel_sizes]
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=k, padding=p),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ) for k, p in zip(kernel_sizes, padding)
        ])
        self.merge = nn.Sequential(
            nn.Conv2d(out_channels * len(kernel_sizes), out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        branch_outputs = [branch(x) for branch in self.branches]
        concat = torch.cat(branch_outputs, dim=1)
        out = self.merge(concat)
        return out

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)

class MultiKernelUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()

        # ===== Encoder (Multi-kernel, 低分辨率，值得) =====
        self.enc1 = MultiKernelConvBlock(in_channels, 64)
        self.enc2 = MultiKernelConvBlock(64, 128)
        self.enc3 = MultiKernelConvBlock(128, 256)

        self.pool = nn.MaxPool2d(2)

        # ===== Decoder (轻量，省显存) =====
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = ConvBlock(256, 128)   # cat(d2, e2)

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = ConvBlock(128, 64)    # cat(d1, e1)

        self.final = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        # ===== Encoder =====
        e1 = self.enc1(x)                 # (B, 64,  H,   W)
        e2 = self.enc2(self.pool(e1))     # (B, 128, H/2, W/2)
        e3 = self.enc3(self.pool(e2))     # (B, 256, H/4, W/4)

        # ===== Decoder =====
        d2 = self.up2(e3)                 # (B, 128, ~H/2, ~W/2)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)                 # (B, 64, ~H, ~W)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.final(d1)
