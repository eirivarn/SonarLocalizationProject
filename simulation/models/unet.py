"""U-Net model with GroupNorm for heatmap-based net detection."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(conv => GroupNorm => ReLU) * 2"""
    
    def __init__(self, in_channels, out_channels, num_groups=8):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(min(num_groups, out_channels), out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(min(num_groups, out_channels), out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    
    def __init__(self, in_channels, out_channels, num_groups=8):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, num_groups)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""
    
    def __init__(self, in_channels, out_channels, num_groups=8):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels, num_groups)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Handle size mismatch due to padding
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """U-Net with multiple output heads for heatmap and direction prediction."""
    
    def __init__(self, n_channels=4, num_groups=8, base_channels=32):
        """
        Args:
            n_channels: Number of input channels (4 for image + mask + y_map + z_map)
            num_groups: Number of groups for GroupNorm
            base_channels: Base number of channels (will be multiplied at each level)
        """
        super().__init__()
        self.n_channels = n_channels
        
        # Encoder
        self.inc = DoubleConv(n_channels, base_channels, num_groups)
        self.down1 = Down(base_channels, base_channels * 2, num_groups)
        self.down2 = Down(base_channels * 2, base_channels * 4, num_groups)
        self.down3 = Down(base_channels * 4, base_channels * 8, num_groups)
        self.down4 = Down(base_channels * 8, base_channels * 16, num_groups)
        
        # Decoder
        self.up1 = Up(base_channels * 16, base_channels * 8, num_groups)
        self.up2 = Up(base_channels * 8, base_channels * 4, num_groups)
        self.up3 = Up(base_channels * 4, base_channels * 2, num_groups)
        self.up4 = Up(base_channels * 2, base_channels, num_groups)
        
        # Output heads
        self.heatmap_head = nn.Conv2d(base_channels, 1, kernel_size=1)
        self.direction_head = nn.Conv2d(base_channels, 2, kernel_size=1)
        self.visibility_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base_channels, 1)
        )
    
    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # Output heads
        heatmap = self.heatmap_head(x)
        direction = self.direction_head(x)
        visibility = self.visibility_head(x)
        
        return {
            'heatmap': heatmap,       # (B, 1, H, W)
            'direction': direction,    # (B, 2, H, W)
            'visibility': visibility   # (B, 1)
        }


def test_model():
    """Test model forward pass."""
    model = UNet(n_channels=4, num_groups=8, base_channels=32)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model: U-Net")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    batch_size = 2
    x = torch.randn(batch_size, 4, 512, 512)
    
    with torch.no_grad():
        outputs = model(x)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Heatmap output shape: {outputs['heatmap'].shape}")
    print(f"Direction output shape: {outputs['direction'].shape}")
    print(f"Visibility output shape: {outputs['visibility'].shape}")
    
    print("\n✓ Model test passed!")


if __name__ == '__main__':
    test_model()
