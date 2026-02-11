import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """(Conv3D -> BN -> ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet3D(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet3D, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # --- ENCODER (Down) ---
        # We start with fewer filters (16) to save VRAM. 
        # Standard U-Net starts with 64, but that melts GPUs in 3D.
        self.inc = DoubleConv(n_channels, 16)
        self.down1 = nn.Sequential(nn.MaxPool3d(2), DoubleConv(16, 32))
        self.down2 = nn.Sequential(nn.MaxPool3d(2), DoubleConv(32, 64))
        self.down3 = nn.Sequential(nn.MaxPool3d(2), DoubleConv(64, 128))
        
        # --- BRIDGE (Bottom) ---
        self.down4 = nn.Sequential(nn.MaxPool3d(2), DoubleConv(128, 256))

        # --- DECODER (Up) ---
        # We use Transposed Convolution to "upscale" the cube
        self.up1 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(256, 128) # 256 because we concat skip connection (128+128)
        
        self.up2 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(128, 64)
        
        self.up3 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(64, 32)
        
        self.up4 = nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(32, 16)

        # --- OUTPUT ---
        self.outc = nn.Conv3d(16, n_classes, kernel_size=1)

    def forward(self, x):
        # Down
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Up (with Skip Connections)
        x = self.up1(x5)
        # Note: If your input size isn't divisible by 16 (e.g. 128 is fine), 
        # you might need padding here. For 128^3, this aligns perfectly.
        x = torch.cat([x4, x], dim=1) 
        x = self.conv1(x)

        x = self.up2(x)
        x = torch.cat([x3, x], dim=1)
        x = self.conv2(x)

        x = self.up3(x)
        x = torch.cat([x2, x], dim=1)
        x = self.conv3(x)

        x = self.up4(x)
        x = torch.cat([x1, x], dim=1)
        x = self.conv4(x)
        
        logits = self.outc(x)
        return logits

# Quick Test to verify it works
if __name__ == "__main__":
    # Create a random 3D tensor: (Batch=1, Channel=1, Depth=128, Height=128, Width=128)
    dummy_input = torch.randn(1, 1, 128, 128, 128)
    model = UNet3D(n_channels=1, n_classes=2)
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    # Expected Output: [1, 2, 128, 128, 128]
