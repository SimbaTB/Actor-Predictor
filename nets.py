import torch
import torch.nn as nn
import torch.nn.functional as F
import tools
import math

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, layers_num):
        super().__init__()
        
        net = []
        for i in range(layers_num):
            in_dim = input_dim if i == 0 else hidden_dim
            out_dim = output_dim if i == layers_num-1 else hidden_dim
            net.append(nn.Linear(in_dim, out_dim))
            # 最后一层不加归一化和激活函数
            if i != layers_num - 1:
                net.append(nn.LayerNorm(out_dim))
                net.append(nn.SiLU())

        self.net = nn.Sequential(*net)
    
    def forward(self, x):
        return self.net(x)

class SeperableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=stride, groups=in_channels),
            nn.InstanceNorm2d(in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.InstanceNorm2d(out_channels),
            nn.SiLU()
        )
    
    def forward(self, x):
        return self.layers(x)

# 深度可分离卷积 + 残差连接
class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.layers = nn.Sequential(
            SeperableConv(in_channels, out_channels, stride=stride),
            SeperableConv(out_channels, out_channels, stride=1)
        )
        
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.InstanceNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
        
    def forward(self, x):
        Y = self.layers(x) + self.shortcut(x)
        return Y

# 编码器
class ImageEncoder(nn.Module):
    def __init__(self, channels, hidden_dim):
        super().__init__()
        self.channels = channels
        self.hidden_dim = hidden_dim
        self.layers = nn.Sequential(
            # (B, 3, 64, 64)
            nn.Conv2d(channels, 16, kernel_size=3, padding=1, stride=1),   # (B, 16, 64, 64)     
            Residual(16, 16, stride=2),     # (B, 16, 32, 32)
            Residual(16, 32, stride=2),     # (B, 32, 16, 16)
            Residual(32, 64, stride=2),     # (B, 64, 8, 8)
            Residual(64, 128, stride=2),    # (B, 128, 4, 4)
            nn.Flatten(),                   # (B, 128*16)
            nn.Linear(128*16, hidden_dim),  # (B, hidden_dim)
            nn.LayerNorm(hidden_dim),
            nn.SiLU()
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        assert (C, H, W) == (self.channels, 64, 64)
        x = self.layers(x.reshape(B*T, self.channels, 64, 64))
        return x.reshape(B, T, self.hidden_dim)

# 解码器
class ImageDecoder(nn.Module):
    def __init__(self, channels, hidden_dim):
        super().__init__()
        self.channels = channels
        self.hidden_dim = hidden_dim
        self.layers = nn.Sequential(
            nn.Linear(hidden_dim, 128*16),
            nn.LayerNorm(128*16),
            nn.SiLU(),
            nn.Unflatten(-1, (128, 4, 4)),      # (B, T, 128, 4, 4)
            Residual(128, 256, 1),              # (B, T, 256, 4, 4)
            nn.PixelShuffle(2),                 # (B, T, 64, 8, 8)
            Residual(64, 128, 1),               # (B, T, 128, 8, 8)
            nn.PixelShuffle(2),                 # (B, T, 32, 16, 16)
            Residual(32, 64, 1),                # (B, T, 64, 16, 16)
            nn.PixelShuffle(2),                 # (B, T, 16, 32, 32)
            Residual(16, 32, 1),                # (B, T, 32, 32, 32)
            nn.PixelShuffle(2),                 # (B, T, 8, 64, 64)
            nn.Conv2d(8, channels, kernel_size=3, padding=1, stride=1),    # (B, T, C, 64, 64)                 
            nn.Sigmoid()
        )

    def forward(self, x):
        B, T, _ = x.shape           # 输入形状：(B, T, hidden_dim)
        x = self.layers(x.reshape(B*T, self.hidden_dim))
        return x.reshape(B, T, self.channels, 64, 64)
    
class Decoder(nn.Module):
    def __init__(self, hidden_dim, obs_space):
        super().__init__()
        self.is_image = tools.is_image(obs_space)
        self.hidden_dim = hidden_dim

        if self.is_image:
            channels = obs_space.shape[0]
            self.obs = ImageDecoder(channels, hidden_dim)
        else:
            self.obs = MLP(hidden_dim, hidden_dim, math.prod(obs_space.shape), 3)
        
        # rew和con共用一个网络，减少参数量
        self.info = MLP(hidden_dim, hidden_dim, 3, 3)

    def forward(self, pred):
        obs = self.obs(pred)
        info = self.info(pred)
        rew, con, exceed = torch.chunk(info, chunks=3, dim=-1)
        return obs, rew, con, exceed	# o_{t+1}、r_{t+1}、环境继续信号c_{t+1}
