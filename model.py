import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepUnmixingCNN(nn.Module):
    """
    基于深度卷积神经网络的高光谱解混模型
    参考: Li et al., 2018, IEEE TGRS
    """
    def __init__(self, n_bands, n_endmembers, hidden_dims=[256, 128, 64]):
        super(DeepUnmixingCNN, self).__init__()
        
        self.n_bands = n_bands
        self.n_endmembers = n_endmembers
        
        # 构建编码器（特征提取）
        layers = []
        input_dim = n_bands
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            input_dim = hidden_dim
        
        self.encoder = nn.Sequential(*layers)
        
        # 输出层：映射到丰度空间
        self.abundance_layer = nn.Linear(hidden_dims[-1], n_endmembers)
        
        # 端元层（可学习或固定）
        self.endmember_layer = nn.Parameter(
            torch.randn(n_bands, n_endmembers),
            requires_grad=True
        )
        
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入高光谱像素 (batch_size, n_bands)
        
        Returns:
            abundance: 估计的丰度 (batch_size, n_endmembers)
            reconstruction: 重建的光谱 (batch_size, n_bands)
        """
        # 特征提取
        features = self.encoder(x)
        
        # 丰度估计
        abundance = self.abundance_layer(features)
        
        # 应用和为1约束和非负约束
        abundance = F.softmax(abundance, dim=1)
        
        # 光谱重建：Y = M * A
        reconstruction = torch.matmul(abundance, self.endmember_layer.T)
        
        return abundance, reconstruction
    
    def get_endmembers(self):
        """获取学习到的端元"""
        return self.endmember_layer.detach().cpu().numpy()


class DeepUnmixingAutoencoder(nn.Module):
    """
    基于自编码器的深度解混模型
    """
    def __init__(self, n_bands, n_endmembers):
        super(DeepUnmixingAutoencoder, self).__init__()
        
        self.n_bands = n_bands
        self.n_endmembers = n_endmembers
        
        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(n_bands, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, n_endmembers),
            nn.Softmax(dim=1)  # 确保丰度和为1
        )
        
        # 解码器（端元矩阵）
        self.decoder = nn.Linear(n_endmembers, n_bands, bias=False)
        
    def forward(self, x):
        # 编码：光谱 -> 丰度
        abundance = self.encoder(x)
        
        # 解码：丰度 -> 重建光谱
        reconstruction = self.decoder(abundance)
        
        return abundance, reconstruction
    
    def get_endmembers(self):
        """获取端元矩阵"""
        return self.decoder.weight.detach().cpu().numpy().T
