import numpy as np
import scipy.io as sio
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader

class HyperspectralDataset(Dataset):
    """高光谱数据集类"""
    def __init__(self, data, abundance, endmembers):
        self.data = torch.FloatTensor(data)
        self.abundance = torch.FloatTensor(abundance)
        if endmembers is not None:
            self.endmembers = torch.FloatTensor(endmembers)
        else:
            self.endmembers = None
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.abundance[idx]

def load_dataset(dataset_name='jasper'):
    """
    加载高光谱数据集
    
    Args:
        dataset_name: 'jasper' 或 'urban'
    
    Returns:
        data: 高光谱数据 (n_samples, n_bands)
        abundance: 丰度图 (n_samples, n_endmembers)
        endmembers: 端元矩阵 (n_bands, n_endmembers)
    """
    if dataset_name.lower() == 'jasper':
        # 加载Jasper Ridge数据集
        # 假设数据存储在 ./data/JasperRidge.mat
        mat_data = sio.loadmat('./data/JasperRidge.mat')
        data = mat_data['Y'].T  # (n_pixels, n_bands)
        endmembers = mat_data['M']  # (n_bands, n_endmembers)
        abundance = mat_data['A'].T if 'A' in mat_data else None  # (n_pixels, n_endmembers)
        
    elif dataset_name.lower() == 'urban':
        # 加载Urban数据集
        mat_data = sio.loadmat('./data/Urban.mat')
        data = mat_data['Y'].T
        endmembers = mat_data['M']
        abundance = mat_data['A'].T if 'A' in mat_data else None
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # 数据归一化
    data = normalize_data(data)
    
    return data, abundance, endmembers

def normalize_data(data):
    """归一化数据到[0, 1]"""
    data_min = np.min(data, axis=0, keepdims=True)
    data_max = np.max(data, axis=0, keepdims=True)
    normalized = (data - data_min) / (data_max - data_min + 1e-10)
    return normalized

def prepare_dataloaders(data, abundance, batch_size=128, train_ratio=0.8):
    """
    准备训练和测试数据加载器
    
    Args:
        data: 输入数据
        abundance: 丰度标签
        batch_size: 批次大小
        train_ratio: 训练集比例
    
    Returns:
        train_loader, test_loader
    """
    # 分割训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        data, abundance, train_size=train_ratio, random_state=42
    )
    
    # 创建数据集
    train_dataset = HyperspectralDataset(X_train, y_train, None)
    test_dataset = HyperspectralDataset(X_test, y_test, None)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader
