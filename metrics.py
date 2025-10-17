import numpy as np

def rmse(y_true, y_pred):
    """均方根误差"""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def sad(spectrum1, spectrum2):
    """
    光谱角距离 (Spectral Angle Distance)
    
    Args:
        spectrum1, spectrum2: 光谱向量或矩阵
    
    Returns:
        SAD值（弧度）
    """
    # 归一化
    norm1 = np.linalg.norm(spectrum1, axis=-1, keepdims=True)
    norm2 = np.linalg.norm(spectrum2, axis=-1, keepdims=True)
    
    spectrum1_norm = spectrum1 / (norm1 + 1e-10)
    spectrum2_norm = spectrum2 / (norm2 + 1e-10)
    
    # 计算余弦相似度
    cos_sim = np.sum(spectrum1_norm * spectrum2_norm, axis=-1)
    cos_sim = np.clip(cos_sim, -1.0, 1.0)
    
    # 计算角度
    angle = np.arccos(cos_sim)
    
    return np.mean(angle)

def mean_sad(endmembers_true, endmembers_pred):
    """计算端元之间的平均SAD"""
    n_endmembers = endmembers_true.shape[1]
    sad_values = []
    
    for i in range(n_endmembers):
        sad_val = sad(endmembers_true[:, i], endmembers_pred[:, i])
        sad_values.append(sad_val)
    
    return np.mean(sad_values)

def sre(y_true, y_pred):
    """信号重建误差 (Signal Reconstruction Error)"""
    numerator = np.sum((y_true - y_pred) ** 2)
    denominator = np.sum(y_true ** 2)
    return 10 * np.log10(denominator / (numerator + 1e-10))

def calculate_metrics(y_true, y_pred, abundance_true=None, abundance_pred=None):
    """
    计算所有评估指标
    
    Args:
        y_true: 真实光谱
        y_pred: 预测光谱
        abundance_true: 真实丰度（可选）
        abundance_pred: 预测丰度（可选）
    
    Returns:
        字典包含所有指标
    """
    metrics = {
        'RMSE': rmse(y_true, y_pred),
        'SAD': sad(y_true, y_pred),
        'SRE': sre(y_true, y_pred)
    }
    
    if abundance_true is not None and abundance_pred is not None:
        metrics['Abundance_RMSE'] = rmse(abundance_true, abundance_pred)
    
    return metrics
