import torch
import numpy as np
import matplotlib.pyplot as plt
from data_loader import load_dataset
from model import DeepUnmixingCNN

def plot_endmember_comparison(true_endmembers, learned_endmembers, save_path=None):
    """
    可视化真实端元与学习端元的光谱曲线对比
    
    Args:
        true_endmembers: (n_bands, n_endmembers) 真实端元矩阵
        learned_endmembers: (n_bands, n_endmembers) 学习到的端元矩阵
        save_path: 保存路径（可选）
    """
    endmember_names = ["Tree", "Water", "Soil", "Road"]
    n_endmembers = true_endmembers.shape[1]
    n_bands = true_endmembers.shape[0]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for i in range(n_endmembers):
        ax = axes[i] if i < len(axes) else plt.subplot(2, 2, i + 1)
        
        # 绘制真实端元
        ax.plot(range(n_bands), true_endmembers[:, i], 
                'b-', linewidth=2, label='True', alpha=0.7)
        
        # 绘制学习端元
        ax.plot(range(n_bands), learned_endmembers[:, i], 
                'r--', linewidth=2, label='Learned', alpha=0.7)
        
        title = endmember_names[i] if i < len(endmember_names) else f'Endmember {i+1}'
        ax.set_title(f'{title} Spectral Signature', fontsize=12, fontweight='bold')
        ax.set_xlabel('Band Index', fontsize=10)
        ax.set_ylabel('Reflectance', fontsize=10)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_abundance_comparison(true_abundance, pred_abundance, img_shape, save_path=None):
    """
    可视化真实丰度与估计丰度的伪彩色热力图对比

    Args:
        true_abundance: (n_pixels, n_endmembers)
        pred_abundance: (n_pixels, n_endmembers)
        img_shape: (height, width)
        save_path: 保存路径（可选）
    """
    endmember_names = ["Tree", "Water", "Soil", "Road"]
    n_endmembers = true_abundance.shape[1]
    
    fig, axes = plt.subplots(2, n_endmembers, figsize=(4 * n_endmembers, 8))
    
    for i in range(n_endmembers):
        # 真实丰度
        ax_true = axes[0, i]
        abund_img = true_abundance[:, i].reshape(img_shape)
        title_true = f'True {endmember_names[i]}' if i < len(endmember_names) else f'True Endmember {i+1}'
        
        im_true = ax_true.imshow(abund_img, cmap='jet', vmin=0, vmax=1)
        ax_true.set_title(title_true, fontsize=12, fontweight='bold')
        ax_true.axis('off')
        plt.colorbar(im_true, ax=ax_true, fraction=0.046, pad=0.04)
        
        # 估计丰度
        ax_pred = axes[1, i]
        pred_img = pred_abundance[:, i].reshape(img_shape)
        title_pred = f'Estimated {endmember_names[i]}' if i < len(endmember_names) else f'Estimated Endmember {i+1}'
        
        im_pred = ax_pred.imshow(pred_img, cmap='jet', vmin=0, vmax=1)
        ax_pred.set_title(title_pred, fontsize=12, fontweight='bold')
        ax_pred.axis('off')
        plt.colorbar(im_pred, ax=ax_pred, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    # 加载数据
    data, abundance, endmembers = load_dataset('jasper')
    img_shape = (100, 100)

    # 加载模型
    n_bands = data.shape[1]
    n_endmembers = endmembers.shape[1]
    model = DeepUnmixingCNN(n_bands, n_endmembers)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.load_state_dict(torch.load('best_model.pth', map_location=device))
    model = model.to(device)  # 修正：模型移到device
    model.eval()

    # 推理得到估计丰度
    with torch.no_grad():
        data_tensor = torch.FloatTensor(data).to(device)
        pred_abundance, _ = model(data_tensor)
        pred_abundance = pred_abundance.cpu().numpy()

    # 可视化对比
    plot_abundance_comparison(abundance, pred_abundance, img_shape, save_path='abundance_compare.png')
    
    # 可视化端元重构
    learned_endmembers = model.get_endmembers()
    # 转置learned_endmembers以匹配endmembers的形状 (n_bands, n_endmembers)
    learned_endmembers_transposed = learned_endmembers.T
    print("\nVisualizing endmember reconstruction...")
    plot_endmember_comparison(endmembers, learned_endmembers_transposed, save_path='endmember_comparison.png')
