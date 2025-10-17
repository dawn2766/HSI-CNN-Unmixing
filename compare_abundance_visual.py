import torch
import numpy as np
import matplotlib.pyplot as plt
from data_loader import load_dataset
from model import DeepUnmixingCNN

def plot_abundance_comparison(true_abundance, pred_abundance, img_shape, save_path=None):
    """
    可视化真实丰度与估计丰度的灰度分布对比

    Args:
        true_abundance: (n_pixels, n_endmembers)
        pred_abundance: (n_pixels, n_endmembers)
        img_shape: (height, width)
        save_path: 保存路径（可选）
    """
    endmember_names = ["Tree", "Water", "Soil", "Road"]
    n_endmembers = true_abundance.shape[1]
    plt.figure(figsize=(8 * n_endmembers, 8))
    for i in range(n_endmembers):
        # 真实丰度
        plt.subplot(2, n_endmembers, i + 1)
        abund_img = true_abundance[:, i].reshape(img_shape)
        title_true = f'True {endmember_names[i]}' if i < len(endmember_names) else f'True Endmember {i+1}'
        plt.imshow(abund_img, cmap='gray')
        plt.title(title_true)
        plt.axis('off')
        # 估计丰度
        plt.subplot(2, n_endmembers, n_endmembers + i + 1)
        pred_img = pred_abundance[:, i].reshape(img_shape)
        title_pred = f'Estimated {endmember_names[i]}' if i < len(endmember_names) else f'Estimated Endmember {i+1}'
        plt.imshow(pred_img, cmap='gray')
        plt.title(title_pred)
        plt.axis('off')
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
