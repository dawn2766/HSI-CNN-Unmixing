import torch
import numpy as np
import matplotlib.pyplot as plt
from data_loader import load_dataset, prepare_dataloaders
from model import DeepUnmixingCNN, DeepUnmixingAutoencoder
from train import UnmixingTrainer
from metrics import calculate_metrics, mean_sad
from compare_abundance_visual import plot_abundance_comparison, plot_endmember_comparison

def plot_abundance_comparison(true_abundance, pred_abundance, img_shape, save_path=None):
    """
    可视化真实丰度与估计丰度的灰度分布对比
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

def run_experiment(dataset_name='jasper', model_type='cnn', mode='train'):
    """
    运行高光谱解混实验

    Args:
        dataset_name: 'jasper' 或 'urban'
        model_type: 'cnn' 或 'autoencoder'
        mode: 'train' 或 'test'
    """
    print(f"\n{'='*50}")
    print(f"Running experiment on {dataset_name.upper()} dataset")
    print(f"Model: {model_type.upper()}")
    print(f"Mode: {mode.upper()}")
    print(f"{'='*50}\n")

    # 加载数据
    print("Loading dataset...")
    data, abundance, endmembers = load_dataset(dataset_name)

    n_bands = data.shape[1]
    n_endmembers = endmembers.shape[1]

    print(f"Data shape: {data.shape}")
    print(f"Number of bands: {n_bands}")
    print(f"Number of endmembers: {n_endmembers}")

    # 准备数据加载器
    train_loader, test_loader = prepare_dataloaders(
        data, abundance, batch_size=128, train_ratio=0.8
    )

    # 创建模型
    if model_type == 'cnn':
        model = DeepUnmixingCNN(n_bands, n_endmembers)
    else:
        model = DeepUnmixingAutoencoder(n_bands, n_endmembers)

    print(f"\nModel architecture:")
    print(model)

    trainer = UnmixingTrainer(model)

    if mode == 'train':
        # 训练
        print("\nStarting training...")
        trainer.fit(
            train_loader, test_loader, 
            epochs=100, lr=0.001,
            use_abundance_loss=(abundance is not None)
        )
    else:
        # 测试模式：只加载模型参数
        print("\nLoading model weights from best_model.pth ...")
        model.load_state_dict(torch.load('best_model.pth', map_location=trainer.device))

    # 评估
    print("\nEvaluating on test set...")
    model.eval()

    test_data = []
    test_abundance = []
    pred_abundance = []
    pred_reconstruction = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(trainer.device)
            abundance_pred, recon_pred = model(batch_x)

            test_data.append(batch_x.cpu().numpy())
            test_abundance.append(batch_y.numpy())
            pred_abundance.append(abundance_pred.cpu().numpy())
            pred_reconstruction.append(recon_pred.cpu().numpy())

    test_data = np.vstack(test_data)
    test_abundance = np.vstack(test_abundance)
    pred_abundance = np.vstack(pred_abundance)
    pred_reconstruction = np.vstack(pred_reconstruction)

    # 计算指标
    metrics = calculate_metrics(
        test_data, pred_reconstruction,
        test_abundance, pred_abundance
    )

    print("\n" + "="*50)
    print("RESULTS")
    print("="*50)
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.6f}")

    # 比较端元
    learned_endmembers = model.get_endmembers()
    if endmembers is not None:
        endmember_sad = mean_sad(endmembers, learned_endmembers)
        print(f"Mean SAD (Endmembers): {endmember_sad:.6f}")
        
        # 可视化端元重构
        print("\nVisualizing endmember reconstruction...")
        plot_endmember_comparison(endmembers, learned_endmembers, 
                                 save_path=f'endmember_comparison_{dataset_name}.png')
        plot_endmember_heatmap(endmembers, learned_endmembers, 
                              save_path=f'endmember_heatmap_{dataset_name}.png')

    # 可视化
    plot_results(trainer.history, pred_abundance[:100], test_abundance[:100])

    # 丰度分布灰度图可视化（全像素，Jasper Ridge）
    if dataset_name == 'jasper' and data.shape[0] == 10000 and abundance is not None:
        print("\nVisualizing abundance maps for all pixels...")
        img_shape = (100, 100)
        device = trainer.device
        model.eval()
        with torch.no_grad():
            data_tensor = torch.FloatTensor(data).to(device)
            pred_abundance_all, _ = model(data_tensor)
            pred_abundance_all = pred_abundance_all.detach().cpu().numpy()
        plot_abundance_comparison(abundance, pred_abundance_all, img_shape, save_path='abundance_compare.png')

    return model, metrics

def plot_results(history, pred_abundance, true_abundance):
    """可视化结果"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # 损失曲线
    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].plot(history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training History')
    axes[0].legend()
    axes[0].grid(True)
    
    # 丰度对比
    n_samples = min(100, len(pred_abundance))
    axes[1].scatter(true_abundance[:n_samples].flatten(), 
                   pred_abundance[:n_samples].flatten(), alpha=0.5)
    axes[1].plot([0, 1], [0, 1], 'r--', label='Ideal')
    axes[1].set_xlabel('True Abundance')
    axes[1].set_ylabel('Predicted Abundance')
    axes[1].set_title('Abundance Comparison')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('results.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    # 选择模式：'train' 或 'test'
    mode = 'train'  # 修改为 'test' 即可只评估
    mode = 'test'
    print("Testing on Jasper Ridge dataset...")
    model_jasper, metrics_jasper = run_experiment('jasper', 'cnn', mode=mode)

    # # 在Urban数据集上测试
    # print("\n\nTesting on Urban dataset...")
    # model_urban, metrics_urban = run_experiment('urban', 'autoencoder', mode=mode)
