import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

class UnmixingTrainer:
    """高光谱解混训练器"""
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.history = {'train_loss': [], 'val_loss': []}
        
    def reconstruction_loss(self, x, reconstruction):
        """重建损失（MSE）"""
        return nn.MSELoss()(reconstruction, x)
    
    def abundance_loss(self, abundance, target_abundance):
        """丰度损失（如果有真实丰度标签）"""
        return nn.MSELoss()(abundance, target_abundance)
    
    def sad_loss(self, x, reconstruction):
        """光谱角距离损失"""
        # 归一化
        x_norm = F.normalize(x, p=2, dim=1)
        recon_norm = F.normalize(reconstruction, p=2, dim=1)
        
        # 计算余弦相似度
        cos_sim = torch.sum(x_norm * recon_norm, dim=1)
        cos_sim = torch.clamp(cos_sim, -1.0, 1.0)
        
        # SAD = arccos(cos_sim)
        sad = torch.acos(cos_sim)
        return torch.mean(sad)
    
    def train_epoch(self, train_loader, optimizer, use_abundance_loss=False):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        
        for batch_x, batch_y in tqdm(train_loader, desc='Training'):
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            optimizer.zero_grad()
            
            # 前向传播
            abundance, reconstruction = self.model(batch_x)
            
            # 计算损失
            recon_loss = self.reconstruction_loss(batch_x, reconstruction)
            sad = self.sad_loss(batch_x, reconstruction)
            
            loss = recon_loss + 0.1 * sad
            
            # 如果有真实丰度，添加丰度损失
            if use_abundance_loss and batch_y is not None:
                abund_loss = self.abundance_loss(abundance, batch_y)
                loss += 0.5 * abund_loss
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader, use_abundance_loss=False):
        """验证"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                abundance, reconstruction = self.model(batch_x)
                
                recon_loss = self.reconstruction_loss(batch_x, reconstruction)
                sad = self.sad_loss(batch_x, reconstruction)
                loss = recon_loss + 0.1 * sad
                
                if use_abundance_loss and batch_y is not None:
                    abund_loss = self.abundance_loss(abundance, batch_y)
                    loss += 0.5 * abund_loss
                
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def fit(self, train_loader, val_loader, epochs=100, lr=0.001, use_abundance_loss=False):
        """训练模型"""
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10
        )
        
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 20
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader, optimizer, use_abundance_loss)
            val_loss = self.validate(val_loader, use_abundance_loss)
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            
            scheduler.step(val_loss)
            
            print(f'Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
            
            # 早停
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f'Early stopping at epoch {epoch+1}')
                    break
        
        # 加载最佳模型
        self.model.load_state_dict(torch.load('best_model.pth'))
