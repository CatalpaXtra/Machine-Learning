import os
from datetime import datetime
from utils.config import load_config
from data_loader import load_data
from extractor import FeatureExtractor
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import numpy as np
from torch.nn import functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout_rate=0.3):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(out_features, out_features),
            nn.BatchNorm1d(out_features)
        )
        self.shortcut = nn.Linear(in_features, out_features) if in_features != out_features else nn.Identity()
        
    def forward(self, x):
        return F.relu(self.block(x) + self.shortcut(x))

class ImprovedNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout_rate=0.3):
        super(ImprovedNN, self).__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.BatchNorm1d(hidden_sizes[0]),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_sizes[i], hidden_sizes[i+1], dropout_rate)
            for i in range(len(hidden_sizes)-1)
        ])
        
        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)
        
    def forward(self, x):
        x = self.input_layer(x)
        for block in self.residual_blocks:
            x = block(x)
        return self.output_layer(x)

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # 数据增强
        if np.random.random() < 0.5:
            noise = torch.randn_like(inputs) * 0.1
            inputs = inputs + noise
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # L2正则化
        l2_lambda = 0.01
        l2_reg = torch.tensor(0., device=device)
        for param in model.parameters():
            l2_reg += torch.norm(param)
        loss += l2_lambda * l2_reg
        
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    return total_loss / len(train_loader), 100. * correct / total

def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return total_loss / len(val_loader), 100. * correct / total

def main(args):
    # 加载配置
    config = load_config(args.config)
    print(f"开始特征提取: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 初始化特征提取器
    feature_extractor = FeatureExtractor(config)
    
    # 加载训练数据
    train_features, train_labels = load_data(
        config['data']['train_dir'],
        feature_extractor,
        batch_size=1000
    )
    
    # 特征降维
    if config['dimensionality_reduction']['use']:
        feature_extractor.fit_dim_reducer(train_features)
        train_features = feature_extractor.transform_features(train_features, train_labels)
    
    # 转换为PyTorch张量
    train_features = torch.FloatTensor(train_features)
    train_labels = torch.LongTensor(train_labels)
    
    # 划分训练集和验证集
    dataset = TensorDataset(train_features, train_labels)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    # 初始化模型
    model = ImprovedNN(
        input_size=train_features.shape[1],
        hidden_sizes=config['model']['hidden_sizes'],
        output_size=len(set(train_labels))
    ).to(device)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # 添加标签平滑
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    
    # 训练参数
    num_epochs = 100  # 增加训练轮数
    best_val_acc = 0
    patience = 15  # 增加早停耐心值
    patience_counter = 0
    
    print(f"开始训练PyTorch模型: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # 学习率调整
        scheduler.step()
        
        # 早停检查
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            # 保存最佳模型
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_acc': best_val_acc,
            }, config['output']['model_path'])
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
    
    print(f"训练完成: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"最佳验证准确率: {best_val_acc:.2f}%")
    
    # 保存特征提取器
    feature_extractor.save(config['output']['feature_extractor_path'])

if __name__ == '__main__':
    from utils.args import parse_args
    main(parse_args())