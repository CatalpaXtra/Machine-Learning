import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from data_loader import get_data_loaders
from models import ImageClassifier
import logging
from torch.cuda.amp import autocast, GradScaler

def setup_logging(model_dir):
    """设置日志，同时输出到控制台和文件"""
    os.makedirs(model_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(model_dir, 'training.log')),
            logging.StreamHandler()
        ]
    )


def train_epoch(model, train_loader, criterion, optimizer, device, scaler):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for images, labels in pbar:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        # 使用混合精度训练
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        # 使用scaler进行反向传播
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({'loss': total_loss/total, 'acc': 100.*correct/total})
    
    return total_loss/len(train_loader), 100.*correct/total


def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc='Validation'):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            # 使用混合精度推理
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return total_loss/len(val_loader), 100.*correct/total


def main():
    # 加载配置
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 设置日志
    setup_logging(config['output']['model_dir'])
    logging.info(f'Using device: {device}')
    
    # 获取数据加载器
    train_loader, val_loader, num_classes = get_data_loaders(config)
    logging.info(f'Number of classes: {num_classes}')
    
    # 创建模型
    model = ImageClassifier(num_classes).to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['model']['learning_rate'],
        weight_decay=config['model']['weight_decay'],
        betas=(
            config['model']['optimizer']['beta1'],
            config['model']['optimizer']['beta2']
        ),
        eps=config['model']['optimizer']['eps']
    )
    
    # 创建GradScaler用于混合精度训练
    scaler = GradScaler()
    
    # 学习率调度器
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=config['model']['lr_scheduler']['factor'],
        patience=config['model']['lr_scheduler']['patience'],
        min_lr=config['model']['lr_scheduler']['min_lr'],
        verbose=True  # 添加verbose参数以显示学习率变化
    )
    
    # 训练循环
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(config['model']['num_epochs']):
        logging.info(f'\nEpoch {epoch+1}/{config["model"]["num_epochs"]}')
        
        # 训练
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, scaler)
        logging.info(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        
        # 验证
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        logging.info(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # 学习率调整
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        logging.info(f'Current learning rate: {current_lr:.8f}')
        
        # 保存最佳模型
        if val_loss < best_val_loss - config['model']['early_stopping']['min_delta']:
            best_val_loss = val_loss
            patience_counter = 0
            # 保存模型到models目录
            model_path = config['output']['model_path']
            torch.save(model.state_dict(), model_path)
            logging.info(f'Saved best model to {model_path}')
        else:
            patience_counter += 1
            logging.info(f'No improvement for {patience_counter} epochs')
            
        # 早停
        if patience_counter >= config['model']['early_stopping']['patience']:
            logging.info('Early stopping triggered')
            break
    
    logging.info('Training completed')


if __name__ == '__main__':
    main()