import os
import yaml
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from data_loader import get_data_loaders
from models import ImageClassifier
import logging
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def setup_logging(results_dir):
    os.makedirs(results_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(results_dir, 'testing.log')),
            logging.StreamHandler()
        ]
    )

def test(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Testing'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return total_loss/len(test_loader), all_preds, all_labels

def plot_confusion_matrix(cm, classes, results_dir):
    plt.figure(figsize=(20, 20))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(results_dir, 'confusion_matrix.png'))
    plt.close()

def main():
    # 加载配置
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')
    
    # 设置日志
    setup_logging(config['output']['results_dir'])
    
    # 获取数据加载器
    _, test_loader, num_classes = get_data_loaders(config)
    logging.info(f'Number of classes: {num_classes}')
    
    # 创建模型
    model = ImageClassifier(num_classes).to(device)
    
    # 加载最佳模型
    model.load_state_dict(torch.load(os.path.join(config['output']['results_dir'], 'best_model.pth')))
    logging.info('Loaded best model')
    
    # 定义损失函数
    criterion = nn.CrossEntropyLoss()
    
    # 测试模型
    test_loss, all_preds, all_labels = test(model, test_loader, criterion, device)
    logging.info(f'Test Loss: {test_loss:.4f}')
    
    # 计算分类报告
    report = classification_report(all_labels, all_preds)
    logging.info('\nClassification Report:\n' + report)
    
    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, range(num_classes), config['output']['results_dir'])
    logging.info('Confusion matrix saved')
    
    # 保存分类报告
    with open(os.path.join(config['output']['results_dir'], 'classification_report.txt'), 'w') as f:
        f.write(report)

if __name__ == '__main__':
    main() 