import numpy as np
from datetime import datetime
from utils.config import load_config
from data_loader import load_data
from extractor import FeatureExtractor
from utils.visualization import plot_confusion_matrix, plot_roc_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score
import torch
from train import ImprovedNN
import warnings

# 忽略警告
warnings.filterwarnings('ignore')

def calculate_metrics(y_true, y_pred, y_prob):
    """计算各种评估指标
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        y_prob: 预测概率
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1': f1_score(y_true, y_pred, average='weighted')
    }
    
    return metrics


def test(feature_extractor, model, config, device):
    # 加载测试数据
    test_features, test_labels = load_data(
        config['data']['test_dir'],
        feature_extractor,
        batch_size=1000
    )
    
    # 特征降维
    if config['dimensionality_reduction']['use']:
        test_features = feature_extractor.transform_features(test_features, test_labels)
    
    # 将NumPy数组转换为PyTorch张量
    test_features = torch.FloatTensor(test_features)
    test_labels = torch.LongTensor(test_labels)
    
    # 创建数据加载器
    test_dataset = torch.utils.data.TensorDataset(test_features, test_labels)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config['model']['batch_size'], shuffle=False)
    
    # 预测
    model.eval()
    all_predictions = []
    all_probabilities = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predictions = torch.max(outputs, 1)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 计算评估指标
    metrics = calculate_metrics(
        np.array(all_labels),
        np.array(all_predictions),
        np.array(all_probabilities)
    )
    
    # 输出评估结果
    print("\n评估结果:")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")
    
    # 绘制混淆矩阵和ROC曲线
    plot_confusion_matrix(
        np.array(all_labels),
        np.array(all_predictions),
        config['output']['results_dir']
    )
    # plot_roc_curve(np.array(all_labels), np.array(all_probabilities), config['output']['results_dir'])


def main(args):
    # 加载配置
    config = load_config(args.config)
    print(f"开始测试: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载模型
    model = ImprovedNN(
        input_size=config['model']['input_size'],
        hidden_sizes=config['model']['hidden_sizes'],
        output_size=config['model']['output_size'],
        dropout_rate=config['model']['dropout_rate']
    ).to(device)
    
    # 加载检查点
    checkpoint = torch.load(config['output']['model_path'])
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"加载模型完成，最佳验证准确率: {checkpoint['best_val_acc']:.2f}%")
    
    # 加载特征提取器
    feature_extractor = FeatureExtractor.load(config['output']['feature_extractor_path'])
    
    test(feature_extractor, model, config, device)
    
    print(f"测试完成: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == '__main__':
    from utils.args import parse_args
    main(parse_args()) 