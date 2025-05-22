import numpy as np
import joblib
from datetime import datetime
from sklearn.svm import SVC
from utils.config import load_config
from data_loader import load_data
from extractor import FeatureExtractor
from utils.visualization import plot_confusion_matrix, plot_roc_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score


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
    
    if y_prob is not None:
        try:
            metrics['auc'] = roc_auc_score(y_true, y_prob, multi_class='ovr')
        except:
            metrics['auc'] = np.nan
    
    return metrics 


def test(feature_extractor, classifier, config):
    # 加载测试数据
    test_features, test_labels = load_data(
        config['data']['test_dir'],
        feature_extractor,
        batch_size=1000
    )
    
    # 特征降维
    if config['dimensionality_reduction']['use']:
        test_features = feature_extractor.transform_features(test_features, test_labels)
    
    # 预测
    predictions = classifier.predict(test_features)
    probabilities = classifier.predict_proba(test_features)
    
    # 计算评估指标
    metrics = calculate_metrics(test_labels, predictions, probabilities)
    
    # 输出评估结果
    print("\n评估结果:")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")
    
    # 绘制混淆矩阵和ROC曲线
    plot_confusion_matrix(test_labels, predictions, config['output']['results_dir'])
    plot_roc_curve(test_labels, probabilities, config['output']['results_dir'])


def main(args):
    # 加载配置
    config = load_config(args.config)
    print(f"开始测试: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 加载模型
    svm = SVC(
        kernel=config['model']['kernel'],
        C=config['model']['C'],
        gamma=config['model']['gamma'],
        probability=config['model']['probability'],
        verbose=config['model']['verbose'],
        cache_size=config['model']['cache_size'],
        max_iter=config['model']['max_iter'],
        tol=config['model']['tol'],
        class_weight=config['model']['class_weight']
    )
    svm = joblib.load(config['output']['model_path'])
    
    # 加载特征提取器
    feature_extractor = FeatureExtractor.load(config['output']['feature_extractor_path'])
    
    test(feature_extractor, svm, config)
    
    print(f"测试完成: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == '__main__':
    from utils.args import parse_args
    main(parse_args()) 