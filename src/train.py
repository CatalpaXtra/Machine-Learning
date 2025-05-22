import os
from datetime import datetime
from utils.config import load_config
from data_loader import load_data
from extractor import FeatureExtractor
from sklearn.svm import SVC
import joblib

def main(args):
    # 加载配置
    config = load_config(args.config)
    print(f"开始特征提取: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
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
    
    # 训练SVM模型
    print(f"开始训练SVM模型: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
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
    
    # 训练模型
    svm.fit(train_features, train_labels)
    print(f"SVM模型训练完成: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 保存模型和特征提取器
    os.makedirs(os.path.dirname(config['output']['model_path']), exist_ok=True)
    joblib.dump(svm, config['output']['model_path'])
    feature_extractor.save(config['output']['feature_extractor_path'])
    print(f"训练完成: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == '__main__':
    from utils.args import parse_args
    main(parse_args())