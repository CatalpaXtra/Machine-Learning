import os
from datetime import datetime
from utils.config import load_config
from data_loader import load_data
from extractor import FeatureExtractor
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

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
    
    # 转换为PyTorch张量
    train_features = torch.FloatTensor(train_features)
    train_labels = torch.LongTensor(train_labels)
    
    # 创建数据加载器
    train_dataset = TensorDataset(train_features, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # 初始化模型
    model = SimpleNN(input_size=train_features.shape[1], hidden_size=128, output_size=len(set(train_labels)))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 训练模型
    print(f"开始训练PyTorch模型: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    for epoch in range(10):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')
    
    print(f"PyTorch模型训练完成: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 保存模型和特征提取器
    os.makedirs(os.path.dirname(config['output']['model_path']), exist_ok=True)
    torch.save(model.state_dict(), config['output']['model_path'])
    feature_extractor.save(config['output']['feature_extractor_path'])
    print(f"训练完成: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == '__main__':
    from utils.args import parse_args
    main(parse_args())