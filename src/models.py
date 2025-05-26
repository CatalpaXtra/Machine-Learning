import torch.nn as nn
import torchvision.models as models


class ImageClassifier(nn.Module):
    def __init__(self, num_classes):
        super(ImageClassifier, self).__init__()
        # 使用预训练的ResNet50作为基础模型
        self.base_model = models.resnet50(pretrained=True)
        
        # 冻结基础模型的参数
        for param in self.base_model.parameters():
            param.requires_grad = False
            
        # 修改最后的全连接层
        num_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        return self.base_model(x)
    
    def unfreeze_layers(self, num_layers=0):
        """解冻最后几层进行微调"""
        if num_layers == 0:
            return
            
        # 获取所有层
        layers = list(self.base_model.children())
        
        # 解冻最后几层
        for layer in layers[-num_layers:]:
            for param in layer.parameters():
                param.requires_grad = True 