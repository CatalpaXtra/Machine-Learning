import numpy as np
import cv2
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import joblib
import torch
import torchvision.models as models
import torchvision.transforms as transforms


class FeatureExtractor:
    """
    特征提取模块
    """
    
    def __init__(self, config):
        self.config = config
        self.method = config['features']['method']
        self.target_size = config['features']['size']
        self.cnn_type = config['features']['cnn_type']
        self.cnn_model = None
        
        # 降维配置
        self.use_dim_reduction = config['dimensionality_reduction']['use']
        if self.use_dim_reduction:
            self.dim_reduction_method = config['dimensionality_reduction']['method']
            self.n_components = config['dimensionality_reduction']['n_components']
            self.dim_reducer = None
    
    
    def _load_image(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法读取图像: {image_path}")
        return cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), self.target_size)
    
    
    def _extract_cnn_features(self, img):
        if self.cnn_model is None:
            print(f"正在加载{self.cnn_type.upper()}模型...")
            if self.cnn_type == 'vgg16':
                self.cnn_model = models.vgg16(pretrained=True)
                self.cnn_model.eval()
                self.transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            else:
                raise ValueError(f"未识别的CNN类型: {self.cnn_type}")
            print(f"{self.cnn_type.upper()}特征提取器加载完成")
        
        img_tensor = self.transform(img).unsqueeze(0)
        with torch.no_grad():
            features = self.cnn_model(img_tensor)
        return features.numpy().flatten()
    
    
    def extract_features(self, image_path):
        img = self._load_image(image_path)
        
        # 根据配置选择特征提取方法
        if self.method == 'cnn':
            return self._extract_cnn_features(img)
        else:
            raise ValueError(f"不支持的特征提取方法: {self.method}")
    
    
    def extract_batch_features(self, image_paths):
        features_list = []
        
        for path in image_paths:
            try:
                features_list.append(self.extract_features(path))
            except Exception as e:
                print(f"处理图像 {path} 时出错: {e}")
        
        return np.array(features_list)
    
    
    def fit_dim_reducer(self, features):
        if not self.use_dim_reduction:
            return self
            
        print(f"训练{self.dim_reduction_method.upper()}降维器")
        if self.dim_reduction_method == 'pca':
            self.dim_reducer = PCA(n_components=min(self.n_components, features.shape[1]))
        elif self.dim_reduction_method == 'lda':
            self.dim_reducer = LinearDiscriminantAnalysis(n_components=min(self.n_components, features.shape[1]))
            
        return self
    
    
    def transform_features(self, features, labels=None):
        """
        对特征进行降维
        
        参数:
            features: 特征矩阵
            labels: 类别标签，仅在使用LDA时需要
            
        返回:
            降维后的特征
        """
        if not self.use_dim_reduction or self.dim_reducer is None:
            return features
            
        if self.dim_reduction_method == 'pca':
            if not hasattr(self.dim_reducer, 'components_'):
                self.dim_reducer.fit(features)
            return self.dim_reducer.transform(features)
        elif self.dim_reduction_method == 'lda':
            if not hasattr(self.dim_reducer, 'coef_'):
                if labels is None:
                    raise ValueError("使用LDA降维时需要提供标签")
                self.dim_reducer.fit(features, labels)
            return self.dim_reducer.transform(features)
    
    def save(self, path):
        cnn_model = self.cnn_model
        self.cnn_model = None
        
        joblib.dump(self, path)
        
        self.cnn_model = cnn_model
    
    
    @classmethod
    def load(cls, path):
        extractor = joblib.load(path)
        
        if extractor.method == 'cnn':
            print(f"正在加载{extractor.cnn_type.upper()}模型...")
            if extractor.cnn_type == 'vgg16':
                extractor.cnn_model = models.vgg16(pretrained=True)
                extractor.cnn_model.eval()
                extractor.transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            else:
                raise ValueError(f"未识别的CNN类型: {extractor.cnn_type}")
            print(f"{extractor.cnn_type.upper()}特征提取器加载完成")
        
        return extractor 