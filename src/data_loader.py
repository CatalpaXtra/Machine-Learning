import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import yaml

class ImageDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.classes = self._get_classes()
        self.class_to_idx = {str(i+1): i for i in range(len(self.classes))}  # 使用数字作为键
        self.images = self._load_images()
        
    def _get_classes(self):
        with open('dataset/classes.txt', 'r', encoding='utf-8') as f:
            return [line.strip().split(' ')[1] for line in f.readlines()]  # 只取类别名称
    
    def _load_images(self):
        images = []
        for class_idx in self.class_to_idx.keys():  # 使用数字索引
            class_dir = os.path.join(self.data_dir, class_idx)
            if os.path.isdir(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.endswith(('.jpg', '.jpeg', '.png')):
                        images.append((os.path.join(class_dir, img_name), class_idx))
        return images
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path, class_idx = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        label = self.class_to_idx[class_idx]
        return image, label

def get_data_loaders(config):
    # 定义数据转换
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # 创建训练集和测试集
    train_dataset = ImageDataset(config['data']['train_dir'], transform=transform)
    test_dataset = ImageDataset(config['data']['test_dir'], transform=transform)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['model']['batch_size'],
        shuffle=True,
        num_workers=4
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['model']['batch_size'],
        shuffle=False,
        num_workers=4
    )
    
    return train_loader, test_loader, len(train_dataset.classes)