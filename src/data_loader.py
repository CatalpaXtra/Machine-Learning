import os
import glob
import numpy as np
from tqdm import tqdm

def get_image_paths(data_dir):
    """获取所有图像路径和对应标签"""
    paths, labels = [], []
    for class_dir in sorted(os.listdir(data_dir)):
        if not os.path.isdir(os.path.join(data_dir, class_dir)):
            continue
            
        label = int(class_dir)
        class_images = glob.glob(os.path.join(data_dir, class_dir, "*.jpg")) + \
                      glob.glob(os.path.join(data_dir, class_dir, "*.JPG"))
        
        paths.extend(class_images)
        labels.extend([label] * len(class_images))
    
    return paths, np.array(labels)


def load_data(data_dir, feature_extractor, batch_size=1000, verbose=True):
    """统一加载数据并提取特征，支持本地缓存"""
    # 创建缓存目录
    cache_dir = os.path.join(os.path.dirname(data_dir), 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    
    # 生成缓存文件名
    cache_name = os.path.basename(data_dir)
    features_cache = os.path.join(cache_dir, f'{cache_name}_features.npy')
    labels_cache = os.path.join(cache_dir, f'{cache_name}_labels.npy')
    
    # 检查缓存是否存在
    if os.path.exists(features_cache) and os.path.exists(labels_cache):
        print("从 dataset/cache/ 下加载数据:")
        features = np.load(features_cache)
        labels = np.load(labels_cache)
        print("加载完成")
        return features, labels
    
    # 如果缓存不存在，则提取特征
    print("提取特征并保存到 dataset/cache/")
    image_paths, labels = get_image_paths(data_dir)
    features = []
    
    for i in tqdm(range(0, len(image_paths), batch_size), desc="提取特征", disable=not verbose):
        batch = image_paths[i:i+batch_size]
        features.append(feature_extractor.extract_batch_features(batch))
    
    features = np.vstack(features)
    
    # 保存到缓存
    np.save(features_cache, features)
    np.save(labels_cache, labels)
    print("特征提取完成，已保存到 dataset/cache/")
    
    return features, labels