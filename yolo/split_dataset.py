import os
import json
import shutil
import random
import argparse


def split_dataset(train_images_dir, train_annotations_file, val_ratio=0.2, seed=42):
    # 设置随机种子
    random.seed(seed)
    
    # 创建验证集目录
    val_images_dir = train_images_dir.replace('train', 'val')
    val_annotations_file = train_annotations_file.replace('train', 'val')
    
    # 确保验证集目录存在
    os.makedirs(val_images_dir, exist_ok=True)
    
    # 读取COCO格式的标注文件
    print(f"正在读取标注文件: {train_annotations_file}")
    with open(train_annotations_file, 'r', encoding='utf-8') as f:
        annotations = json.load(f)
    
    # 获取所有图片文件名
    all_images = annotations['images']
    total_images = len(all_images)
    val_count = int(total_images * val_ratio)
    
    print(f"总图片数量: {total_images}")
    print(f"验证集数量: {val_count}")
    print(f"训练集数量: {total_images - val_count}")
    
    # 随机选择验证集图片
    val_images = random.sample(all_images, val_count)
    val_image_ids = {img['id'] for img in val_images}
    
    # 分离训练集和验证集的图片
    train_images = [img for img in all_images if img['id'] not in val_image_ids]
    
    # 分离标注信息
    val_annotations = []
    train_annotations = []
    
    for ann in annotations['annotations']:
        if ann['image_id'] in val_image_ids:
            val_annotations.append(ann)
        else:
            train_annotations.append(ann)
    
    # 创建验证集标注文件
    val_annotations_data = {
        'info': annotations.get('info', {}),
        'licenses': annotations.get('licenses', []),
        'categories': annotations.get('categories', []),
        'images': val_images,
        'annotations': val_annotations
    }
    
    # 创建新的训练集标注文件
    train_annotations_data = {
        'info': annotations.get('info', {}),
        'licenses': annotations.get('licenses', []),
        'categories': annotations.get('categories', []),
        'images': train_images,
        'annotations': train_annotations
    }
    
    # 复制验证集图片
    print("正在复制验证集图片...")
    for img in val_images:
        src_path = os.path.join(train_images_dir, img['file_name'])
        dst_path = os.path.join(val_images_dir, img['file_name'])
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
        else:
            print(f"警告: 图片文件不存在: {src_path}")
    
    # 保存验证集标注文件
    print(f"正在保存验证集标注文件: {val_annotations_file}")
    with open(val_annotations_file, 'w', encoding='utf-8') as f:
        json.dump(val_annotations_data, f, ensure_ascii=False, indent=2)
    
    # 保存新的训练集标注文件
    new_train_annotations_file = train_annotations_file.replace('.json', '_split.json')
    print(f"正在保存新的训练集标注文件: {new_train_annotations_file}")
    with open(new_train_annotations_file, 'w', encoding='utf-8') as f:
        json.dump(train_annotations_data, f, ensure_ascii=False, indent=2)
    
    print("数据集分割完成！")
    print(f"验证集图片目录: {val_images_dir}")
    print(f"验证集标注文件: {val_annotations_file}")
    print(f"新训练集标注文件: {new_train_annotations_file}")


def main():
    parser = argparse.ArgumentParser(description='将训练集分割为训练集和验证集')
    parser.add_argument('--val-ratio', type=float, default=0.2, help='验证集比例')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not os.path.exists(args.train_images):
        print(f"错误: 训练图片目录不存在: {args.train_images}")
        return
    
    if not os.path.exists(args.train_annotations):
        print(f"错误: 训练标注文件不存在: {args.train_annotations}")
        return
    
    # 执行分割
    split_dataset(
        train_images_dir='train/images',
        train_annotations_file='train/train.json',
        val_ratio=args.val_ratio,
        seed=args.seed
    )


if __name__ == '__main__':
    main() 