import json
import argparse
from pathlib import Path


def convert_coco_to_yolo(coco_json_path, output_dir):
    with open(coco_json_path, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)
    
    # 创建输出目录
    labels_dir = Path(output_dir) / 'labels'
    labels_dir.mkdir(exist_ok=True)
    
    # 创建文件名到ID的映射
    file_to_id = {img['file_name']: img['id'] for img in coco_data['images']}
    
    # 按图像分组标注
    annotations_by_image = {}
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        if image_id not in annotations_by_image:
            annotations_by_image[image_id] = []
        annotations_by_image[image_id].append(ann)
    
    # 统计信息
    total_annotations = 0
    valid_annotations = 0
    skipped_annotations = 0
    
    # 为每个图像创建YOLO格式的标注文件
    for img in coco_data['images']:
        image_id = img['id']
        file_name = img['file_name']
        width = img['width']
        height = img['height']
        
        # 创建对应的标签文件名
        label_file = labels_dir / f"{Path(file_name).stem}.txt"
        
        if image_id in annotations_by_image:
            with open(label_file, 'w') as f:
                for ann in annotations_by_image[image_id]:
                    total_annotations += 1
                    category_id = ann['category_id']
                    bbox = ann['bbox']  # [x, y, width, height]
                    
                    # 验证类别ID
                    if category_id < 0 or category_id > 2:
                        print(f"警告: 图像 {file_name} 中的类别ID {category_id} 超出范围 [0-2]，跳过")
                        skipped_annotations += 1
                        continue
                    
                    # 验证边界框坐标
                    x, y, w, h = bbox
                    if x < 0 or y < 0 or w <= 0 or h <= 0:
                        print(f"警告: 图像 {file_name} 中的边界框坐标无效: {bbox}，跳过")
                        skipped_annotations += 1
                        continue
                    
                    # 检查边界框是否超出图像边界
                    if x + w > width or y + h > height:
                        print(f"警告: 图像 {file_name} 中的边界框超出图像边界: {bbox} (图像尺寸: {width}x{height})，跳过")
                        skipped_annotations += 1
                        continue
                    
                    # 转换为YOLO格式 [x_center, y_center, width, height] (归一化)
                    x_center = (x + w / 2) / width
                    y_center = (y + h / 2) / height
                    w_norm = w / width
                    h_norm = h / height
                    
                    # 验证归一化后的坐标
                    if x_center < 0 or x_center > 1 or y_center < 0 or y_center > 1 or w_norm <= 0 or h_norm <= 0:
                        print(f"警告: 图像 {file_name} 中归一化后的坐标无效: [{x_center:.6f}, {y_center:.6f}, {w_norm:.6f}, {h_norm:.6f}]，跳过")
                        skipped_annotations += 1
                        continue
                    
                    f.write(f"{category_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")
                    valid_annotations += 1
    
    print(f"COCO到YOLO转换完成:")
    print(f"  总标注数: {total_annotations}")
    print(f"  有效标注数: {valid_annotations}")
    print(f"  跳过标注数: {skipped_annotations}")
    print(f"  输出目录: {labels_dir}")
    
    return {
        'total': total_annotations,
        'valid': valid_annotations,
        'skipped': skipped_annotations,
        'output_dir': str(labels_dir)
    }


def main():
    parser = argparse.ArgumentParser(description='COCO到YOLO格式转换工具')
    parser.add_argument('--input', type=str, required=True, help='COCO格式标注文件路径')
    parser.add_argument('--output', type=str, required=True, help='输出目录路径')
    
    args = parser.parse_args()
    
    # 执行转换
    print(f"\n开始转换: {args.input} -> {args.output}")
    try:
        result = convert_coco_to_yolo(args.input, args.output)
        print(f"\n转换成功完成!")
        print(f"转换结果: {result}")
    except Exception as e:
        print(f"转换失败: {e}")


if __name__ == '__main__':
    main() 