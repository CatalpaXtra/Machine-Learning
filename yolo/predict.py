import argparse
import os
import json
import torch
from ultralytics import YOLO


def predict_and_fill_annotations(model_path, batch_size, test_images_dir, pred_json_path, conf=0.25, imgsz=640):
    print(f"加载模型: {model_path}")
    
    # 检查CUDA可用性并设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    model = YOLO(model_path)
    
    # 加载pred.json文件
    print(f"加载pred.json文件: {pred_json_path}")
    with open(pred_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 创建文件名到image_id的映射
    image_id_mapping = {}
    for img in data['images']:
        image_id_mapping[img['file_name']] = img['id']
    
    # 获取所有测试图片路径
    test_images = []
    for img_info in data['images']:
        img_path = os.path.join(test_images_dir, img_info['file_name'])
        if os.path.exists(img_path):
            test_images.append(img_path)
        else:
            print(f"警告: 图片文件不存在: {img_path}")
    
    print(f"开始对 {len(test_images)} 张图片进行推理...")
    
    # 分批处理以避免内存不足
    all_results = []
    for i in range(0, len(test_images), batch_size):
        batch_images = test_images[i:i + batch_size]
        print(f"处理批次 {i//batch_size + 1}/{(len(test_images) + batch_size - 1)//batch_size}: {len(batch_images)} 张图片")
        
        # 批量推理
        batch_results = model.predict(
            source=batch_images,
            conf=conf,
            imgsz=imgsz,
            save=False,
            verbose=False,
            device=device,
            iou=0.5,
            augment=True  # 启用TTA
        )
        all_results.extend(batch_results)
        
    
    # 处理预测结果
    annotations = []
    annotation_id = 0
    processed_count = 0
    
    for result in all_results:
        if result.boxes is not None:
            boxes = result.boxes
            image_path = result.path
            image_name = os.path.basename(image_path)
            
            # 获取对应的image_id
            if image_name in image_id_mapping:
                image_id = image_id_mapping[image_name]
                processed_count += 1
                
                for j in range(len(boxes)):
                    # 获取边界框坐标 (YOLO格式: [x1, y1, x2, y2])
                    bbox_xyxy = boxes.xyxy[j].cpu().numpy()
                    
                    # 转换为COCO格式 [x, y, width, height]
                    x = float(bbox_xyxy[0])
                    y = float(bbox_xyxy[1])
                    width = float(bbox_xyxy[2] - bbox_xyxy[0])
                    height = float(bbox_xyxy[3] - bbox_xyxy[1])
                    
                    # 获取置信度分数
                    score = float(boxes.conf[j].cpu().numpy())
                    
                    # 获取类别ID
                    category_id = int(boxes.cls[j].cpu().numpy())
                    
                    # 创建annotation对象
                    annotation = {
                        "image_id": image_id,
                        "category_id": category_id,
                        "bbox": [x, y, width, height],
                        "score": score
                    }
                    
                    annotations.append(annotation)
                    annotation_id += 1
            else:
                print(f"警告: 无法找到图片 {image_name} 对应的image_id")
    
    print(f"成功处理 {processed_count}/{len(test_images)} 张图片")
    
    # 保存更新后的pred.json文件
    data['annotations'] = annotations
    with open(pred_json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"预测完成！总共检测到 {len(annotations)} 个目标")
    print(f"结果已保存到: {pred_json_path}")


def main():
    parser = argparse.ArgumentParser(description='YOLO测试集推理脚本')
    parser.add_argument('--model', type=str, required=True, help='模型路径')
    parser.add_argument('--batch_size', type=int, default=100, help='批次大小')
    parser.add_argument('--imgsz', type=int, default=640, help='图像尺寸')
    args = parser.parse_args()
    
    predict_and_fill_annotations(
        model_path=args.model,
        batch_size=args.batch_size,
        test_images_dir='test/images',
        pred_json_path='test/pred.json',
    )


if __name__ == '__main__':
    main() 