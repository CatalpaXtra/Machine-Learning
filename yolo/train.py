from ultralytics import YOLO
import argparse


def train_model(args):
    print("开始训练模型...")
    
    # 创建模型
    model = YOLO(args.model)
    
    # 开始训练
    results = model.train(
        data='yolo/data.yaml',
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch_size,
        patience=20,
        save=True,
        project='model_yolo',
        name='train',
        
        augment=True,      # 启用数据增强
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.2,
        translate=0.1,
        scale=0.5,
        shear=0.01,
        flipud=0.5,
        fliplr=0.5,
        mosaic=0.5, # good
        mixup=0.1   # good
    )
    
    print(f"训练完成！模型保存在: {results.save_dir}")
    return results


def main():
    parser = argparse.ArgumentParser(description='YOLOv8 训练和推理脚本')
    parser.add_argument('--model', type=str, default='yolov8s.pt', help='YOLOv8模型类型')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--imgsz', type=int, default=640, help='图像尺寸')
    parser.add_argument('--batch_size', type=int, default=10, help='批次大小')
    args = parser.parse_args()
    
    train_model(args)

if __name__ == '__main__':
    main() 