"""
YOLOv8 训练脚本模板
适用于COCO格式数据集
"""
from ultralytics import YOLO
import argparse

def main():
    parser = argparse.ArgumentParser(description='YOLOv8 Training Script')
    parser.add_argument('--data', type=str, required=True, help='Path to data yaml (COCO格式)')
    parser.add_argument('--model', type=str, default='yolov8s.pt', help='YOLOv8 model type (yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    args = parser.parse_args()

    model = YOLO(args.model)
    model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch)

if __name__ == '__main__':
    main() 