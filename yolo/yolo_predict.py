"""
YOLOv8 推理脚本模板
支持单张图片或文件夹批量预测
"""
from ultralytics import YOLO
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description='YOLOv8 Inference Script')
    parser.add_argument('--model', type=str, required=True, help='Path to trained YOLOv8 model (如yolov8s.pt或runs/detect/exp/weights/best.pt)')
    parser.add_argument('--source', type=str, required=True, help='Image file or directory for prediction')
    parser.add_argument('--save', action='store_true', help='Save results to file')
    args = parser.parse_args()

    model = YOLO(args.model)
    # 判断是单张图片还是文件夹
    if os.path.isdir(args.source):
        results = model.predict(source=args.source, save=args.save)
    else:
        results = model.predict(source=args.source, save=args.save)
    print('Prediction completed.')

if __name__ == '__main__':
    main() 