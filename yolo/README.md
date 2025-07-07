## YOLOv8 目标检测项目
本项目基于 YOLOv8 实现自定义数据集的目标检测训练、推理与评估，支持 COCO 与 YOLO 格式标注互转

### 数据集配置（yolo/data.yaml）
```yaml
# 数据集路径配置
path: ./
train: train/images
val: val/images
test: test/images

# 类别配置
nc: 3  # 类别数量
names: ['floating_window', 'X_mark', 'text_button']
```

### 使用方法
#### 数据集划分
```bash
python yolo/split_dataset.py --val-ratio 0.2
```

#### 格式转换COCO → YOLO
```bash
python yolo/coco_to_yolo_converter.py --input train/train.json --output train
python yolo/coco_to_yolo_converter.py --input val/val.json --output val
```

#### 训练并预测
```bash
python yolo/train.py --epochs 50 --batch_size 10
python yolo/predict.py --model runs/train/weights/best.pt --batch_size 100
```


