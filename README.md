# 移动应用漂浮窗目标检测项目
## 任务描述
检测移动应用漂浮窗的无障碍性，实现对移动应用截图中的漂浮窗、关闭按钮"X"号、文字按钮的检测与分类
- **输入**：移动应用屏幕截图
- **输出**：漂浮窗、关闭叉号按钮、文字按钮的定位坐标和分类，结果以 COCO 格式保存

类别对应：
- 0：漂浮窗（floating_window）
- 1：关闭叉号（X_mark）
- 2：文字按钮（text_button）


## 项目结构
```
│
├── data/                # 数据集目录
│   ├── train/
│   └── test/
├── rcnn/                # Faster R-CNN 相关代码
│   ├── dataset.py
│   ├── main.py     
│   ├── model.py      
│   ├── predict.py
│   ├── train.py
│   └── README.md
├── yolo/                # YOLO 相关代码
│   ├── coco_to_yolo_converter.py
│   ├── data.yaml
│   ├── predict.py
│   ├── split_dataset.py
│   ├── train.py
│   └── README.md
├── requirements.txt     # 依赖包列表
└── README.md            # 项目说明文档
```


## 环境依赖
```bash
pip install -r requirements.txt
```


## Faster R-CNN 项目
基于 PyTorch 和 torchvision，
### 训练并预测
```bash
python rcnn/main.py --mode train --train_mode kfold --epochs 20 --batch_size 4
python rcnn/main.py --mode predict --score_thresh 0.5
```


## YOLOv8 项目
本项目基于 YOLOv8 实现自定义数据集的目标检测训练、推理与评估，支持 COCO 与 YOLO 格式标注互转

### 数据集配置（yolo/data.yaml）
```yaml
# 数据集路径配置
path: ./
train: data/train/images
val: data/val/images
test: data/test/images

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
python yolo/coco_to_yolo_converter.py
```

#### 训练并预测
```bash
python yolo/train.py --epochs 50 --batch_size 10
python yolo/predict.py --model model_yolo/train/weights/best.pt --batch_size 100
```
