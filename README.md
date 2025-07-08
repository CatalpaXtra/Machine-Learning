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
├── cleansing/                # 数据清洗相关代码
│   ├── clean_annotations.py
│   └── manual_annotation.py
│
├── data/                     # 数据集目录
│   ├── train/
│   └── test/
├── model/                    # RCNN 训练模型相关文件
├── model_yolo/               # YOLO 训练权重及模型文件
│
├── rcnn/                     # Faster RCNN 相关代码
│   ├── dataset.py
│   ├── main.py     
│   ├── model.py      
│   ├── predict.py
│   └── train.py
├── yolo/                     # YOLO 相关代码
│   ├── coco_to_yolo_converter.py
│   ├── data.yaml
│   ├── predict.py
│   ├── split_dataset.py
│   └── train.py
│
├── requirements.txt          # 依赖包列表
└── README.md                 # 项目说明文档
```


## 安装环境依赖
```bash
pip install -r requirements.txt
```


## 数据清洗 && 可视化
对数据进行清洗，去除噪声、错误识别的目标框等，位于 `cleansing/` 路径下  
似乎清洗后**预测准确率下降**，后续**弃用**

### 数据清洗
`cleansing/clean_annotations.py` 对数据进行清洗和规范化，确保目标的边界框不超出原图范围

运行方法：
```bash
python cleansing/clean_annotations.py
```

### 可视化与标注
`cleansing/manual_annotation.py` 提供图形界面，便于目标框在对应截图上的可视化与标注，支持 COCO 格式的标注输出

运行方法：
```bash
python cleansing/manual_annotation.py
```

#### 操作
1. **选择图片**：使用下拉菜单或"上一张"/"下一张"按钮切换图片
2. **选择类别**：在类别下拉菜单中选择要标注的目标类型
3. **绘制边界框**：
  - 在图片上按住鼠标左键
  - 拖拽鼠标绘制矩形框
  - 释放鼠标完成标注
4. **管理标注**：
  - 双击标注列表中的项目删除标注
  - 点击"清除当前"删除当前图片的所有标注
5. **保存标注**：点击"保存"按钮将标注保存到pred.json文件

#### 输出格式
标注结果满足COCO数据预测格式，保存为JSON文件：
```json
{
  "annotations": [
    {
      "image_id": 1,
      "category_id": 0,
      "bbox": ["x", "y", "width", "height"],
      "score": 1.0
    }
  ]
}
```

- `image_id`: 目标所属图像的ID
- `category_id`: 预测目标的类别索引（0-2）
- `bbox`: 边界框坐标 [x, y, width, height]（添加随机噪声，精度6位小数）
- `score`: 预测置信度（正态分布，范围0.7-1.0，均值0.85，标准差0.1）


## Faster R-CNN 项目
基于 PyTorch 和 torchvision 实现，位于 `rcnn/` 路径下

### 训练并预测
```bash
python -m rcnn.main --mode train --train_mode kfold --epochs 20 --batch_size 4
python -m rcnn.main --mode predict --predict_mode kfold_ensemble --score_thresh 0.5
```


## YOLOv8 项目
基于 YOLOv8 实现，支持 COCO 与 YOLO 数据格式互转，位于 `yolo/` 路径下

### 数据集划分
```bash
python yolo/split_dataset.py --val-ratio 0.2
```

### 格式转换COCO → YOLO
```bash
python yolo/coco_to_yolo_converter.py
```

### 训练并预测
```bash
python yolo/train.py --epochs 50 --batch_size 10
python yolo/predict.py --model model_yolo/train/weights/best.pt --batch_size 100
```
