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


# 数据清洗 && 可视化工具
这是一个用于手动标注移动应用截图的Python工具，支持COCO格式的标注输出。

## 功能特点

- 图形化界面，易于使用
- 支持三种类别标注：
  - 0: 漂浮窗（floating_window）
  - 1: 关闭叉号（X_mark）
  - 2: 文字按钮（text_button）
- 鼠标拖拽绘制边界框
- 实时预览标注结果
- 支持标注的增删改查
- 自动保存为COCO格式
- **边界框坐标添加随机噪声**（小数点后6位精度）
- **Score值使用正态分布**（0.7-1.0之间，均值0.85，标准差0.1）


## 使用方法

1. 确保项目结构如下：
```
├── data/
│   ├── images/          # 待标注的图片
│   └── pred.json        # COCO格式的预测文件
├── manual_annotation.py # 标注工具
└── requirements.txt     # 依赖文件
```

2. 运行标注工具：
```bash
python cleansing/clean_annotations.py
python cleansing/manual_annotation.py
```

## 操作说明

### 界面布局
- **控制面板**：图片选择、类别选择、导航按钮
- **图片标注区域**：显示图片和绘制边界框的画布
- **当前图片标注**：显示当前图片的所有标注列表
- **状态栏**：显示当前操作状态

### 标注操作
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

### 快捷键
- 鼠标左键拖拽：绘制边界框
- 双击标注列表：删除选中的标注

## 输出格式

标注结果将保存为COCO格式的JSON文件，包含以下字段：

```json
{
  "annotations": [
    {
      "image_id": 1,
      "category_id": 0,
      "bbox": [x, y, width, height],
      "score": 1.0
    }
  ]
}
```

- `image_id`: 目标所属图像的ID
- `category_id`: 预测目标的类别索引（0-2）
- `bbox`: 边界框坐标 [x, y, width, height]（添加随机噪声，精度6位小数）
- `score`: 预测置信度（正态分布，范围0.7-1.0，均值0.85，标准差0.1）

## 注意事项

1. 确保test/images目录存在且包含待标注的图片
2. 确保test/pred.json文件存在且包含正确的images字段
3. 标注工具会自动调整图片大小以适应显示窗口
4. 边界框坐标基于显示后的图片尺寸
5. 建议定期保存标注结果，避免数据丢失

## 故障排除

- **找不到图片目录**：检查test/images目录是否存在
- **找不到pred.json**：检查test/pred.json文件是否存在
- **图片加载失败**：检查图片文件是否损坏或格式不支持
- **保存失败**：检查文件权限和磁盘空间 