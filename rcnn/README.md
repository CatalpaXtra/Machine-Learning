# 移动应用漂浮窗目标检测项目
基于 PyTorch 和 torchvision，实现对移动应用截图中的漂浮窗、关闭按钮"X"号、文字按钮的检测与分类
输入为移动应用屏幕截图，输出为三类目标的定位坐标和分类，结果以 COCO 格式保存

## 任务描述
- **输入**：移动应用屏幕截图
- **输出**：漂浮窗、关闭叉号按钮、文字按钮的定位坐标和分类（COCO格式）

类别对应：
- 0：漂浮窗（floating_window）
- 1：关闭叉号（X_mark）
- 2：文字按钮（text_button）

## 数据结构
```
data/
├── train/
│   ├── images/         # 训练图片
│   └── train.json      # 训练集COCO标注
└── test/
    ├── images/         # 测试图片
    └── pred.json       # 预测结果（COCO格式，需填充annotations字段）
```

## 项目结构
```
```

## 安装依赖
```bash
pip install -r requirements.txt
```

## 使用方法
#### 训练并预测
```bash
python rcnn/main.py --mode train --train_mode kfold --epochs 20 --batch_size 4
python rcnn/main.py --mode predict --score_thresh 0.5
```

