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
│
├── main.py           # 主入口，命令行参数选择训练或推理
├── dataset.py        # 数据集定义与collate_fn
├── model.py          # 检测模型构建
├── train.py          # 训练流程
├── predict.py        # 推理与结果保存
├── requirements.txt  # 依赖包
├── README.md         # 项目说明
└── data/
    ├── train/
    └── test/
```

## 安装依赖
建议使用 Python 3.7+，安装依赖：
```bash
pip install -r requirements.txt
```

## 使用方法
### 训练模型
```bash
python main.py --mode train --epochs 10 --batch_size 5
```

### 推理并生成COCO格式结果
```bash
python main.py --mode predict --score_thresh 0.5
```

## 其他说明
- 训练完成后模型权重会保存在 `model.pth`
- 若需自定义数据路径、模型参数等，可在各脚本中进一步修改

