# SVM航空器图像分类系统
## 项目结构
将 `test` `train` 文件夹放至 `dataset` 文件夹下
```
├─dataset
│  ├─cache
│  ├─test
│  └─train
│
├─models
├─results
└─src
    └─utils
```

## 环境要求
- Python 3.6+
- 安装依赖 `pip install -r requirements.txt`

## 数据集
- 训练集：19,569张图像，分辨率224×224，9种航空器类型
- 测试集：2,305张图像，不固定分辨率，9种航空器类型

## 使用方法
为节省时间，本项目将经CNN提取到的特征存至 `dataset/cache` 文件夹下，将训练好的模型存至 `models` 文件夹下  

### 训练模型
```cmd
python src/train.py --config config.yaml
```

### 测试模型
```cmd
python src/test.py --config config.yaml
```

### 完整流程
```cmd
.\run.bat
```
