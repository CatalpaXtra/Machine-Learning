@echo off
echo ========================================================================
echo                 PyTorch Aircraft Image Classification System
echo ========================================================================

:: 确保目录存在
if not exist models mkdir models

:: 训练CNN+SVM模型
echo train CNN+SVM
python src/main.py train --config config.yaml

:: 评估模型
echo evaluate CNN+SVM
python src/main.py test --config config.yaml

echo DONE
echo see results in results folder
pause