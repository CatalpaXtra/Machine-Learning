@echo off
echo ========================================================================
echo                 PyTorch Aircraft Image Classification System
echo ========================================================================

:: 训练模型
python src/main.py train

:: 评估模型
python src/main.py test

echo DONE
echo see results in results folder
pause