import argparse
from train import main as train_main
from test import main as test_main

def parse_args():
    parser = argparse.ArgumentParser(description='图像分类训练和测试')
    parser.add_argument('mode', choices=['train', 'test'], help='运行模式：train或test')
    parser.add_argument('--config', type=str, default='config.yaml', help='配置文件路径')
    return parser.parse_args()

def main():
    """主程序入口"""
    args = parse_args()
    
    if args.mode == 'train':
        train_main()
    else:
        test_main()

if __name__ == '__main__':
    main()