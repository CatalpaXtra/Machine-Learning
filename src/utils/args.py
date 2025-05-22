import argparse

def create_parser():
    """创建命令行参数解析器"""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command')
    
    # 训练命令
    train_parser = subparsers.add_parser('train', help='训练模型')
    train_parser.add_argument('--config', default='config.yaml', help='配置文件路径')
    
    # 测试命令
    test_parser = subparsers.add_parser('test', help='测试模型')
    test_parser.add_argument('--config', default='config.yaml', help='配置文件路径')
    
    return parser


def parse_args():
    """解析命令行参数"""
    parser = create_parser()
    return parser.parse_args() 