from utils.args import parse_args
from train import main as train_main
from test import main as test_main

def main():
    """主程序入口"""
    args = parse_args()
    
    if args.command == 'train':
        train_main(args)
    elif args.command == 'test':
        test_main(args)

if __name__ == '__main__':
    main()