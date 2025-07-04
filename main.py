import argparse
from train import train
from predict import predict


def parse_args():
    parser = argparse.ArgumentParser(description='Floating Window Detection')
    parser.add_argument('--mode', choices=['train', 'predict'], required=True, help='train or predict')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=5, help='Batch size for training')
    parser.add_argument('--score_thresh', type=float, default=0.5, help='Score threshold for prediction')
    return parser.parse_args()


def main():
    args = parse_args()
    if args.mode == 'train':
        train(epochs=args.epochs, batch_size=args.batch_size)
    elif args.mode == 'predict':
        predict(score_thresh=args.score_thresh)


if __name__ == '__main__':
    main() 