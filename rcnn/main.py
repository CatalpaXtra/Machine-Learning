import argparse
from rcnn.train import main as train_main
from rcnn.predict import main as predict_main


def parse_args():
    parser = argparse.ArgumentParser(description='Floating Window Detection')
    parser.add_argument('--mode', choices=['train', 'predict'], required=True, help='train or predict')
    parser.add_argument('--train_mode', choices=['normal', 'with_val', 'kfold'], default='kfold', help='Train mode')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=5, help='Batch size for training')
    parser.add_argument('--patience', type=int, default=3, help='Early stopping patience for training')
    parser.add_argument('--score_thresh', type=float, default=0.5, help='Score threshold for prediction')
    return parser.parse_args()


def main():
    args = parse_args()
    if args.mode == 'train':
        train_main(train_mode=args.train_mode, epochs=args.epochs, batch_size=args.batch_size, patience=args.patience)
    elif args.mode == 'predict':
        predict_main(score_thresh=args.score_thresh)


if __name__ == '__main__':
    main() 