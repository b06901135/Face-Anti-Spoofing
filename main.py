import torch
from torch.utils.data import DataLoader

import argparse

from solver import Solver
from dataset import OuluDataset


def get_args():
    parser = argparse.ArgumentParser(description='DLCV Final Project!?')

    # dataset args
    parser.add_argument('--num_workers',      type=int,   default=8)
    parser.add_argument('--train_dir',        type=str,   default='oulu_npu_cropped/train')
    parser.add_argument('--val_dir',          type=str,   default='oulu_npu_cropped/val')

    parser.add_argument('--augment',          type=int,   default=1)
    parser.add_argument('--image_dim',        type=int,   default=112)
    parser.add_argument('--limit_num',        type=int,   default=None)
    parser.add_argument('--batch_size',       type=int,   default=32)

    # training setting
    parser.add_argument('--name',             type=str,   default='_')
    parser.add_argument('--load_checkpoint',  type=str,   default=None)

    parser.add_argument('--lr',               type=float, default=1e-4)
    parser.add_argument('--warmup',           type=int,   default=1)
    parser.add_argument('--total_epoch',      type=int,   default=200)
    parser.add_argument('--checkpoint_epoch', type=int,   default=20)

    return parser.parse_args()


def main():
    args = get_args()

    train_set = OuluDataset(args.train_dir, args.augment, args.image_dim, args.limit_num)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=args.num_workers)

    val_set = OuluDataset(args.val_dir, False, args.image_dim, args.limit_num)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=args.num_workers)

    solver = Solver(args)
    solver.train(train_loader, val_loader)


if __name__ == '__main__':
    main()
