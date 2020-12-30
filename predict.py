import os
import argparse

import torch
from torch.utils.data import DataLoader

from solver import Solver
from dataset import OuluDataset


def get_args():
    parser = argparse.ArgumentParser(description='DLCV Final Project!?')

    # dataset args
    parser.add_argument('--num_workers',      type=int,   default=8)
    parser.add_argument('--test_dir',         type=str,   default='oulu_npu_cropped/test')

    parser.add_argument('--image_dim',        type=int,   default=112)
    parser.add_argument('--limit_num',        type=int,   default=None)
    parser.add_argument('--batch_size',       type=int,   default=32)

    # testing setting
    parser.add_argument('--name',             type=str,   default='_')
    parser.add_argument('--load_checkpoint',  type=str,   default=None)
    parser.add_argument('--output_csv',       type=str,   default='output/_.csv')

    parser.add_argument('--lr',               type=float, default=1e-4)
    parser.add_argument('--total_epoch',      type=int,   default=0)
    parser.add_argument('--checkpoint_epoch', type=int,   default=0)

    return parser.parse_args()


def main():
    args = get_args()
    assert args.load_checkpoint is not None, '--load_checkpoint can not be None'

    test_set = OuluDataset(args.test_dir, False, args.image_dim, args.limit_num, return_label=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=args.num_workers)

    solver = Solver(args)
    predict = solver.predict(test_loader)

    with open(args.output_csv, 'w') as f:
        f.write('video_id,label\n')
        for video_id, label in zip(test_set.sub_dirs, predict):
            f.write(f'{video_id},{label}\n')


if __name__ == '__main__':
    main()
