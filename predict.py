import os
import argparse
import numpy as np

import torch
from torch.utils.data import DataLoader

from solver import Solver
from dataset import *


def get_args():
    parser = argparse.ArgumentParser(description='DLCV Final Project!?')

    # dataset args
    parser.add_argument('--num_workers',      type=int,   default=8)
    parser.add_argument('--test_dir',         type=str,   default='oulu_npu_cropped/test')

    parser.add_argument('--image_dim',        type=int,   default=112)
    parser.add_argument('--limit_num',        type=int,   default=None)
    parser.add_argument('--batch_size',       type=int,   default=32)

    parser.add_argument('--texture',          action='store_true')
    parser.add_argument('--spec',             action='store_true')
    parser.add_argument('--category',         action='store_true')

    # testing setting
    parser.add_argument('--name',             type=str,   default='_')
    parser.add_argument('--load_checkpoint',  type=str,   default=None)
    parser.add_argument('--output_csv',       type=str,   default='output/_.csv')

    parser.add_argument('--model',            type=str,   default='resnet3d')

    parser.add_argument('--lr',               type=float, default=1e-4)
    parser.add_argument('--warmup',           type=int,   default=1)
    parser.add_argument('--total_epoch',      type=int,   default=0)
    parser.add_argument('--checkpoint_epoch', type=int,   default=0)

    return parser.parse_args()


def main():
    args = get_args()
    assert args.load_checkpoint is not None, '--load_checkpoint can not be None'

    if args.spec:
        assert args.texture, '--texture must be true when setting --spec'
        FaceDataset = SpecDataset
    elif args.texture:
        FaceDataset = TextureDataset
    else:
        FaceDataset = VideoDataset

    test_set = FaceDataset(args.test_dir, False, args.image_dim, 10, five_crop=args.texture, return_label=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=args.num_workers)

    print(f'Running prediction on {len(test_set)} samples ...')

    solver = Solver(args)
    if args.category:
        predict = solver.predict(test_loader, category=True)
        predict = np.array(predict)
        predict[predict == 2] = 1
        predict[predict >= 3] = 2

        if args.texture:
            predict = np.reshape(predict, (-1, 10))
            predict = np.apply_along_axis(lambda x: np.bincount(x, minlength=3), axis=1, arr=predict)
            predict = np.argmax(predict, axis=1)

    else:
        predict = solver.predict(test_loader)
        predict = np.array(predict)
        if args.texture:
            predict = np.reshape(predict, (-1, 10))
            predict = np.mean(predict, axis=1)

    with open(args.output_csv, 'w') as f:
        f.write('video_id,label\n')
        for video_id, label in zip(test_set.sub_dirs, predict):
            f.write(f'{video_id},{label}\n')


if __name__ == '__main__':
    main()
