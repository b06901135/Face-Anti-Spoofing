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
    parser.add_argument('--score',            action='store_true')

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


def geo_mean(a, axis=None):
    a[a < 1e-16] = 1e-16
    a = np.log(a)
    a = a.mean(axis=axis)
    a = np.exp(a)
    return a


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

    solver = Solver(args, weights_only=True)
    if args.score:
        predict = solver.predict(test_loader, return_score=True)
        predict = np.array(predict)
        predict[:, 1] = geo_mean(predict[:, 1:3], axis=1)
        predict[:, 2] = geo_mean(predict[:, 3: ], axis=1)
        predict = predict[:, :3]

        if args.texture:
            predict = np.reshape(predict, (-1, 10, 3))
            predict = np.mean(predict, axis=1)

        temp = np.zeros(predict.shape[0], np.int)
        temp_set = set(list(range(len(temp))))

        count = np.array([0, 0, 0], np.int)
        limit = (np.array([0.29105, 0.16422, 0.54471]) * len(temp)).astype(np.int)

        while len(temp_set) > 0:
            for i in range(3):
                while True:
                    if count[i] <= limit[i]:
                        idx = np.argmax(predict[:, i])
                        predict[idx, i] = 0
                        if idx in temp_set:
                            temp[idx] = i
                            temp_set.remove(idx)
                            count[i] += 1
                            break
                    else:
                        break

        predict = temp

    elif args.category:
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
            # predict = np.mean(predict, axis=1)
            predict = geo_mean(predict, axis=1)

    with open(args.output_csv, 'w') as f:
        f.write('video_id,label\n')
        for video_id, label in zip(test_set.sub_dirs, predict):
            f.write(f'{video_id},{label}\n')


def test():
    test_set = TextureDataset('siw_test')
    for i in range(3):
        predict = np.ones(len(test_set.sub_dirs), np.int) * i

        with open(f'output/all_{i}_cat.csv', 'w') as f:
            f.write('video_id,label\n')
            for video_id, label in zip(test_set.sub_dirs, predict):
                f.write(f'{video_id},{label}\n')


if __name__ == '__main__':
    main()
    # test()
