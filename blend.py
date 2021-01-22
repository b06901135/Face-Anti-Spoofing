import os
import argparse
import numpy as np
import pandas as pd


def get_args():
    parser = argparse.ArgumentParser('Blend models!')

    parser.add_argument('--category', action='store_true')
    parser.add_argument('--tag',    type=str, required=True)
    parser.add_argument('--output', type=str, required=True)

    return parser.parse_args()


def geo_mean(arr):
    arr[arr < 1e-16] = 1e-16
    a = np.log(arr)
    return np.exp(a.mean())


def blend(csv_files, out_file, category=False):
    predicts = []
    for i, csv_file in enumerate(csv_files):
        df = pd.read_csv(csv_file, dtype={'video_id': str})
        if i == 0:
            videos_id = df['video_id']
        predict = df['label'].to_numpy(dtype=np.float if not category else np.int)
        predicts.append(predict)

    predicts = np.array(predicts)
    if category:
        predicts = np.swapaxes(predicts, 0, 1)
        predicts = np.apply_along_axis(lambda x: np.bincount(x, minlength=3), axis=1, arr=predicts)
        predicts = np.argmax(predicts, axis=1)
    else:
        predicts = np.apply_along_axis(lambda x: geo_mean(x), axis=0, arr=predicts)

    with open(out_file, 'w') as f:
        f.write('video_id,label\n')
        for video_id, label in zip(videos_id, predicts):
            f.write(f'{video_id},{label}\n')


if __name__ == '__main__':
    args = get_args()
    print(f'Category: {args.category}')

    models = [
        '01_resnet18',
        '02_resnet50',
        '03_vgg11',
        '04_vgg16',
        '05_vgg19'
    ]

    csv_files = [f'output/{model}{args.tag}.csv' for model in models]
    blend(csv_files, args.output, category=args.category)
