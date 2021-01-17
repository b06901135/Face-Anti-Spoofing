import os
import numpy as np
import pandas as pd


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
        # print(predict[:5])
        predicts.append(predict)

    predicts = np.array(predicts)
    # print(predicts[:3])
    # print(predicts)
    # print(predicts.shape)
    if category:
        predicts = np.swapaxes(predicts, 0, 1)
        predicts = np.apply_along_axis(lambda x: np.bincount(x, minlength=3), axis=1, arr=predicts)
        predicts = np.argmax(predicts, axis=1)
    else:
        print(predicts.shape)
        print(predicts[:, :4])
        # predicts = np.mean(predicts, axis=0)
        predicts = np.apply_along_axis(lambda x: geo_mean(x), axis=0, arr=predicts)
        print(predicts[:4])
    # print(predicts)
    # print(predicts.shape)

    # (unique, counts) = np.unique(predicts, return_counts=True)
    # frequencies = np.asarray((unique, counts)).T
    # print(frequencies)

    with open(out_file, 'w') as f:
        f.write('video_id,label\n')
        for video_id, label in zip(videos_id, predicts):
            f.write(f'{video_id},{label}\n')


def test():
    df = pd.read_csv('output/test_cat.csv')
    p = df['label'].to_numpy(dtype=np.int)
    (unique, counts) = np.unique(p, return_counts=True)
    frequencies = np.asarray((unique, counts)).T
    print(frequencies)


if __name__ == '__main__':
    models = [
        '01_resnet18',
        '02_resnet50',
        '03_vgg11',
        '04_vgg16',
        '05_vgg19',
        '06_resnet3d',
        '07_resnet_mc3',
        '08_resnet_r21d'
    ]

    for tag in ['', '_siw', '_cat']:
        print(f'Blending {tag}')
        csv_files = [f'output/{model}{tag}.csv' for model in models]
        blend(csv_files, f'output/blend_03{tag}.csv', tag == '_cat')
