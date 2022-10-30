import os.path as osp

import cv2
import numpy as np
import torch
import csv
from torch.utils.data import Dataset


class DataLoader(Dataset):
    def __init__(self, data_path, max_depth=-1, vertical=False) -> None:

        try:
            csv_file = osp.join(data_path, 'data.csv')
            data_reader = csv.reader(open(csv_file, 'r'))
            self.data_path = data_path
        except:
            csv_file = osp.join(data_path, 'camera/data.csv')
            data_reader = csv.reader(open(csv_file, 'r'))
            self.data_path = osp.join(data_path, 'camera')
        self.data_index = [f for f in data_reader][50:-50]
        self.max_depth = max_depth
        self.vertical = vertical

        self.K = self.load_intrinsic()
        self.num_imgs = len(self.data_index)

    def load_intrinsic(self):
        last_item = self.data_index[-1]

        K = np.eye(3)
        K[0, 0] = K[1, 1] = float(last_item[7]) / 5.0
        K[0, 2] = float(last_item[9]) / 5.0
        K[1, 2] = float(last_item[10]) / 5.0
        if self.vertical:
            K[0, 2], K[1, 2] = K[1, 2], K[0, 2]
        print(K)
        return K

    def load_depth(self, depth_filename):
        depth = cv2.imread(
            osp.join(self.data_path, 'depth', depth_filename), -1)
        depth[depth == 65535] = 0
        if np.count_nonzero(depth) == 0:
            raise ValueError("depth nan")
        depth = depth / 1000.0
        if self.max_depth > 0:
            depth[depth > self.max_depth] = 0
        return depth

    def get_init_pose(self):
        return np.eye(4)

    def load_image(self, rgb_filename):
        rgb = cv2.imread(osp.join(self.data_path, 'images', rgb_filename), -1)
        rgb = cv2.resize(rgb, (256, 144))
        rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return rgb / 255.0

    def __len__(self):
        return self.num_imgs

    def __getitem__(self, index):
        item = self.data_index[index]
        rgb_filename = item[1]
        depth_filename = rgb_filename.split('.')[0] + '.png'
        img = torch.from_numpy(self.load_image(rgb_filename)).float()
        depth = self.load_depth(depth_filename)
        depth = None if depth is None else torch.from_numpy(depth).float()
        if self.vertical:
            img = img.transpose(0, 1)
            depth = depth.transpose(0, 1) if depth is not None else None
        pose = None
        return index, img, depth, self.K, pose


if __name__ == '__main__':
    import sys
    loader = DataLoader(sys.argv[1])
    for data in loader:
        index, img, depth, _ = data
        print(index, img.shape)
        cv2.imshow('img', img.numpy())
        cv2.imshow('depth', depth.numpy())
        cv2.waitKey(1)
