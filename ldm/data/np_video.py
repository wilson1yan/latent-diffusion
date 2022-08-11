import os.path as osp
import numpy as np
import warnings
import glob
import pickle
import math

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class NumpyVideoDataset(Dataset):
    """ Generic dataset for video files stored in folders
    Returns BCTHW videos in the range [-1, 1] """

    def __init__(self, size, data_path, train=True):
        super().__init__()
        self.train = train
        self.size = size
        self.data_path = data_path
        self.seq_len = 4

        folder = osp.join(data_path, 'train' if train else 'test')
        files = glob.glob(osp.join(folder, '**', '*.npz'), recursive=True)
        self.files = files
        print('Found', len(files))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        video = data['video']
        start = np.random.randint(low=0, high=video.shape[0] - self.seq_len)
        video = video[start:start + self.seq_len]
        video = torch.from_numpy(video)
        video = preprocess(video, self.size)
        return dict(image=video)

        
class NumpyVideoDatasetTrain(NumpyVideoDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, train=True)

class NumpyVideoDatasetTest(NumpyVideoDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, train=False)
        
def preprocess(video, image_size, sequence_length=None):
    # video: THWC, {0, ..., 255}

    resolution = image_size

    video = video.movedim(-1, 1).float() / 255. # TCHW
    t, c, h, w = video.shape

    # temporal crop
    if sequence_length is not None:
        assert sequence_length <= t
        video = video[:sequence_length]

    # scale shorter side to resolution
    scale = resolution / min(h, w)
    if h < w:
        target_size = (resolution, math.ceil(w * scale))
    else:
        target_size = (math.ceil(h * scale), resolution)
    video = F.interpolate(video, size=target_size, mode='bilinear',
                          align_corners=False)

    # center crop
    t, c, h, w = video.shape
    w_start = (w - resolution) // 2
    h_start = (h - resolution) // 2
    video = video[:, :, h_start:h_start + resolution, w_start:w_start + resolution]

    video = 2 * video - 1 # [0, 1] -> [-1, 1]

    return video.movedim(1, -1)
