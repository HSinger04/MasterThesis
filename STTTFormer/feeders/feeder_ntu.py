import numpy as np
from torch.utils.data import Dataset
from . import tools
import os.path as osp

class Feeder(Dataset):
    def __init__(self, data_path, label_path=None, names_path=None, p_interval=1, split='train', random_choose=False, random_shift=False,
                 random_move=False, random_rot=False, window_size=-1, normalization=False, debug=False, use_mmap=True,
                 bone=False, vel=False):
        """
        data_path:
        label_path:
        names_path:
        split: training set or test set
        random_choose: If true, randomly choose a portion of the input sequence
        random_shift: If true, randomly pad zeros at the begining or end of sequence
        random_move:
        random_rot: rotate skeleton around xyz axis
        window_size: The length of the output sequence
        normalization: If true, normalize input sequence
        debug: If true, only use the first 100 samples
        use_mmap: If true, use mmap mode to load data, which can save the running memory
        bone: use bone modality or not
        vel: use motion modality or not
        only_label: only load label for ensemble score compute
        """

        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.names_path = names_path
        self.split = split
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.p_interval = p_interval
        self.random_rot = random_rot
        self.bone = bone
        self.vel = vel
        self.load_data()
        if normalization:
            self.get_mean_map()
    
    def normalize_labels(self):
        """ Remap the labels to only go up to the number of unique labels to avoid bugs with e.g. CrossEntropyLoss. """
        label_to_new_labels = dict(zip(np.unique(self.label), np.array(range(len(np.unique(self.label))))))
        self.label = np.array([label_to_new_labels[label] for label in self.label])

    def load_data(self):
        mmap_mode = None
        if self.use_mmap:
            mmap_mode = 'r'
        # data: N C V T M
        if self.label_path and self.names_path:
            self.data = np.load(self.data_path, mmap_mode=mmap_mode)
            self.label = np.load(self.label_path)
            self.sample_name = np.load(self.names_path)
        else:
            self.data = np.load(osp.join(self.data_path, "x_" + self.split + ".npy"), mmap_mode=mmap_mode)
            self.label = np.load(osp.join(self.data_path, "y_" + self.split + ".npy"))
            self.sample_name = np.load(osp.join(self.data_path, "names_" + self.split + ".npy"))

        if not np.sum(np.core.defchararray.find(self.sample_name, ".skeleton")!=-1) == self.__len__():
            self.sample_name = np.char.add(self.sample_name, ".skeleton")

        try:
            self.label = np.where(self.label > 0)[1]
        except IndexError:
            pass


        if self.split not in ("train", "test", "sample", "test_and_sample"):
            raise NotImplementedError('data split only supports train/test')

        try:
            N, T, _ = self.data.shape
            self.data = self.data.reshape((N, T, 2, 25, 3)).transpose(0, 4, 1, 3, 2)
        except ValueError:
            pass


    def get_mean_map(self):
        data = self.data
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))

    def __len__(self):
        return len(self.label)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        data_numpy = self.data[index]
        label = self.label[index]
        data_numpy = np.array(data_numpy)
        valid_frame_num = np.sum(data_numpy.sum(0).sum(-1).sum(-1) != 0)
        # reshape Tx(MVC) to CTVM
        data_numpy = tools.valid_crop_resize(data_numpy, valid_frame_num, self.p_interval, self.window_size)
        if self.random_rot:
            data_numpy = tools.random_rot(data_numpy)
        # TODO: Adjust bone and vel to also support mmap
        if self.bone:
            ntu_pairs = ((1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5),
                (7, 6), (8, 7), (9, 21), (10, 9), (11, 10), (12, 11),
                (13, 1), (14, 13), (15, 14), (16, 15), (17, 1), (18, 17),
                (19, 18), (20, 19), (22, 23), (21, 21), (23, 8), (24, 25),(25, 12))
            bone_data_numpy = np.zeros_like(data_numpy)
            for v1, v2 in ntu_pairs:
                bone_data_numpy[:, :, v1 - 1] = data_numpy[:, :, v1 - 1] - data_numpy[:, :, v2 - 1]
            data_numpy = bone_data_numpy
        if self.vel:
            data_numpy[:, :-1] = data_numpy[:, 1:] - data_numpy[:, :-1]
            data_numpy[:, -1] = 0

        return data_numpy, label#, index