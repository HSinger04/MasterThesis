import numpy as np
import pickle
import torch
from torch.utils.data import Dataset, Subset
import sys

sys.path.extend(['../'])
from feeders import tools
import os
import os.path as osp
from shutil import rmtree
from tqdm import tqdm


class Feeder(Dataset):
    def __init__(self, data_path, label_path, train=False,
                 random_choose=False, random_shift=False, random_move=False,
                 window_size=-1, normalization=False, debug=False, use_mmap=False):
        """
        
        :param data_path: 
        :param label_path: 
        :param random_choose: If true, randomly choose a portion of the input sequence
        :param random_shift: If true, randomly pad zeros at the begining or end of sequence
        :param random_move: 
        :param window_size: The length of the output sequence
        :param normalization: If true, normalize input sequence
        :param debug: If true, only use the first 100 samples
        :param use_mmap: If true, use mmap mode to load data, which can save the running memory
        """

        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        # FAQ: Can lead to memory overload
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.train = train
        self.load_data()
        if normalization:
            self.get_mean_map()
            self.data[:10] = (self.data[:10] - self.mean_map) / self.std_map

    def load_data(self):
        # data: N C V T M
        mmap_mode = None
        if self.use_mmap:
            mmap_mode = 'r'

        self.data = np.load(self.data_path, mmap_mode=mmap_mode)

        try:
            with open(self.label_path) as f:
                self.sample_name, self.label = pickle.load(f)
        except:
            # for pickle file from python2
            with open(self.label_path, 'rb') as f:
                self.sample_name, self.label = pickle.load(f, encoding='latin1')

        self.label = np.array(self.label)
        self.sample_name = np.array(self.sample_name)

        if self.debug:
            self.label = self.label[0:self.debug]
            self.data = self.data[0:self.debug]
            self.sample_name = self.sample_name[0:self.debug]


    def normalize_labels(self):
        """ Remap the labels to only go up to the number of unique labels to avoid bugs with e.g. CrossEntropyLoss. """
        label_to_new_labels = dict(zip(np.unique(self.label), np.array(range(len(np.unique(self.label))))))
        self.label = np.array([label_to_new_labels[label] for label in self.label])

    def get_mean_map(self):
        """ Gets used for normalization. """
        data = self.data
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        # standard deviation
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))

    def __len__(self):
        return len(self.label)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        data_numpy = self.data[index]
        label = self.label[index]
        data_numpy = np.array(data_numpy)

        if self.random_shift:
            data_numpy = tools.random_shift(data_numpy)
        if self.random_choose:
            data_numpy = tools.random_choose(data_numpy, self.window_size)
        elif self.window_size > 0:
            data_numpy = tools.auto_pading(data_numpy, self.window_size)
        if self.random_move:
            data_numpy = tools.random_move(data_numpy)

        return data_numpy, label #, index

    def top_k(self, score, top_k):
        """ Only gets used in Processor's eval method, thus start method

        :param score:
        :param top_k:
        :return:
        """
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)


def get_data_from_idxs(data, idxs, mmap_dict={"mmap_filename": "temp", "mem_limit": 0}):
    """ From data, take the subset given by idxs.

    :param data: Data from which to generate subset of data.
    :param idxs: Indices that dictate the subset to take from data.
    :param mmap_dict: If your RAM isn't big enough, you can specify how many datapoints at once can be loaded into RAM.
    In that case, your dataset will be saved in numpy.memmap mode and thus need to specify where the memmap should
    be saved via key "mmap_filename" and how many datapoints can be loaded up at once in "mem_limit".
    :return: Subset of data given by idxs as a np.array-like.
    """
    filename = ""
    if mmap_dict["mem_limit"]:
        filename = mmap_dict["mmap_filename"] + ".dat"
        mmap_shape = list(data.data.shape)
        mmap_shape[0] = idxs.sum()
        mmap_arr = np.memmap(filename, dtype=data.data.dtype, mode='w+', shape=tuple(mmap_shape))
        mem_limit = mmap_dict["mem_limit"]
        j = (idxs.size // mem_limit) + 1
        prev = 0
        print("Processing " + os.path.basename(filename))
        for i in tqdm(range(j)):
            start_idx = i * mem_limit
            temp_idxs = np.zeros(idxs.shape)
            try:
                end_idx = (i + 1) * mem_limit
                temp_idxs = idxs[start_idx:end_idx]
                data_slice = data.data[start_idx:end_idx][idxs[start_idx:end_idx]]
            except IndexError:
                data_slice = data.data[start_idx:][idxs[start_idx:]]
            mmap_arr[prev:prev + data_slice.shape[0]] = data_slice
            prev += data_slice.shape[0]

        data.data = mmap_arr

    else:
        data.data = data.data[idxs]
    data.label = data.label[idxs]
    data.sample_name = data.sample_name[idxs]
    return data, filename


def get_train_and_os_val(feeder_class, data_path, label_path, name_path, val_classes, val_sample_names,
                         mem_limits={"val_samples": 0, "val": 0, "train": 0}, debug=False, data_kwargs={}):
    """ Of the original train dataset given by data_path and label_path, generate the true train dataset that
    the model gets trained on as well as a one-shot validation dataset and corresponding samples for it.

    :param data_path: Path to the original train dataset
    :param label_path: Path to the original train dataset's label
    :param val_classes: List of classes to use for validation (name classes as values from {1, ..., 120})
    :param val_sample_names: List of skeletons to use as samples for the representatives of the validation classes
    :param mem_limits: Dictionary that says up to how many examples can be loaded into RAM at once. If you want
    a dataset loaded fully into RAM, specify 0 as the corresponding key's value.
    :param debug: Set true for debugging purposes - generates the true train dataset faster,
    which is the main bottleneck in speed.
    :param data_kwargs: Additional keyword arguments for initializing the datasets
    :return: true train, validation and validation samples datasets from the original dataset as np.array-likes.
    """

    if mem_limits:
        # Use a temporary folder for mmap_mode
        mmap_folder = osp.join(osp.dirname(data_path), "temp")
        if osp.exists(mmap_folder):
            # Remove mmap_folder if it exists.
            rmtree(mmap_folder, ignore_errors=True)

        os.makedirs(mmap_folder)

    # Create validation samples dataset
    val_samples_dataset = feeder_class(data_path, label_path, name_path, use_mmap=bool(mem_limits["val_samples"]),
                                       **data_kwargs["val_samples"])
    orig_train_length = len(val_samples_dataset)

    # Only pick skeletons that are listed in val_sample_names
    val_samples_idxs = np.isin(val_samples_dataset.sample_name, [val_sample_names])
    val_samples_dataset, val_samples_filename = get_data_from_idxs(
        val_samples_dataset, val_samples_idxs, mmap_dict={"mmap_filename": osp.join(mmap_folder, "val_samples"),
                                                      "mem_limit": mem_limits["val_samples"]})
    if len(val_samples_dataset) < mem_limits["val_samples"]:
        val_samples_dataset.data = np.array(val_samples_dataset.data)

    # Create validation dataset
    val_dataset = feeder_class(data_path, label_path, name_path, use_mmap=bool(mem_limits["val"]), **data_kwargs["val"])
    # Only pick skeletons that are of the val_classes and also not part of the samples.
    val_data_idxs = np.logical_xor(np.isin(val_dataset.label + 1, val_classes), val_samples_idxs)
    #TODO: Comment out
    if debug:
        true_count = 0
        idx = 0
        for i, bool_val in enumerate(val_data_idxs):
            if bool_val:
                true_count += 1
                if true_count == 128:
                    idx = i
                    break
        val_data_idxs[idx:] = False
    val_dataset, val_filename = get_data_from_idxs(val_dataset, val_data_idxs, mmap_dict={"mmap_filename": osp.join(mmap_folder,
                                                                                                      "val"),
                                                      "mem_limit": mem_limits["val"]})
    if len(val_dataset) < mem_limits["val"]:
        val_dataset.data = np.array(val_dataset.data)

    # Create true train set.
    if not debug:
        train_dataset = feeder_class(data_path, label_path, name_path, use_mmap=bool(mem_limits["train"]), **data_kwargs["train"])
        # Pick skeletons that are neither part of validation samples nor validation dataset.
        train_data_idxs = np.logical_not(np.logical_or(val_samples_idxs, val_data_idxs))
        del val_samples_idxs
        del val_data_idxs
        train_dataset, train_filename = get_data_from_idxs(train_dataset, train_data_idxs,
                                           mmap_dict={"mmap_filename": osp.join(mmap_folder, "train"),
                                                      "mem_limit": mem_limits["train"]})

        assert orig_train_length == len(val_samples_dataset) + len(val_dataset) + len(train_dataset)

    else:
        train_dataset = feeder_class(data_path, label_path, name_path, use_mmap=bool(mem_limits["train"]), **data_kwargs["train"])

    train_dataset.normalize_labels()

    if len(train_dataset) < mem_limits["train"]:
        train_dataset.data = np.array(train_dataset.data)

    return train_dataset, val_dataset, val_samples_dataset


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def test(data_path, label_path, vid=None, graph=None, is_3d=False):
    '''
    vis the samples using matplotlib
    :param data_path: 
    :param label_path: 
    :param vid: the id of sample
    :param graph: 
    :param is_3d: when vis NTU, set it True
    :return: 
    '''
    import matplotlib.pyplot as plt
    loader = torch.utils.data.DataLoader(
        dataset=Feeder(data_path, label_path),
        batch_size=64,
        shuffle=False,
        num_workers=2)

    if vid is not None:
        sample_name = loader.dataset.sample_name
        sample_id = [name.split('.')[0] for name in sample_name]
        index = sample_id.index(vid)
        data, label, index = loader.dataset[index]
        data = data.reshape((1,) + data.shape)

        # for batch_idx, (data, label) in enumerate(loader):
        N, C, T, V, M = data.shape

        plt.ion()
        fig = plt.figure()
        if is_3d:
            from mpl_toolkits.mplot3d import Axes3D
            ax = fig.add_subplot(111, projection='3d')
        else:
            ax = fig.add_subplot(111)

        if graph is None:
            p_type = ['b.', 'g.', 'r.', 'c.', 'm.', 'y.', 'k.', 'k.', 'k.', 'k.']
            pose = [
                ax.plot(np.zeros(V), np.zeros(V), p_type[m])[0] for m in range(M)
            ]
            ax.axis([-1, 1, -1, 1])
            for t in range(T):
                for m in range(M):
                    pose[m].set_xdata(data[0, 0, t, :, m])
                    pose[m].set_ydata(data[0, 1, t, :, m])
                fig.canvas.draw()
                plt.pause(0.001)
        else:
            p_type = ['b-', 'g-', 'r-', 'c-', 'm-', 'y-', 'k-', 'k-', 'k-', 'k-']
            import sys
            from os import path
            sys.path.append(
                path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))
            G = import_class(graph)()
            edge = G.inward
            pose = []
            for m in range(M):
                a = []
                for i in range(len(edge)):
                    if is_3d:
                        a.append(ax.plot(np.zeros(3), np.zeros(3), p_type[m])[0])
                    else:
                        a.append(ax.plot(np.zeros(2), np.zeros(2), p_type[m])[0])
                pose.append(a)
            ax.axis([-1, 1, -1, 1])
            if is_3d:
                ax.set_zlim3d(-1, 1)
            for t in range(T):
                for m in range(M):
                    for i, (v1, v2) in enumerate(edge):
                        x1 = data[0, :2, t, v1, m]
                        x2 = data[0, :2, t, v2, m]
                        if (x1.sum() != 0 and x2.sum() != 0) or v1 == 1 or v2 == 1:
                            pose[m][i].set_xdata(data[0, 0, t, [v1, v2], m])
                            pose[m][i].set_ydata(data[0, 1, t, [v1, v2], m])
                            if is_3d:
                                pose[m][i].set_3d_properties(data[0, 2, t, [v1, v2], m])
                fig.canvas.draw()
                # plt.savefig('/home/lshi/Desktop/skeleton_sequence/' + str(t) + '.jpg')
                plt.pause(0.01)


if __name__ == '__main__':
    import os

    os.environ['DISPLAY'] = 'localhost:10.0'
    data_path = "../data/ntu/one_shot/val_data_joint.npy"
    label_path = "../data/ntu/one_shot/val_label.pkl"
    graph = 'graph.ntu_rgb_d.Graph'
    test(data_path, label_path, vid='S004C001P003R001A110', graph=graph, is_3d=True)
    # data_path = "../data/kinetics/val_data.npy"
    # label_path = "../data/kinetics/val_label.pkl"
    # graph = 'graph.Kinetics'
    # test(data_path, label_path, vid='UOD7oll3Kqo', graph=graph)
