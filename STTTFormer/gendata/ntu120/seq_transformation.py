KAGGLE = False

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import os
import os.path as osp
import numpy as np
import pickle5 as pickle
import logging
import h5py
from sklearn.model_selection import train_test_split
import argparse



# TODO: HD-GCN uses seq_transform for normal NTU, not 120
root_path = "./"
save_path = "./"
stat_path = osp.join('/kaggle/input/ntupreseqstatistics')

raw_denoised_joints_pkl = '/kaggle/input/NTU-Pre-Seq/raw_denoised_joints.pkl'
frames_file = osp.join('/kaggle/input/ntupreseq/frames_cnt.txt')

data_format = "np"
one_hot = False

if not KAGGLE:
    import pickle as pickle

    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', help='./')
    parser.add_argument('--save_path', help='./')
    parser.add_argument('--stat_path', help='./statistics')
    parser.add_argument('--raw_denoised_joints.pkl', help='./denoised_data/raw_denoised_joints.pkl')
    parser.add_argument('--frames_file', help='./denoised_data/frames_cnt.txt')
    parser.add_argument('--data_format', help='Either h5 or np')
    parser.add_argument('--one_hot', help='Whether y should be one_hot or not')

    args = parser.parse_args()
    args_dict = vars(parser.parse_args())

    root_path = args_dict['root_path']
    save_path = args_dict['save_path']
    stat_path = args_dict['stat_path']
    raw_denoised_joints_pkl = args_dict['raw_denoised_joints.pkl']
    frames_file = args_dict['frames_file']
    data_format = args_dict['data_format']
    one_hot = args_dict['one_hot']

setup_file = osp.join(stat_path, 'setup.txt')
camera_file = osp.join(stat_path, 'camera.txt')
performer_file = osp.join(stat_path, 'performer.txt')
replication_file = osp.join(stat_path, 'replication.txt')
label_file = osp.join(stat_path, 'label.txt')
skes_name_file = osp.join(stat_path, 'skes_available_name.txt')

if not osp.exists(save_path):
    os.mkdir(save_path)


# def remove_nan_frames(ske_name, ske_joints, nan_logger):
#     num_frames = ske_joints.shape[0]
#     valid_frames = []
#
#     for f in range(num_frames):
#         if not np.any(np.isnan(ske_joints[f])):
#             valid_frames.append(f)
#         else:
#             nan_indices = np.where(np.isnan(ske_joints[f]))[0]
#             nan_logger.info('{}\t{:^5}\t{}'.format(ske_name, f + 1, nan_indices))
#
#     return ske_joints[valid_frames]

def seq_translation(skes_joints):
    for idx, ske_joints in enumerate(skes_joints):
        num_frames = ske_joints.shape[0]
        num_bodies = 1 if ske_joints.shape[1] == 75 else 2
        if num_bodies == 2:
            missing_frames_1 = np.where(ske_joints[:, :75].sum(axis=1) == 0)[0]
            missing_frames_2 = np.where(ske_joints[:, 75:].sum(axis=1) == 0)[0]
            cnt1 = len(missing_frames_1)
            cnt2 = len(missing_frames_2)

        i = 0  # get the "real" first frame of actor1
        while i < num_frames:
            if np.any(ske_joints[i, :75] != 0):
                break
            i += 1

        origin = np.copy(ske_joints[i, 3:6])  # new origin: joint-2

        for f in range(num_frames):
            # Subtract origin such that it gets "centered" around the origin
            if num_bodies == 1:
                ske_joints[f] -= np.tile(origin, 25)
            else:  # for 2 actors
                ske_joints[f] -= np.tile(origin, 50)

        # Set the values of missing joint to 0 again after "centering"
        # TODO: A joint value of 0 is overloaded with representing the origin joint as well as missing joints.
        # TODO: But maybe that's not a problem, as 0 as input does not get propagate through the network? (depends on network)
        if (num_bodies == 2) and (cnt1 > 0):
            ske_joints[missing_frames_1, :75] = np.zeros((cnt1, 75), dtype=np.float32)

        if (num_bodies == 2) and (cnt2 > 0):
            ske_joints[missing_frames_2, 75:] = np.zeros((cnt2, 75), dtype=np.float32)

        skes_joints[idx] = ske_joints  # Update

    return skes_joints


# def frame_translation(skes_joints, skes_name, frames_cnt):
#     nan_logger = logging.getLogger('nan_skes')
#     nan_logger.setLevel(logging.INFO)
#     nan_logger.addHandler(logging.FileHandler("./nan_frames.log"))
#     nan_logger.info('{}\t{}\t{}'.format('Skeleton', 'Frame', 'Joints'))
#
#     for idx, ske_joints in enumerate(skes_joints):
#         num_frames = ske_joints.shape[0]
#         # Calculate the distance between spine base (joint-1) and spine (joint-21)
#         j1 = ske_joints[:, 0:3]
#         j21 = ske_joints[:, 60:63]
#         dist = np.sqrt(((j1 - j21) ** 2).sum(axis=1))
#
#         for f in range(num_frames):
#             origin = ske_joints[f, 3:6]  # new origin: middle of the spine (joint-2)
#             if (ske_joints[f, 75:] == 0).all():
#                 ske_joints[f, :75] = (ske_joints[f, :75] - np.tile(origin, 25)) / \
#                                       dist[f] + np.tile(origin, 25)
#             else:
#                 ske_joints[f] = (ske_joints[f] - np.tile(origin, 50)) / \
#                                  dist[f] + np.tile(origin, 50)
#
#         ske_name = skes_name[idx]
#         ske_joints = remove_nan_frames(ske_name, ske_joints, nan_logger)
#         frames_cnt[idx] = num_frames  # update valid number of frames
#         skes_joints[idx] = ske_joints
#
#     return skes_joints, frames_cnt


def align_frames(skes_joints):
    """
    Align all sequences with the same frame length. Pad all skes_joints with zero frames to get maximum number of frames.

    """
    num_skes = len(skes_joints)
    # TODO: Requires global info
    max_num_frames = max_frames_cnt  # 300
    aligned_skes_joints = np.zeros((num_skes, max_num_frames, 150), dtype=np.float32)

    for idx, ske_joints in enumerate(skes_joints):
        num_frames = ske_joints.shape[0]
        num_bodies = 1 if ske_joints.shape[1] == 75 else 2
        if num_bodies == 1:
            # TODO: Diff to HD-GCN: Why do we stack the ske_joints twice instead of np.zeros_like for the 2nd?
            # TODO: The second option makes much more sense. I suspect a bug. Otherwise, can also be used as
            # TODO: training "both" networks to learn that particular form of action instead of the "2nd" network
            # TODO: not learning anything
            aligned_skes_joints[idx, :num_frames] = np.hstack((ske_joints, ske_joints))
            # aligned_skes_joints[idx, :num_frames] = np.hstack((ske_joints, np.zeros_like(ske_joints)))
        else:
            aligned_skes_joints[idx, :num_frames] = ske_joints

    return aligned_skes_joints


def one_hot_vector(labels):
    num_skes = len(labels)
    labels_vector = np.zeros((num_skes, 120))
    for idx, l in enumerate(labels):
        labels_vector[idx, l] = 1

    return labels_vector


def split_train_val(train_indices, method='sklearn', ratio=0.05):
    """
    Get validation set by splitting data randomly from training set with two methods.
    In fact, I thought these two methods are equal as they got the same performance.

    """
    if method == 'sklearn':
        return train_test_split(train_indices, test_size=ratio, random_state=10000)
    else:
        np.random.seed(10000)
        np.random.shuffle(train_indices)
        val_num_skes = int(np.ceil(0.05 * len(train_indices)))
        val_indices = train_indices[:val_num_skes]
        train_indices = train_indices[val_num_skes:]
        return train_indices, val_indices


def split_dataset(skes_joints, label, performer, setup, evaluation, save_path, data_format, one_hot):
    train_indices, test_indices, sample_indices = get_indices(performer, setup, label, evaluation)
    # m = 'sklearn'  # 'sklearn' or 'numpy'
    # Select validation set from training set
    # train_indices, val_indices = split_train_val(train_indices, m)

    # Ske_names
    skes_name = np.loadtxt(skes_name_file, dtype=str)
    train_names = skes_name[train_indices]
    test_names = skes_name[test_indices]

    # Save labels and num_frames for each sequence of each data set
    train_labels = label[train_indices]
    test_labels = label[test_indices]

    train_x = skes_joints[train_indices]
    train_y = train_labels
    test_x = skes_joints[test_indices]
    test_y = test_labels

    if one_hot:
        train_y = one_hot_vector(train_y)
        test_y = one_hot_vector(test_y)

    if len(sample_indices):
        sample_names = skes_name[sample_indices]
        sample_labels = label[sample_indices]
        sample_x = skes_joints[sample_indices]
        sample_y = sample_labels

        if one_hot:
            sample_y = one_hot_vector(sample_labels)

    if data_format == 'np':
        save_name = osp.join(save_path, 'NTU120_%s.npz') % evaluation

        if len(sample_indices):
            if one_hot:
                np.savez(save_name, x_train=train_x, y_train=train_y, names_train=train_names, x_test=test_x,
                         y_test=test_y, names_test=test_names, x_sample=sample_x, y_sample=sample_y,
                         names_sample=sample_names)
            else:
                np.savez(save_name, x_train=train_x, y_not_oh_train=train_y, names_train=train_names, x_test=test_x,
                         y_not_oh_test=test_y, names_test=test_names, x_sample=sample_x, y_not_oh_sample=sample_y,
                         names_sample=sample_names)
        else:
            if one_hot:
                np.savez(save_name, x_train=train_x, y_train=train_y, names_train=train_names, x_test=test_x,
                         y_test=test_y, names_test=test_names)
            else:
                np.savez(save_name, x_train=train_x, y_not_oh_train=train_y, names_train=train_names,
                         x_test=test_x, y_not_oh_test=test_y, names_test=test_names)

    elif data_format == 'h5':
        # Save data into a .h5 file
        h5file = h5py.File(osp.join(save_path, 'NTU_%s.h5' % (evaluation)), 'w')
        # Training set
        h5file.create_dataset('x', data=train_x)
        h5file.create_dataset('names', data=train_names)
        # Test set
        h5file.create_dataset('test_x', data=test_x)
        h5file.create_dataset('test_names', data=test_names)
        # Sample set
        if len(sample_indices):
            h5file.create_dataset('sample_x', data=sample_x)
            h5file.create_dataset('sample_names', data=sample_names)
            if one_hot:
                h5file.create_dataset('sample_y', data=sample_y)
            else:
                h5file.create_dataset('sample_y_not_oh', data=sample_y)

        if one_hot:
            h5file.create_dataset('y', data=train_y)
            h5file.create_dataset('test_y', data=test_y)
        else:
            h5file.create_dataset('y_not_oh', data=train_y)
            h5file.create_dataset('test_y_not_oh', data=test_y)

        h5file.close()

    else:
        ValueError('Unsupported save data format.')


def get_indices(performer, setup, label, evaluation='XSub'):
    test_indices = np.empty(0)
    train_indices = np.empty(0)
    sample_indices = np.empty(0)

    if evaluation == 'XSub':  # Cross Subject (Subject IDs)
        train_ids = [1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28,
                     31, 34, 35, 38, 45, 46, 47, 49, 50, 52, 53, 54, 55, 56, 57,
                     58, 59, 70, 74, 78, 80, 81, 82, 83, 84, 85, 86, 89, 91, 92,
                     93, 94, 95, 97, 98, 100, 103]
        test_ids = [i for i in range(1, 107) if i not in train_ids]

        # Get indices of test data
        for idx in test_ids:
            temp = np.where(performer == idx)[0]  # 0-based index
            test_indices = np.hstack((test_indices, temp)).astype(int)

        # Get indices of training data
        for train_id in train_ids:
            temp = np.where(performer == train_id)[0]  # 0-based index
            train_indices = np.hstack((train_indices, temp)).astype(int)
    elif evaluation == 'XSet':  # Cross Setup (Setup IDs)
        train_ids = [i for i in range(1, 33) if i % 2 == 0]  # Even setup
        test_ids = [i for i in range(1, 33) if i % 2 == 1]  # Odd setup

        # Get indices of test data
        for test_id in test_ids:
            temp = np.where(setup == test_id)[0]  # 0-based index
            test_indices = np.hstack((test_indices, temp)).astype(int)

        # Get indices of training data
        for train_id in train_ids:
            temp = np.where(setup == train_id)[0]  # 0-based index
            train_indices = np.hstack((train_indices, temp)).astype(int)

    elif evaluation == 'one_shot':

        test_ids = [1, 7, 13, 19, 25, 31, 37, 43, 49, 55, 61, 67, 73, 79, 85, 91, 97, 103, 109, 115]
        test_ids = [test_id - 1 for test_id in test_ids]

        skes_name = np.loadtxt(skes_name_file, dtype=str)
        sample_names = ['S001C003P008R001A001', 'S001C003P008R001A007', 'S001C003P008R001A013', 'S001C003P008R001A019',
                        'S001C003P008R001A025', 'S001C003P008R001A031', 'S001C003P008R001A037', 'S001C003P008R001A043',
                        'S001C003P008R001A049', 'S001C003P008R001A055', 'S018C003P008R001A061', 'S018C003P008R001A067',
                        'S018C003P008R001A073', 'S018C003P008R001A079', 'S018C003P008R001A085', 'S018C003P008R001A091',
                        'S018C003P008R001A097', 'S018C003P008R001A103', 'S018C003P008R001A109', 'S018C003P008R001A115']

        train_ids = list(set([i for i in range(120)]) - set(test_ids))

        # Get indices of sample data
        for sample_id in sample_names:
            temp = np.where(sample_id == skes_name)[0]
            sample_indices = np.hstack((sample_indices, temp)).astype(int)

        # Get indices of test data + sample data
        for test_id in test_ids:
            temp = np.where(label == test_id)[0]  # 0-based index
            test_indices = np.hstack((test_indices, temp)).astype(int)

        # Get actual indices of test data
        test_indices = np.array(list(set(test_indices) - set(sample_indices))).astype(int)
        test_indices.sort()

        # Get indices of train data
        for train_id in train_ids:
            temp = np.where(label == train_id)[0]  # 0-based index
            train_indices = np.hstack((train_indices, temp)).astype(int)

    else:
        raise ValueError('Unsupported evaluation protocol.')

    return train_indices, test_indices, sample_indices


if __name__ == '__main__':
    # setup = np.loadtxt(setup_file, dtype=int)  # camera id: 1~32
    # performer = np.loadtxt(performer_file, dtype=int)  # subject id: 1~106
    # label = np.loadtxt(label_file, dtype=int) - 1  # action label: 0~119
    #
    # get_indices(performer, setup, label, evaluation='one_shot')
    #
    # f = np.load('NTU_one_shot_npy/y_train.npy', mmap_mode='r')

    with open(raw_denoised_joints_pkl, 'rb') as fr:
        skes_joints = pickle.load(fr)  # a list

    frames_cnt = np.loadtxt(frames_file, dtype=int)  # frames_cnt
    max_frames_cnt = frames_cnt.max()
    del frames_cnt

    skes_joints = seq_translation(skes_joints)

    skes_joints = align_frames(skes_joints)  # aligned to the same frame length

    # TODO: Maybe load later for memory?
    setup = np.loadtxt(setup_file, dtype=int)  # camera id: 1~32
    performer = np.loadtxt(performer_file, dtype=int)  # subject id: 1~106
    label = np.loadtxt(label_file, dtype=int) - 1  # action label: 0~119

    # evaluations = ['XSet', 'XSub']
    evaluations = ['one_shot']
    for evaluation in evaluations:
        split_dataset(skes_joints, label, performer, setup, evaluation, save_path, data_format, one_hot)
