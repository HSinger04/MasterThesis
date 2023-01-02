import argparse
import numpy as np
import pickle
import os.path as osp

def read_data(data_path):
    data = None

    if data_path.endswith("_label.pkl"):
        sample_name = None
        label = None
        try:
            with open(data_path) as f:
                sample_name, label = pickle.load(f)
        except:
            # for pickle file from python2
            with open(data_path, 'rb') as f:
                sample_name, label = pickle.load(f, encoding='latin1')
        return sample_name, label

    elif data_path.endswith(".npy"):
        data = np.load(data_path, mmap_mode='r')
    return data


def assert_y_equivalence(gmlm_y, other_y, one_hot):
    if one_hot:
        assert np.all(np.nonzero(other_y)[1] == gmlm_y)
    else:
        assert np.all(gmlm_y == other_y)


def main(gmlm_data_path, other_x_train_path, other_y_train_path, other_y_one_hot):
    """

    :param gmlm_data_path: data path to graph_metric_learning-master's one_shot data.
    :param other_y_train_path: path to the other's train data's labels.
    :param other_y_one_hot: True if other's data's label is one-hot encoded.
    """
    # TODO: Fill in the other data of gmlm
    gmlm_train_sample_name, gmlm_train_label = read_data(osp.join(gmlm_data_path, "train_label.pkl"))
    other_x_train = read_data(other_x_train_path)
    other_y_train = read_data(other_y_train_path)

    assert_y_equivalence(gmlm_train_label, other_y_train, other_y_one_hot)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gmlm_data_path', default='../data/ntu/one_shot/')
    parser.add_argument('--other_x_train_path')
    parser.add_argument('--other_y_train_path')
    parser.add_argument('--other_y_one_hot')

    main(**vars(parser.parse_args()))

