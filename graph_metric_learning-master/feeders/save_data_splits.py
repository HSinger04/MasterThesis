import os.path as osp
import sys

sys.path.append(osp.dirname(osp.dirname(osp.dirname(osp.dirname(__file__)))))

import yaml
import numpy as np

from feeder import get_train_and_os_val
from argparse import ArgumentParser
from MasterThesis.STTTFormer.feeders import feeder_ntu


def save_dataset(save_path, dataset):
    if save_path.endswith(".npy"):
        save_path = save_path[:-4]
    with open(save_path + "_data.npy", "wb") as f:
        np.save(f, dataset.data)
    with open(save_path + "_labels.npy", "wb") as f:
        np.save(f, dataset.label)
    with open(save_path + "_names.npy", "wb") as f:
        np.save(f, dataset.sample_name)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--feeder_class', help='Name of the feeder class to use. Either STTFormer, HD-GCN"')
    parser.add_argument('--config_path')
    arg_parser = vars(parser.parse_args())

    with open(arg_parser["config_path"], "r") as f:
        data_config = yaml.load(f, yaml.Loader)

    feeder_class = arg_parser["feeder_class"]
    if feeder_class == "STTFormer" or feeder_class == "HD-GCN" or feeder_class == "Hyperformer":
        arg_parser["feeder_class"] = feeder_ntu.Feeder


    arg_parser["data_path"] = data_config["data_path"][len("../"):]
    arg_parser["label_path"] = data_config["label_path"][len("../"):]
    arg_parser["name_path"] = data_config["name_path"][len("../"):]
    arg_parser["mem_limits"] = data_config["dataset_split"]["mem_limits"]
    arg_parser["val_classes"] = data_config["dataset_split"]["val_classes"]
    arg_parser["data_kwargs"] = data_config["data_kwargs"]

    del arg_parser["config_path"]

    val_sample_names = []

    for i in data_config["dataset_split"]["val_classes"]:
        sample_name = ""

        if i < 61:
            sample_name += "S001"
        else:
            sample_name += "S018"

        sample_name += "C003P008R001A"

        if i < 10:
            sample_name += "00" + str(i)
        elif i < 100:
            sample_name += "0" + str(i)
        else:
            sample_name += str(i)

        sample_name += ".skeleton"
        val_sample_names.append(sample_name)

    arg_parser["val_sample_names"] = val_sample_names

    train_dataset, val_dataset, val_samples_dataset = get_train_and_os_val(**arg_parser)

    save_dataset("train", train_dataset)
    save_dataset("val", val_dataset)
    save_dataset("val_samples", val_samples_dataset)