import random
import sys
from os.path import dirname, abspath
# e.g. import src module to test dir where src and test are siblings
sys.path.append(dirname(dirname(dirname(abspath(__file__)))))

import time, gc

import numpy as np
from torch.utils.data import Dataset
from pytorch_metric_learning import losses, miners, samplers, trainers, testers, utils
import torch.nn as nn
import record_keeper
import pytorch_metric_learning.utils.logging_presets as logging_presets
#from torchvision import datasets, models, transforms
#import torchvision
import logging
logging.getLogger().setLevel(logging.INFO)
import os.path as osp

from hydra import compose, initialize
from omegaconf import OmegaConf
import pytorch_metric_learning
from pytorch_metric_learning.testers.base_tester import BaseTester
from pytorch_metric_learning.trainers import TrainWithClassifier
from scipy.spatial import distance_matrix

logging.info("pytorch-metric-learning VERSION %s"%pytorch_metric_learning.__version__)
logging.info("record_keeper VERSION %s"%record_keeper.__version__)

from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
#from efficientnet_pytorch import EfficientNet
import torch
from torch.utils.data import Dataset
import numpy as np
import pickle

import hydra
from omegaconf import DictConfig

from numpy.testing import assert_almost_equal
from model import agcn, msg3d
from graph import ntu_rgb_d
from feeders import feeder
import hydra
from tester.with_autocast_one_shot_tester import WithAMPGlobalEmbeddingSpaceTester

class MLP(nn.Module):
    # layer_sizes[0] is the dimension of the input
    # layer_sizes[-1] is the dimension of the output
    def __init__(self, layer_sizes, final_relu=False):
        super().__init__()
        layer_list = []
        layer_sizes = [int(x) for x in layer_sizes]
        num_layers = len(layer_sizes) - 1
        final_relu_layer = num_layers if final_relu else num_layers - 1
        for i in range(len(layer_sizes) - 1):
            input_size = layer_sizes[i]
            curr_size = layer_sizes[i + 1]
            if i < final_relu_layer:
                layer_list.append(nn.ReLU(inplace=True))
            layer_list.append(nn.Linear(input_size, curr_size))
        self.net = nn.Sequential(*layer_list)
        self.last_linear = self.net[-1]

    def forward(self, x):
        return self.net(x)

def get_datasets(data_dir, cfg, mode="train"):
    data_path = "/home/work/PycharmProjects/MA/MasterThesis/graph_metric_learning-master"

    debug_val = 128

    # train_dataset = feeder.Feeder(data_path=osp.join(data_path, "data/ntu/one_shot/train_data_joint.npy"),
    #                               label_path=osp.join(data_path, "data/ntu/one_shot/train_label.pkl"),
    #                               train=True,
    #                               debug=128,
    #                               use_mmap=True)

    test_dataset = feeder.Feeder(data_path=osp.join(data_path, "data/ntu/one_shot/val_data_joint.npy"),
                               label_path=osp.join(data_path, "data/ntu/one_shot/val_label.pkl"),
                                train=False,
                               debug=8)

    sample_dataset = feeder.Feeder(data_path=osp.join(data_path, "data/ntu/one_shot/sample_data_joint.npy"),
                               label_path=osp.join(data_path, "data/ntu/one_shot/sample_label.pkl"),
                                   train=False)


    return test_dataset, sample_dataset

# Timing utilities
start_time = None

def start_timer():
    global start_time
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.synchronize()
    start_time = time.time()

def get_time_and_memory():
    torch.cuda.synchronize()
    end_time = time.time()
    return end_time - start_time, torch.cuda.max_memory_allocated()


class TestAMPDataset(Dataset):

    def __init__(self, data, label):
        self.data = np.float32(data)
        self.label = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]


def train_app(cfg, use_amp, base_class, num_false_tests=None, num_tests=None, embed_dim=None):
    if num_false_tests is None:
        num_false_tests = random.randint(1, 10)
    num_samples = random.randint(2, 10)
    if num_tests is None:
        num_tests = num_false_tests * random.randint(1, 10)
    if embed_dim is None:
        embed_dim = random.randint(1, 10)

    print("Num false tests: " + str(num_false_tests))
    print("Num tests: " + str(num_tests))
    print("Num samples: " + str(num_samples))


    samples = np.random.rand(num_samples, embed_dim)
    samples[-1] = -1
    old_samples_labels = np.array(list(range(num_samples)))
    test = np.random.rand(num_tests, embed_dim)

    samples_labels = old_samples_labels.copy()
    test_labels = KNeighborsClassifier(n_neighbors=1).fit(samples, old_samples_labels).predict(test)
    idxs = np.random.choice(range(num_tests), num_false_tests, replace=False)
    for idx in idxs:
        test_labels[idx] = (test_labels[idx] + 1) % num_samples
    if base_class == "Identity":
        model = torch.nn.Sequential(torch.nn.Identity())
    elif base_class == "Matrix":
        model = torch.nn.Sequential(torch.nn.Linear(embed_dim, 200))
        for _ in range(9):
            model.append(torch.nn.Linear(200, 200))
    else:
        ValueError()
    model.to(device="cuda")

    val_samples_dataset = TestAMPDataset(samples, samples_labels)
    #val_dataset =TestAMPDataset(samples, old_samples_labels)
    val_dataset = TestAMPDataset(test, test_labels)
    dataset_dict = {"samples": val_samples_dataset, "val": val_dataset}

    start_timer()
    tester = WithAMPGlobalEmbeddingSpaceTester(use_amp, normalize_embeddings=False,
                                                batch_size=96000, dataloader_num_workers=cfg.tester.tester.dataloader_num_workers,
                                                accuracy_calculator=utils.accuracy_calculator.AccuracyCalculator(include=("precision_at_1", )))
    time, mem = get_time_and_memory()
    test_result = tester.test(dataset_dict, 0, model
                , splits_to_eval=[('val', ['samples'])]
    )
    actual_acc = test_result["val"]["precision_at_1_level0"]
    desired_acc = 1 - (num_false_tests / num_tests)
    return actual_acc, desired_acc, time, mem

def test_autocast_equivalence():
    num_false_tests = 10
    num_tests = 128000 * num_false_tests

    with initialize(version_base=None, config_path="../../config", job_name="test_app"):
        # Run standard TrainWithClassifier - no amp!
        cfg = compose(config_name="test_tester_use_amp_true")
        # Confirm that accuracy gets measured correctly across multiple random settings.
        for _ in range(5):
            actual_acc, desired_acc, _, _ = train_app(cfg, False, "Identity")
            assert_almost_equal(actual_acc, desired_acc)

        # Confirm that it also works correctly with amp
        for _ in range(5):
            actual_acc, desired_acc, _, _ = train_app(cfg, True, "Identity")
            assert_almost_equal(actual_acc, desired_acc)


        _, _, time_no_amp, mem_no_amp = train_app(cfg, False, "Matrix", num_false_tests, num_tests, 100)
        _, _, time_with_amp, mem_with_amp = train_app(cfg, True, "Matrix", num_false_tests, num_tests, 100)

        print(mem_no_amp)
        print(mem_with_amp)
        assert mem_with_amp <= mem_no_amp

        print(time_no_amp)
        print(time_with_amp)
        assert time_with_amp < time_no_amp





    # with initialize(version_base=None, config_path="../../config", job_name="test_app"):
    #     # Run WithAutoCastTrainWithClassifier, though AMP is disabled.
    #     cfg = compose(config_name="test_use_amp_false")
    #     model_amp_1, time_amp_1, mem_amp_1 = train_app(cfg, WithAutocastTrainWithClassifier)
    #     embedder_amp_1_params = list(model_amp_1["embedder"].parameters())
    #
    # with initialize(version_base=None, config_path="../../config", job_name="test_app"):
    #     # Run WithAutoCastTrainWithClassifier, though AMP is enabled.
    #     cfg = compose(config_name="test_use_amp_true")
    #     model_amp_2, time_amp_2, mem_amp_2 = train_app(cfg, WithAutocastTrainWithClassifier)
    #     embedder_amp_2_params = list(model_amp_2["embedder"].parameters())
    #
    # for params_1, params_2 in zip(embedder_def_params, embedder_amp_1_params):
    #     # Assert that TrainWithClassifier and WithAutoCastTrainWithClassifier without AMP results in same weights.
    #     params_1 = params_1.cpu().detach()
    #     params_2 = params_2.cpu().detach()
    #     assert np.array_equal(params_1, params_2)
    #
    # for params_1, params_2 in zip(embedder_amp_1_params, embedder_amp_2_params):
    #     # Assert that WithAutoCastTrainWithClassifier with and without AMP results in similar results.
    #     params_1 = params_1.cpu().detach()
    #     params_2 = params_2.cpu().detach()
    #     total = 0
    #     for compar in np.where(np.isclose(params_1, params_2)):
    #         total += len(compar)
    #     print(total)
    #     print(torch.numel(params_1))
    #     assert total > torch.numel(params_1) // 2
    #
    # # Assert that AMP results in better time and memory.
    # print(time_amp_1)
    # print(time_amp_2)
    # assert time_amp_2 < time_amp_1
    #
    # print(mem_amp_1)
    # print(mem_amp_2)
    # assert mem_amp_2 < mem_amp_1

if __name__ == '__main__':
    test_autocast_equivalence()