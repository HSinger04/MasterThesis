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

logging.info("pytorch-metric-learning VERSION %s"%pytorch_metric_learning.__version__)
logging.info("record_keeper VERSION %s"%record_keeper.__version__)

from sklearn.metrics import accuracy_score
#from efficientnet_pytorch import EfficientNet
import torch
import numpy as np
import pickle

import hydra
from omegaconf import DictConfig

from model import agcn, msg3d
from graph import ntu_rgb_d
from feeders import feeder
import hydra
from trainer.with_autocast_train_with_classifier import WithAutocastTrainWithClassifier

class OneShotTester(BaseTester):

    def __init__(self, end_of_testing_hook=None):
        super().__init__()
        self.max_accuracy = 0.0
        self.embedding_filename = ""
        self.end_of_testing_hook = end_of_testing_hook

    def __get_correct(self, output, target, topk=(1,)):
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))
    #             print(correct)
        return correct


    def __accuracy(self, output, target, topk=(1,)):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            correct = self.__get_correct(output, target, topk)
            batch_size = target.size(0)
            res = []
            for k in topk:
                correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res


    def do_knn_and_accuracies(self, accuracies, embeddings_and_labels, split_name, tag_suffix=''):
        #print(embeddings_and_labels)
        query_embeddings = embeddings_and_labels["val"][0]
        query_labels = embeddings_and_labels["val"][1]
        reference_embeddings = embeddings_and_labels["samples"][0]
        reference_labels = embeddings_and_labels["samples"][1]
        knn_indices, knn_distances = utils.stat_utils.get_knn(reference_embeddings, query_embeddings, 1, False)
        knn_labels = reference_labels[knn_indices][:,0]

        accuracy = accuracy_score(knn_labels, query_labels)
        print(accuracy)
        with open(self.embedding_filename+"_last", 'wb') as f:
            print("Dumping embeddings for new max_acc to file", self.embedding_filename+"_last")
            pickle.dump([query_embeddings, query_labels, reference_embeddings, reference_labels, accuracy], f)
        accuracies["accuracy"] = accuracy
        keyname = self.accuracies_keyname("mean_average_precision_at_r") # accuracy as keyname not working
        accuracies[keyname] = accuracy

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

class Identity(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

def get_datasets(data_dir, cfg, mode="train"):
    data_path = "../../"

    debug_val = 128

    train_dataset = feeder.Feeder(data_path=osp.join(data_path, "data/ntu/one_shot/train_data_joint.npy"),
                                  label_path=osp.join(data_path, "data/ntu/one_shot/train_label.pkl"),
                                  train=True,
                                  debug=debug_val,
                                  use_mmap=True)

    test_dataset = feeder.Feeder(data_path=osp.join(data_path, "data/ntu/one_shot/val_data_joint.npy"),
                               label_path=osp.join(data_path, "data/ntu/one_shot/val_label.pkl"),
                                train=False,
                               debug=debug_val)

    sample_dataset = feeder.Feeder(data_path=osp.join(data_path, "data/ntu/one_shot/sample_data_joint.npy"),
                               label_path=osp.join(data_path, "data/ntu/one_shot/sample_label.pkl"),
                                   train=False,
                               debug=debug_val)


    return train_dataset, test_dataset, sample_dataset

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

def train_app(cfg, train_class):
    # reprodcibile
    np.random.seed(42)
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set trunk model and replace the softmax layer with an identity function
    # trunk = torchvision.models.__dict__[cfg.model.model.model_name](pretrained=cfg.model.model.pretrained)
    # graph = ntu_rgb_d.Graph()
    # trunk = agcn.Model(graph="graph.ntu_rgb_d.Graph")
    trunk = msg3d.Model(graph="graph.ntu_rgb_d.AdjMatrixGraph", num_class=100, num_point=25, num_person=2,
                        num_gcn_scales=13, num_g3d_scales=6)

    # resnet18(pretrained=True)
    # trunk = models.alexnet(pretrained=True)
    # trunk = models.resnet50(pretrained=True)
    # trunk = models.resnet152(pretrained=True)
    # trunk = models.wide_resnet50_2(pretrained=True)
    # trunk = EfficientNet.from_pretrained('efficientnet-b2')
    trunk_output_size = trunk.fc.in_features
    # TODO: @Raphael: Is this the last layer?
    trunk.fc = Identity()
    trunk = trunk.to(device)
    # trunk = trunk.to(device)

    # TODO: Why did he do this?
    embedder = MLP([trunk_output_size, cfg.embedder.embedder.size]).to(device)
    # embedder = MLP([trunk_output_size, cfg.embedder.embedder.size]).to(device)
    classifier = MLP([cfg.embedder.embedder.size, cfg.embedder.embedder.class_out_size]).to(
        device)
    # classifier = MLP([cfg.embedder.embedder.size, cfg.embedder.embedder.class_out_size]).to(device)

    # Set optimizers
    if cfg.optimizer.optimizer.name == "sdg":
        trunk_optimizer = torch.optim.SGD(trunk.parameters(), lr=cfg.optimizer.optimizer.lr,
                                          momentum=cfg.optimizer.optimizer.momentum,
                                          weight_decay=cfg.optimizer.optimizer.weight_decay)
        embedder_optimizer = torch.optim.SGD(embedder.parameters(), lr=cfg.optimizer.optimizer.lr,
                                             momentum=cfg.optimizer.optimizer.momentum,
                                             weight_decay=cfg.optimizer.optimizer.weight_decay)
        classifier_optimizer = torch.optim.SGD(classifier.parameters(), lr=cfg.optimizer.optimizer.lr,
                                               momentum=cfg.optimizer.optimizer.momentum,
                                               weight_decay=cfg.optimizer.optimizer.weight_decay)
    elif cfg.optimizer.optimizer.name == "rmsprop":
        trunk_optimizer = torch.optim.RMSprop(trunk.parameters(), lr=cfg.optimizer.optimizer.lr,
                                              momentum=cfg.optimizer.optimizer.momentum,
                                              weight_decay=cfg.optimizer.optimizer.weight_decay)
        embedder_optimizer = torch.optim.RMSprop(embedder.parameters(), lr=cfg.optimizer.optimizer.lr,
                                                 momentum=cfg.optimizer.optimizer.momentum,
                                                 weight_decay=cfg.optimizer.optimizer.weight_decay)
        classifier_optimizer = torch.optim.RMSprop(classifier.parameters(), lr=cfg.optimizer.optimizer.lr,
                                                   momentum=cfg.optimizer.optimizer.momentum,
                                                   weight_decay=cfg.optimizer.optimizer.weight_decay)

    # Set the datasets
    data_dir = cfg.dataset.dataset.data_dir
    print("Data dir: " + data_dir)

    train_dataset, val_dataset, val_samples_dataset = get_datasets(data_dir, cfg, mode=cfg.mode.mode.type)
    print("Trainset: ", len(train_dataset), "Testset: ", len(val_dataset), "Samplesset: ", len(val_samples_dataset))

    # Set the loss function
    if cfg.embedder_loss.embedder_loss.name == "margin_loss":
        loss = losses.MarginLoss(margin=cfg.embedder_loss.embedder_loss.margin, nu=cfg.embedder_loss.embedder_loss.nu,
                                 beta=cfg.embedder_loss.embedder_loss.beta)
    if cfg.embedder_loss.embedder_loss.name == "triplet_margin":
        loss = losses.TripletMarginLoss(margin=cfg.embedder_loss.embedder_loss.margin)
    if cfg.embedder_loss.embedder_loss.name == "multi_similarity":
        loss = losses.MultiSimilarityLoss(alpha=cfg.embedder_loss.embedder_loss.alpha,
                                          beta=cfg.embedder_loss.embedder_loss.beta,
                                          base=cfg.embedder_loss.embedder_loss.base)

    # Set the classification loss:
    classification_loss = torch.nn.CrossEntropyLoss()

    # Set the mining function

    if cfg.miner.miner.name == "triplet_margin":
        # miner = miners.TripletMarginMiner(margin=0.2)
        miner = miners.TripletMarginMiner(margin=cfg.miner.miner.margin)
    if cfg.miner.miner.name == "multi_similarity":
        miner = miners.MultiSimilarityMiner(epsilon=cfg.miner.miner.epsilon)
        # miner = miners.MultiSimilarityMiner(epsilon=0.05)

    # loss = losses.CrossBatchMemory(loss, cfg.embedder.embedder.size, memory_size=1024, miner=miner)
    # extra_str = "cb_mem"
    extra_str = ""

    batch_size = cfg.trainer.trainer.batch_size
    # 100 by default
    num_epochs = cfg.trainer.trainer.num_epochs
    iterations_per_epoch = cfg.trainer.trainer.iterations_per_epoch
    # Set the dataloader sampler
    sampler = samplers.MPerClassSampler(train_dataset.label, m=4, length_before_new_iter=len(train_dataset))
    # sampler = samplers.MPerClassSampler(train_dataset.label, m=4, length_before_new_iter=iterations_per_epoch)

    # Package the above stuff into dictionaries.
    models = {"trunk": trunk, "embedder": embedder, "classifier": classifier}
    optimizers = {"trunk_optimizer": trunk_optimizer, "embedder_optimizer": embedder_optimizer,
                  "classifier_optimizer": classifier_optimizer}
    loss_funcs = {"metric_loss": loss, "classifier_loss": classification_loss}
    mining_funcs = {"tuple_miner": miner}

    # We can specify loss weights if we want to. This is optional
    loss_weights = {"metric_loss": cfg.loss.loss.metric_loss, "classifier_loss": cfg.loss.loss.classifier_loss}

    schedulers = {
        # "metric_loss_scheduler_by_epoch": torch.optim.lr_scheduler.StepLR(classifier_optimizer, cfg.optimizer.scheduler.step_size, gamma=cfg.optimizer.scheduler.gamma),
        "embedder_scheduler_by_epoch": torch.optim.lr_scheduler.StepLR(embedder_optimizer,
                                                                       cfg.optimizer.scheduler.step_size,
                                                                       gamma=cfg.optimizer.scheduler.gamma),
        "classifier_scheduler_by_epoch": torch.optim.lr_scheduler.StepLR(classifier_optimizer,
                                                                         cfg.optimizer.scheduler.step_size,
                                                                         gamma=cfg.optimizer.scheduler.gamma),
        "trunk_scheduler_by_epoch": torch.optim.lr_scheduler.StepLR(embedder_optimizer,
                                                                    cfg.optimizer.scheduler.step_size,
                                                                    gamma=cfg.optimizer.scheduler.gamma),
    }

    experiment_name = "%s_model_%s_cl_%s_ml_%s_miner_%s_mix_ml_%02.2f_mix_cl_%02.2f_resize_%d_emb_size_%d_class_size_%d_opt_%s_lr_%02.2f_%s" % (
    cfg.dataset.dataset.name,
    cfg.model.model.model_name,
    "cross_entropy",
    cfg.embedder_loss.embedder_loss.name,
    cfg.miner.miner.name,
    cfg.loss.loss.metric_loss,
    cfg.loss.loss.classifier_loss,
    cfg.transform.transform.transform_resize,
    cfg.embedder.embedder.size,
    cfg.embedder.embedder.class_out_size,
    cfg.optimizer.optimizer.name,
    cfg.optimizer.optimizer.lr,
    extra_str
    # cfg.optimizer.optimizer.momentum,
    # cfg.optimizer.optimizer.weight_decay
    )

    if issubclass(train_class, WithAutocastTrainWithClassifier):
        trainer = train_class(cfg.use_amp.use_amp.use_amp,
                                                  models,
                                                  optimizers,
                                                  batch_size,
                                                  loss_funcs,
                                                  mining_funcs,
                                                  train_dataset,
                                                  # How the data gets sampled
                                                  sampler=sampler,
                                                  lr_schedulers=schedulers,
                                                  dataloader_num_workers=cfg.trainer.trainer.batch_size,
                                                  loss_weights=loss_weights
                                                  )

    elif issubclass(train_class, TrainWithClassifier):
        trainer = train_class(models,
                                                  optimizers,
                                                  batch_size,
                                                  loss_funcs,
                                                  mining_funcs,
                                                  train_dataset,
                                                  # How the data gets sampled
                                                  sampler=sampler,
                                                  lr_schedulers=schedulers,
                                                  dataloader_num_workers=cfg.trainer.trainer.batch_size,
                                                  loss_weights=loss_weights
                                                  )

    start_timer()
    trainer.train(num_epochs=num_epochs)
    time, mem = get_time_and_memory()
    return models, time, mem

def test_autocast_equivalence():
    with initialize(version_base=None, config_path="../../config", job_name="test_app"):
        # Run standard TrainWithClassifier - no amp!
        cfg = compose(config_name="test_trainer_use_amp_false")
        model_def, time_def, mem_def = train_app(cfg, TrainWithClassifier)
        embedder_def_params = list(model_def["embedder"].parameters())

    with initialize(version_base=None, config_path="../../config", job_name="test_app"):
        # Run WithAutoCastTrainWithClassifier, though AMP is disabled.
        cfg = compose(config_name="test_trainer_use_amp_false")
        model_amp_1, time_amp_1, mem_amp_1 = train_app(cfg, WithAutocastTrainWithClassifier)
        embedder_amp_1_params = list(model_amp_1["embedder"].parameters())

    with initialize(version_base=None, config_path="../../config", job_name="test_app"):
        # Run WithAutoCastTrainWithClassifier, though AMP is enabled.
        cfg = compose(config_name="test_trainer_use_amp_true")
        model_amp_2, time_amp_2, mem_amp_2 = train_app(cfg, WithAutocastTrainWithClassifier)
        embedder_amp_2_params = list(model_amp_2["embedder"].parameters())

    for params_1, params_2 in zip(embedder_def_params, embedder_amp_1_params):
        # Assert that TrainWithClassifier and WithAutoCastTrainWithClassifier without AMP results in same weights.
        params_1 = params_1.cpu().detach()
        params_2 = params_2.cpu().detach()
        assert np.array_equal(params_1, params_2)

    for params_1, params_2 in zip(embedder_amp_1_params, embedder_amp_2_params):
        # Assert that WithAutoCastTrainWithClassifier with and without AMP results in similar results.
        params_1 = params_1.cpu().detach()
        params_2 = params_2.cpu().detach()
        total = 0
        for compar in np.where(np.isclose(params_1, params_2)):
            total += len(compar)
        print(total)
        print(torch.numel(params_1))
        assert total > torch.numel(params_1) // 2

    # Assert that AMP results in better time and memory.
    print(time_amp_1)
    print(time_amp_2)
    assert time_amp_2 < time_amp_1

    print(mem_amp_1)
    print(mem_amp_2)
    assert mem_amp_2 < mem_amp_1