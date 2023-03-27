""" File for metric learning. """
from functools import partial
import logging
from shutil import rmtree
import sys
import os
import os.path as osp

import omegaconf.errors

sys.path.append(osp.dirname(osp.dirname(osp.dirname(__file__))))

# The testing module requires faiss
# So if you don't have that, then this import will break
from pytorch_metric_learning import losses, miners, samplers, trainers, testers, utils
import torch.nn as nn
import record_keeper
import pytorch_metric_learning.utils.logging_presets as logging_presets
#from torchvision import datasets, models, transforms
#import torchvision
import yaml
import warnings
from argparse import ArgumentParser
logging.getLogger().setLevel(logging.INFO)

import pytorch_metric_learning
from pytorch_metric_learning.testers.base_tester import BaseTester

logging.info("pytorch-metric-learning VERSION %s"%pytorch_metric_learning.__version__)
logging.info("record_keeper VERSION %s"%record_keeper.__version__)

from sklearn.metrics import accuracy_score
#from efficientnet_pytorch import EfficientNet
import torch
import numpy as np
import pickle

import hydra
from omegaconf import DictConfig, OmegaConf

from model import agcn, msg3d
from MasterThesis.STTTFormer.model import sttformer
from MasterThesis.HD_GCN_main.model import HDGCN
from graph import ntu_rgb_d
from feeders import feeder
from trainer.with_autocast_train_with_classifier import WithAutocastTrainWithClassifier
from tester.with_autocast_one_shot_tester import WithAMPGlobalEmbeddingSpaceTester

# reprodcibile
np.random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Constants
STTFORMER = "sttformer"
HD_GCN = "hd-gcn"
MSG3D = "msg3d"

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


def get_datasets(model_name, data_path, label_path, name_path, val_classes, mem_limits={"val_samples": 0, "val": 0, "train": 0}, debug=False, data_kwargs={}):

    feeder_class = feeder.Feeder
    if model_name == STTFORMER or HD_GCN in model_name:
        from MasterThesis.STTTFormer.feeders import feeder_ntu
        feeder_class = feeder_ntu.Feeder

    val_sample_names = []

    for i in val_classes:
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


    train_dataset, val_dataset, val_samples_dataset = feeder.get_train_and_os_val(feeder_class=feeder_class,
                                                                                  data_path=data_path,
                                                                                  label_path=label_path,
                                                                                  name_path=name_path,
                                                                                  val_classes=val_classes,
                                                                                  val_sample_names=val_sample_names,
                                                                                  mem_limits=mem_limits, debug=debug,
                                                                                  data_kwargs=data_kwargs)

    return train_dataset, val_dataset, val_samples_dataset

def stt_hook(trainer, warm_up_epoch, base_lr, lr_decay_rate, step):
    # TODO: Gotta see if I do it for all optimizers or not
    for optimizer in trainer.optimizers.values():
        if isinstance(optimizer, torch.optim.SGD) or isinstance(optimizer, torch.optim.Adam):
            epoch = 0
            try:
                epoch = trainer.epoch
            except AttributeError:
                pass
            if epoch < warm_up_epoch:
                new_lr = base_lr * (epoch + 1) / warm_up_epoch
            else:
                new_lr = base_lr * (lr_decay_rate ** np.sum(epoch >= np.array(step)))
            # TODO: Gotta see if I do it for all optimizers or not
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr
            return new_lr
        else:
            raise ValueError()

def get_end_of_epoch_hook(hooks, orig_end_of_epoch_hook, model_name, val_size, save_epochs, **kwargs):

    def dummy_hook(trainer):
        pass

    actual_hook_to_use = dummy_hook
    if model_name == STTFORMER:
        actual_hook_to_use = stt_hook

    custom_hook = partial(actual_hook_to_use, **kwargs)

    # If training on dataset with val split
    if val_size:
        def true_hook(trainer):
            custom_hook(trainer)
            return orig_end_of_epoch_hook(trainer)
    # If training on full dataset
    else:
        def true_hook(trainer):
            custom_hook(trainer)
            # Only save models of epochs in save_epochs
            save_model_dir = "example_saved_models"
            if osp.exists(save_model_dir):
                for filename in os.listdir(save_model_dir):
                    for save_epoch in save_epochs:
                        if str(save_epoch) in filename:
                            break
                    else:
                        file_path = os.path.join(save_model_dir, filename)
                        os.unlink(file_path)
            hooks.do_save_models = True
            hooks.save_models(trainer, save_model_dir, str(trainer.epoch))
            return True

    return true_hook

def hd_gcn_hook(trainer, warm_up_epoch, base_lr, lr_ratio, num_epochs):
    for optimizer in trainer.optimizers.values():
        if isinstance(optimizer, torch.optim.SGD) or isinstance(optimizer, torch.optim.Adam):
            epoch = 0
            try:
                epoch = trainer.epoch
            except AttributeError:
                pass
            if epoch < warm_up_epoch:
                new_lr = base_lr * (epoch + 1) / warm_up_epoch
            else:
                # lr = self.arg.base_lr * (
                #         self.arg.lr_decay_rate ** np.sum(epoch >= np.array(self.arg.step)))
                T_max = len(trainer.dataloader) * (num_epochs - warm_up_epoch)
                T_cur = len(trainer.dataloader) * (epoch - warm_up_epoch) + trainer.iteration

                eta_min = base_lr * lr_ratio
                new_lr = eta_min + 0.5 * (base_lr - eta_min) * (1 + np.cos((T_cur / T_max) * np.pi))
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr
            return new_lr
        else:
            raise ValueError()

def get_end_of_iteration_hook(hooks, orig_end_of_epoch_hook, model_name, **kwargs):
    from functools import partial

    def dummy_hook(trainer):
        pass

    actual_hook_to_use = dummy_hook
    if HD_GCN in model_name:
        actual_hook_to_use = hd_gcn_hook

    custom_hook = partial(actual_hook_to_use, **kwargs)

    def true_hook(trainer):
        custom_hook(trainer)
        orig_end_of_epoch_hook(trainer)

    return true_hook


def get_trunk(cfg):
    if cfg.model.model_name == STTFORMER:
        trunk = sttformer.Model(**cfg.model.model_args)
        trunk.fc = nn.Identity()
        trunk_output_size = cfg.model.model_args["config"][-1][1]
    elif HD_GCN in cfg.model.model_name:
        trunk = HDGCN.Model(**cfg.model.model_args)
        trunk_output_size = trunk.fc.in_features
        trunk.fc = nn.Identity()
    elif cfg.model.model_name == MSG3D:
        trunk = msg3d.Model(graph="graph.ntu_rgb_d.AdjMatrixGraph", num_class=100, num_point=25, num_person=2, num_gcn_scales=13, num_g3d_scales=6)
        trunk_output_size = trunk.fc.in_features

        trunk.fc = nn.Identity()
    else:
        raise NotImplementedError("Unsupported model")

    return trunk, trunk_output_size


@hydra.main(config_path="config", version_base="1.1", )
def train_app(cfg):
    print(cfg)

    # Set the datasets
    data_dir = cfg.dataset.data_dir
    print("Data dir: "+data_dir)

    ds_mem_limits = cfg.dataset.dataset_split.mem_limits
    mem_limits = {"val_samples": ds_mem_limits.val_samples, "val": ds_mem_limits.val, "train": ds_mem_limits.train}

    train_dataset, val_dataset, val_samples_dataset = get_datasets(cfg.model.model_name,
                                                                   cfg.dataset.data_path, cfg.dataset.label_path,
                                                                   cfg.dataset.name_path,
                                                                   cfg.dataset.dataset_split.val_classes,
                                                                   mem_limits, debug=cfg.dataset.debug, data_kwargs=cfg.dataset.data_kwargs)
    print("Trainset: ",len(train_dataset), "Testset: ",len(val_dataset), "Samplesset: ",len(val_samples_dataset))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trunk, trunk_output_size = get_trunk(cfg)
    trunk = trunk.to(device)
    embedder = MLP([trunk_output_size, cfg.embedder.size]).to(device)
    # Set output size to the number of classes in training data
    classifier = MLP([cfg.embedder.size, train_dataset.label.max() + 1]).to(device)

    # Set optimizers
    if cfg.optimizer.optimizer.name == "sgd":
        trunk_optimizer = torch.optim.SGD(trunk.parameters(), lr=cfg.optimizer.optimizer.lr, momentum=cfg.optimizer.optimizer.momentum, weight_decay=cfg.optimizer.optimizer.weight_decay, nesterov=cfg.optimizer.optimizer.nesterov)
        embedder_optimizer = torch.optim.SGD(embedder.parameters(), lr=cfg.optimizer.optimizer.lr, momentum=cfg.optimizer.optimizer.momentum, weight_decay=cfg.optimizer.optimizer.weight_decay, nesterov=cfg.optimizer.optimizer.nesterov)
        classifier_optimizer = torch.optim.SGD(classifier.parameters(), lr=cfg.optimizer.optimizer.lr, momentum=cfg.optimizer.optimizer.momentum, weight_decay=cfg.optimizer.optimizer.weight_decay, nesterov=cfg.optimizer.optimizer.nesterov)
    elif cfg.optimizer.optimizer.name == "rmsprop":
        trunk_optimizer = torch.optim.RMSprop(trunk.parameters(), lr=cfg.optimizer.optimizer.lr, momentum=cfg.optimizer.optimizer.momentum, weight_decay=cfg.optimizer.optimizer.weight_decay)
        embedder_optimizer = torch.optim.RMSprop(embedder.parameters(), lr=cfg.optimizer.optimizer.lr, momentum=cfg.optimizer.optimizer.momentum, weight_decay=cfg.optimizer.optimizer.weight_decay)
        classifier_optimizer = torch.optim.RMSprop(classifier.parameters(), lr=cfg.optimizer.optimizer.lr, momentum=cfg.optimizer.optimizer.momentum, weight_decay=cfg.optimizer.optimizer.weight_decay)
    else:
        raise NotImplementedError("Unsupported optimizer")

    # Set the loss function
    if cfg.embedder_loss.name == "margin_loss":
        loss = losses.MarginLoss(margin=cfg.embedder_loss.margin, nu=cfg.embedder_loss.nu, beta=cfg.embedder_loss.beta)
    if cfg.embedder_loss.name == "triplet_margin":
        loss = losses.TripletMarginLoss(margin=cfg.embedder_loss.margin)
    if cfg.embedder_loss.name == "multi_similarity":
        loss = losses.MultiSimilarityLoss(alpha=cfg.embedder_loss.alpha, beta=cfg.embedder_loss.beta, base=cfg.embedder_loss.base)

    # Set the classification loss:
    classification_loss = torch.nn.CrossEntropyLoss()

    if cfg.dataset.debug:
        # def inspect_grads(module, grad_input, grad_output):
        #     print(module)
        #     print(grad_input)
        #     print(grad_output)
        #     return (grad_input, grad_output)
        #
        # loss.register_full_backward_hook(inspect_grads)
#        classification_loss.register_full_backward_hook(inspect_grads)
        pass

    # Set the mining function
    if cfg.miner.name == "triplet_margin":
        miner = miners.TripletMarginMiner(margin=cfg.miner.margin)
    if cfg.miner.name == "multi_similarity":
        miner = miners.MultiSimilarityMiner(epsilon=cfg.miner.epsilon)

    extra_str = ""

    batch_size = cfg.trainer.batch_size
    num_epochs = cfg.trainer.num_epochs
    # Set the dataloader sampler
    sampler = samplers.MPerClassSampler(train_dataset.label, m=4, length_before_new_iter=len(train_dataset))

    # Package the above stuff into dictionaries.
    models = {"trunk": trunk, "embedder": embedder, "classifier": classifier}
    optimizers = {"trunk_optimizer": trunk_optimizer, "embedder_optimizer": embedder_optimizer, "classifier_optimizer": classifier_optimizer}
    loss_funcs = {"metric_loss": loss, "classifier_loss": classification_loss}
    mining_funcs = {"tuple_miner": miner}

    # We can specify loss weights if we want to. This is optional
    loss_weights = {"metric_loss": cfg.loss.metric_loss, "classifier_loss": cfg.loss.classifier_loss}

    schedulers = None
    if cfg.model.model_name == MSG3D:
        schedulers = {
                #"metric_loss_scheduler_by_epoch": torch.optim.lr_scheduler.StepLR(classifier_optimizer, cfg.optimizer.scheduler.step_size, gamma=cfg.optimizer.scheduler.gamma),
                "embedder_scheduler_by_epoch": torch.optim.lr_scheduler.StepLR(embedder_optimizer, cfg.optimizer.scheduler.step_size, gamma=cfg.optimizer.scheduler.gamma),
                "classifier_scheduler_by_epoch": torch.optim.lr_scheduler.StepLR(classifier_optimizer, cfg.optimizer.scheduler.step_size, gamma=cfg.optimizer.scheduler.gamma),
                "trunk_scheduler_by_epoch": torch.optim.lr_scheduler.StepLR(embedder_optimizer, cfg.optimizer.scheduler.step_size, gamma=cfg.optimizer.scheduler.gamma),
                }

    # Be careful the experiment_name isn't too long --> can exceed char limit for folder name and thus lead to bugs
    # experiment_name = "%s_model_%s_cl_%s_ml_%s_miner_%s_mix_ml_%02.2f_mix_cl_%02.2f_emb_size_%d_class_size_%d_opt_%s_lr_%02.2f_%s"%(cfg.dataset.name,
    #                                                                                               cfg.model.model_name,
    #                                                                                               "cross_entropy",
    #                                                                                               cfg.embedder_loss.name,
    #                                                                                               cfg.miner.name,
    #                                                                                               cfg.loss.metric_loss,
    #                                                                                               cfg.loss.classifier_loss,
    #                                                                                               cfg.embedder.size,
    #                                                                                               cfg.embedder.class_out_size,
    #                                                                                               cfg.optimizer.optimizer.name,
    #                                                                                               cfg.optimizer.optimizer.lr,
    #                                                                                               extra_str
    #                                                                                               )
    dataset_dict = {"samples": val_samples_dataset, "val": val_dataset}
    experiment_name = hydra.core.hydra_config.HydraConfig.get().job.config_name
    record_keeper, _, _ = logging_presets.get_record_keeper("logs", "tensorboard",
                                                            is_new_experiment=cfg.mode.type=="train_from_scratch")
    hooks = logging_presets.get_hook_container(record_keeper, primary_metric=cfg.tester.metric)
    model_folder = "example_saved_models"

    # Create the tester
    tester = WithAMPGlobalEmbeddingSpaceTester(
        cfg.tester.use_amp,
        batch_size=cfg.tester.batch_size,
        dataloader_num_workers=cfg.tester.dataloader_num_workers,
            end_of_testing_hook=hooks.end_of_testing_hook,
            #size_of_tsne=20
            )

    tester.embedding_filename=data_dir+"/"+experiment_name+".pkl"
    # Records metric after each epoch on one-shot validation data.

    # Define potentially custom end_of_epoch_hook
    orig_end_of_epoch_hook = hooks.end_of_epoch_hook(tester, dataset_dict, model_folder, splits_to_eval=[('val', ['samples'])])
    epoch_hook_kwargs = {}
    try:
        epoch_hook_kwargs = cfg.end_of_epoch_hook.kwargs
    except omegaconf.errors.ConfigAttributeError:
        pass
    end_of_epoch_hook = get_end_of_epoch_hook(hooks, orig_end_of_epoch_hook, cfg.model.model_name, len(val_dataset), cfg.trainer.save_epochs, **epoch_hook_kwargs)

    # ... and end_of_iteration_hook
    orig_end_of_iteration_hook = hooks.end_of_iteration_hook
    iteration_hook_kwargs = {}
    try:
        iteration_hook_kwargs = cfg.end_of_iteration_hook.kwargs
    except omegaconf.errors.ConfigAttributeError:
        pass
    end_of_iteration_hook = get_end_of_iteration_hook(hooks, orig_end_of_iteration_hook, cfg.model.model_name, **iteration_hook_kwargs)

    # Training for metric learning
    trainer = WithAutocastTrainWithClassifier(cfg.trainer.use_amp, models,
            optimizers,
            batch_size,
            loss_funcs,
            mining_funcs,
            train_dataset,
            iterations_per_epoch=cfg.trainer.iterations_per_epoch,
            # How the data gets sampled
            sampler=sampler,
            lr_schedulers=schedulers,
            dataloader_num_workers=cfg.trainer.dataloader_num_workers,
            loss_weights=loss_weights,
            end_of_iteration_hook=end_of_iteration_hook,
            end_of_epoch_hook=end_of_epoch_hook
            )

    if cfg.model.model_name == STTFORMER:
        assert cfg.end_of_epoch_hook.kwargs.base_lr == cfg.optimizer.optimizer.lr
        stt_hook(trainer, **cfg.end_of_epoch_hook.kwargs)

    elif cfg.model.model_name == HD_GCN:
        assert cfg.end_of_iteration_hook.kwargs.base_lr == cfg.optimizer.optimizer.lr
        assert cfg.end_of_iteration_hook.kwargs.num_epochs == cfg.trainer.num_epochs
        hd_gcn_hook(trainer, **cfg.end_of_iteration_hook.kwargs)

    start_epoch = 1
    if cfg.mode.type in ("train_from_latest", "fine-tune"):
        try:
            with open(osp.join(osp.dirname(cfg.mode.model_folder), ".hydra/config.yaml"), "r") as f:
                old_config = yaml.load(f, yaml.Loader)
                if not old_config == cfg:
                    raise ValueError("Old configuration and current configuration differ!")
            best = cfg.mode.type == "fine-tune"
            start_epoch = hooks.load_latest_saved_models(trainer, cfg.mode.model_folder, device=device, best=best)
        except (TypeError):
            raise ValueError('Specify cfg.mode.model_folder. You can do this by adding e.g. '
                             '+mode.model_name=your_model_folder as a command-line argument.')

    trainer.train(start_epoch=start_epoch, num_epochs=num_epochs)

if __name__ == "__main__":
    # parser = ArgumentParser()
    # parser.add_argument('--mode', help='Pick between "train_from_scratch", "train_from_latest", "fine-tune"')
    # parser.add_argument('--model_folder', help='If mode is "train_from_latest" or "fine-tune", specify the model'
    #                                            'folder to load the model from', default=None)

    from omegaconf import OmegaConf

    with open(osp.join("config", sys.argv[sys.argv.index("--config-name") + 1] + ".yaml")) as cfg_file:
        cfg = yaml.load(cfg_file, yaml.Loader)["defaults"]

    mode_type = None
    model_folder = None

    mode_str = "+mode.type="
    model_folder_str = "+mode.model_folder="

    for i, arg in enumerate(sys.argv):
        if mode_str in arg:
            # arg.find(mode_str) + len(mode_str)
            mode_type = arg[arg.find(mode_str) + len(mode_str):]
        elif model_folder_str in arg:
            model_folder = arg[arg.find(model_folder_str) + len(model_folder_str):]
    if mode_type in ("train_from_latest", "fine-tune"):
        with open(osp.join(osp.dirname(model_folder), ".hydra/config.yaml"), "r") as f:
            old_config = yaml.load(f, yaml.Loader)
            old_config.pop("mode", None)
            for a_config in cfg:
                k = list(a_config.keys())[0]
                v = list(a_config.values())[0]
                with open(osp.join("config", k, v + ".yaml")) as f:
                    sub_cfg = yaml.load(f, yaml.Loader)
                    if not old_config[k] == sub_cfg:
                        err_msg = "Old configuration and current configuration differ!" \
                                  + "\n" + "old sub config: " + str(old_config[k])\
                                  + "\n" + "new sub config: " + str(sub_cfg)

                        if k == "dataset":
                            old_dataset_cfg = old_config[k].copy()
                            sub_dataset_cfg = sub_cfg.copy()

                            # If just the mem_limits are different, it's no problem that configs differ
                            del old_dataset_cfg["dataset_split"]["mem_limits"]
                            del sub_dataset_cfg["dataset_split"]["mem_limits"]

                            if not old_dataset_cfg == sub_dataset_cfg:
                                raise ValueError(err_msg)
                        else:
                            raise ValueError(err_msg)

    elif mode_type == "train_from_scratch":
        old_out_dir = osp.join("outputs", sys.argv[sys.argv.index("--config-name") + 1])
        if osp.isdir(osp.join("outputs", sys.argv[sys.argv.index("--config-name") + 1])):
            answer = input("Output directory for this config already exists. "
                           "Do you want to delete old directory? If yes, type 'yes'. Otherwise, type anything else: ")
            if answer == "yes":
                print("Removing directory!")
                for filename in os.listdir(old_out_dir):
                    file_path = os.path.join(old_out_dir, filename)
                    for dataset_path in feeder.VAL_DATASET + feeder.VAL_SAMPLES_DATASET + feeder.TRAIN_DATASET:
                        if dataset_path in file_path:
                            break
                    else:
                        if os.path.isfile(file_path) or os.path.islink(file_path):
                            os.unlink(file_path)
                        elif os.path.isdir(file_path):
                            rmtree(file_path)
            else:
                raise ValueError("Directory for config already exists.")
    else:
        raise ValueError("Unknown cfg.mode.type: " + mode_type)

    train_app()
