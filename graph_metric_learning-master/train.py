""" File for metric learning. """

# The testing module requires faiss
# So if you don't have that, then this import will break
from pytorch_metric_learning import losses, miners, samplers, trainers, testers, utils
import torch.nn as nn
import record_keeper
import pytorch_metric_learning.utils.logging_presets as logging_presets
#from torchvision import datasets, models, transforms
#import torchvision
import logging
logging.getLogger().setLevel(logging.INFO)
import os.path as osp

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
from omegaconf import DictConfig

from model import agcn, msg3d
from MasterThesis.STTTFormer.model.sttformer import Model
from graph import ntu_rgb_d
from feeders import feeder
from trainer.with_autocast_train_with_classifier import WithAutocastTrainWithClassifier
from tester.with_autocast_one_shot_tester import WithAMPGlobalEmbeddingSpaceTester

# reprodcibile
np.random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


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


# This is for replacing the last layer of a pretrained network.
# This code is from https://github.com/KevinMusgrave/powerful_benchmarker
class Identity(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


def get_datasets(val_classes, mem_limits={"val_samples": 0, "val": 0, "train": 0}, debug=False):
    data_path = "/home/work/PycharmProjects/MA/MasterThesis/graph_metric_learning-master"

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


    train_dataset, val_dataset, val_samples_dataset = feeder.get_train_and_os_val(
        data_path=osp.join(data_path, "data/ntu/one_shot/train_data_joint.npy"),
        label_path=osp.join(data_path, "data/ntu/one_shot/train_label.pkl"),
        val_classes=val_classes, val_sample_names=val_sample_names,
        mem_limits=mem_limits, debug=debug)

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

def get_end_of_epoch_hook(orig_end_of_epoch_hook, model_name, **kwargs):
    from functools import partial

    def dummy_hook(trainer):
        pass

    actual_hook_to_use = dummy_hook
    if model_name == "sttformer":
        actual_hook_to_use = stt_hook

    custom_hook = partial(actual_hook_to_use, **kwargs)

    def true_hook(trainer):
        custom_hook(trainer)
        return orig_end_of_epoch_hook(trainer)

    return true_hook

@hydra.main(config_path="config")
def train_app(cfg):
    print(cfg)

    # Set the datasets
    data_dir = cfg.dataset.data_dir
    print("Data dir: "+data_dir)

    ds_mem_limits = cfg.dataset.dataset_split.mem_limits
    mem_limits = {"val_samples": ds_mem_limits.val_samples, "val": ds_mem_limits.val, "train": ds_mem_limits.train}

    train_dataset, val_dataset, val_samples_dataset = get_datasets(cfg.dataset.dataset_split.val_classes,
                                                                   mem_limits, debug=cfg.dataset.debug)
    print("Trainset: ",len(train_dataset), "Testset: ",len(val_dataset), "Samplesset: ",len(val_samples_dataset))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set trunk model and replace the softmax layer with an identity function
    #trunk = torchvision.models.__dict__[cfg.model.model_name](pretrained=cfg.model.pretrained)
    #graph = ntu_rgb_d.Graph()
    #trunk = agcn.Model(graph="graph.ntu_rgb_d.Graph")
    # TODO: Import STTformer or any other specified model instead
    trunk = msg3d.Model(graph="graph.ntu_rgb_d.AdjMatrixGraph", num_class=100, num_point=25, num_person=2, num_gcn_scales=13, num_g3d_scales=6)
    
    #resnet18(pretrained=True)
    #trunk = models.alexnet(pretrained=True)
    #trunk = models.resnet50(pretrained=True)
    #trunk = models.resnet152(pretrained=True)
    #trunk = models.wide_resnet50_2(pretrained=True)
    #trunk = EfficientNet.from_pretrained('efficientnet-b2')
    trunk_output_size = trunk.fc.in_features

    trunk.fc = Identity()
    trunk = trunk.to(device)
    # trunk = trunk.to(device)

    embedder = MLP([trunk_output_size, cfg.embedder.size]).to(device)
    # embedder = MLP([trunk_output_size, cfg.embedder.size]).to(device)
    # Set output size to the number of classes in training data
    classifier = MLP([cfg.embedder.size, train_dataset.label.max() + 1]).to(device)
    # classifier = MLP([cfg.embedder.size, cfg.embedder.class_out_size]).to(device)

    # Set optimizers
    if cfg.optimizer.optimizer.name == "sdg":
        trunk_optimizer = torch.optim.SGD(trunk.parameters(), lr=cfg.optimizer.optimizer.lr, momentum=cfg.optimizer.optimizer.momentum, weight_decay=cfg.optimizer.optimizer.weight_decay, nesterov=cfg.optimizer.optimizer.nesterov)
        embedder_optimizer = torch.optim.SGD(embedder.parameters(), lr=cfg.optimizer.optimizer.lr, momentum=cfg.optimizer.optimizer.momentum, weight_decay=cfg.optimizer.optimizer.weight_decay, nesterov=cfg.optimizer.optimizer.nesterov)
        classifier_optimizer = torch.optim.SGD(classifier.parameters(), lr=cfg.optimizer.optimizer.lr, momentum=cfg.optimizer.optimizer.momentum, weight_decay=cfg.optimizer.optimizer.weight_decay, nesterov=cfg.optimizer.optimizer.nesterov)
    elif cfg.optimizer.optimizer.name == "rmsprop":
        trunk_optimizer = torch.optim.RMSprop(trunk.parameters(), lr=cfg.optimizer.optimizer.lr, momentum=cfg.optimizer.optimizer.momentum, weight_decay=cfg.optimizer.optimizer.weight_decay)
        embedder_optimizer = torch.optim.RMSprop(embedder.parameters(), lr=cfg.optimizer.optimizer.lr, momentum=cfg.optimizer.optimizer.momentum, weight_decay=cfg.optimizer.optimizer.weight_decay)
        classifier_optimizer = torch.optim.RMSprop(classifier.parameters(), lr=cfg.optimizer.optimizer.lr, momentum=cfg.optimizer.optimizer.momentum, weight_decay=cfg.optimizer.optimizer.weight_decay)

    # Set the loss function
    if cfg.embedder_loss.name == "margin_loss":
        loss = losses.MarginLoss(margin=cfg.embedder_loss.margin,nu=cfg.embedder_loss.nu,beta=cfg.embedder_loss.beta)
    if cfg.embedder_loss.name == "triplet_margin":
        loss = losses.TripletMarginLoss(margin=cfg.embedder_loss.margin)
    if cfg.embedder_loss.name == "multi_similarity":
        loss = losses.MultiSimilarityLoss(alpha=cfg.embedder_loss.alpha, beta=cfg.embedder_loss.beta, base=cfg.embedder_loss.base)

    # Set the classification loss:
    classification_loss = torch.nn.CrossEntropyLoss()

    # Set the mining function

    if cfg.miner.name == "triplet_margin":
        #miner = miners.TripletMarginMiner(margin=0.2)
        miner = miners.TripletMarginMiner(margin=cfg.miner.margin)
    if cfg.miner.name == "multi_similarity":
        miner = miners.MultiSimilarityMiner(epsilon=cfg.miner.epsilon)
        #miner = miners.MultiSimilarityMiner(epsilon=0.05)

    #loss = losses.CrossBatchMemory(loss, cfg.embedder.size, memory_size=1024, miner=miner) 
    #extra_str = "cb_mem"
    extra_str = ""

    batch_size = cfg.trainer.batch_size
    # 100 by default
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
    if not cfg.model.model_name == "sttformer":
        schedulers = {
                #"metric_loss_scheduler_by_epoch": torch.optim.lr_scheduler.StepLR(classifier_optimizer, cfg.optimizer.scheduler.step_size, gamma=cfg.optimizer.scheduler.gamma),
                "embedder_scheduler_by_epoch": torch.optim.lr_scheduler.StepLR(embedder_optimizer, cfg.optimizer.scheduler.step_size, gamma=cfg.optimizer.scheduler.gamma),
                "classifier_scheduler_by_epoch": torch.optim.lr_scheduler.StepLR(classifier_optimizer, cfg.optimizer.scheduler.step_size, gamma=cfg.optimizer.scheduler.gamma),
                "trunk_scheduler_by_epoch": torch.optim.lr_scheduler.StepLR(embedder_optimizer, cfg.optimizer.scheduler.step_size, gamma=cfg.optimizer.scheduler.gamma),
                }

    experiment_name = "%s_model_%s_cl_%s_ml_%s_miner_%s_mix_ml_%02.2f_mix_cl_%02.2f_resize_%d_emb_size_%d_class_size_%d_opt_%s_lr_%02.2f_%s"%(cfg.dataset.name,
                                                                                                  cfg.model.model_name, 
                                                                                                  "cross_entropy", 
                                                                                                  cfg.embedder_loss.name, 
                                                                                                  cfg.miner.name, 
                                                                                                  cfg.loss.metric_loss, 
                                                                                                  cfg.loss.classifier_loss,
                                                                                                  cfg.transform.transform_resize,
                                                                                                  cfg.embedder.size,
                                                                                                  cfg.embedder.class_out_size,
                                                                                                  cfg.optimizer.optimizer.name,
                                                                                                  cfg.optimizer.optimizer.lr,
                                                                                                  extra_str
                                                                                                  )
    record_keeper, _, _ = logging_presets.get_record_keeper("logs/%s"%(experiment_name), "tensorboard/%s"%(experiment_name))
    hooks = logging_presets.get_hook_container(record_keeper, primary_metric=cfg.tester.metric)
    dataset_dict = {"samples": val_samples_dataset, "val": val_dataset}
    model_folder = "example_saved_models/%s/"%(experiment_name)

    # Create the tester
    # TODO: Change back
    tester = WithAMPGlobalEmbeddingSpaceTester(
        cfg.tester.use_amp,
        cfg.tester.batch_size,
        cfg.tester.dataloader_num_workers,
            end_of_testing_hook=hooks.end_of_testing_hook, 
            #size_of_tsne=20
            )
    #tester.embedding_filename=data_dir+"/embeddings_pretrained_triplet_loss_multi_similarity_miner.pkl"
    tester.embedding_filename=data_dir+"/"+experiment_name+".pkl"
    # Records metric after each epoch on one-shot validation data.
    orig_end_of_epoch_hook = hooks.end_of_epoch_hook(tester, dataset_dict, model_folder)
    end_of_epoch_hook = get_end_of_epoch_hook(orig_end_of_epoch_hook, cfg.model.model_name, **cfg.end_of_epoch_hook.kwargs)
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
            end_of_iteration_hook=hooks.end_of_iteration_hook,
            end_of_epoch_hook=end_of_epoch_hook
            )

    if cfg.model.model_name == "sttformer":
        assert cfg.end_of_epoch_hook.kwargs.base_lr == cfg.optimizer.optimizer.lr
        stt_hook(trainer, **cfg.end_of_epoch_hook.kwargs)

    trainer.train(num_epochs=num_epochs)


if __name__ == "__main__":
    train_app()
