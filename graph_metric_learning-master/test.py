import logging

from tqdm import tqdm
import json
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader
import hydra
from pytorch_metric_learning.utils import logging_presets, inference, accuracy_calculator
from pytorch_metric_learning.utils import common_functions as c_f
from sklearn.metrics import silhouette_score, confusion_matrix, ConfusionMatrixDisplay, adjusted_mutual_info_score, \
    normalized_mutual_info_score
import matplotlib.pyplot as plt
import umap
from cycler import cycler

from train import get_trunk, MLP, get_datasets
from tester.with_autocast_one_shot_tester import WithAMPGlobalEmbeddingSpaceTester


def visualizer_hook(umapper, umap_embeddings, labels, split_name, keyname, *args):
    logging.info(
        "UMAP plot for the {} split and label set {}".format(split_name, keyname)
    )
    label_set = np.unique(labels)
    num_classes = len(label_set)
    plt.figure(figsize=(20, 15))
    plt.gca().set_prop_cycle(
        cycler(
            "color", [plt.cm.nipy_spectral(i) for i in np.linspace(0, 0.9, num_classes)]
        )
    )

    if umapper is None:
        labels = labels.flatten()
    for i in range(num_classes):
        idx = labels == label_set[i]
        plt.scatter(umap_embeddings[idx, 0], umap_embeddings[idx, 1], marker=".", s=1, label=label_set[i] + 1)

    plt.legend()
    plt.savefig("UMAP_split_{}_label_set_{}.png".format(split_name.upper(), keyname.upper()))
    plt.show()


@hydra.main(config_path="config", version_base="1.1", )
def main(cfg):
    # Dump the original config of the model that gets tested
    with open(cfg.mode.old_config, "r") as f:
        old_config = yaml.load(f, yaml.Loader)
        with open("old_config.yaml", "w") as out_file:
            yaml.dump(old_config, out_file)

    # Set the datasets
    data_dir = cfg.dataset.data_dir
    print("Data dir: " + data_dir)

    ds_mem_limits = cfg.dataset.dataset_split.mem_limits
    mem_limits = {"val_samples": ds_mem_limits.val_samples, "val": ds_mem_limits.val, "train": ds_mem_limits.train}

    _, test_dataset, test_samples_dataset = get_datasets(cfg.model.model_name, cfg.dataset.data_path,
                                                       cfg.dataset.label_path, cfg.dataset.name_path,
                                                                   cfg.dataset.dataset_split.val_classes,
                                                                   mem_limits, debug=cfg.dataset.debug,
                                                                   data_kwargs=cfg.dataset.data_kwargs)

    print("Testset: ", len(test_dataset), "Samplesset: ", len(test_samples_dataset))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trunk, trunk_output_size = get_trunk(cfg)
    trunk = trunk.to(device)
    embedder = MLP([trunk_output_size, cfg.embedder.size]).to(device)
    # Set output size to the number of classes in training data
    classifier = MLP([cfg.embedder.size, cfg.num_train_labels]).to(device)

    # Package the above stuff into dictionaries.
    models = {"trunk": trunk, "embedder": embedder, "classifier": classifier}

    try:
        epoch, model_suffix = c_f.latest_version(
            cfg.mode.model_folder, "trunk_*.pth", best=cfg.mode.use_best)
    except ValueError:
        print('If the mode config uses use_best: true, maybe there is no saved model with'
              '"best" in the .pth files. If so, just add "best" before the epoch number')
        raise

    c_f.load_dict_of_models(
        models, model_suffix, cfg.mode.model_folder, device, log_if_successful=True
    )

    # Confusion matrix
    im = inference.InferenceModel(trunk, embedder)
    im.train_knn(test_samples_dataset)

    test_dataloader = DataLoader(test_dataset, cfg.dataset.data_loader.batch_size,
                                 num_workers=cfg.dataset.data_loader.num_workers, pin_memory=True)
    true_labels = np.empty(0)
    pred_labels = []

    # Temporarily disable logger to avoid spamming
    c_f.LOGGER.propagate = False

    for input_batch, label_batch in tqdm(test_dataloader):
        _, indices = im.get_nearest_neighbors(input_batch, k=1)
        true_labels = np.append(true_labels, label_batch.cpu().numpy())
        pred_labels += [test_samples_dataset.__getitem__(x[0])[1] for x in indices.cpu().numpy()]

    # Enable logger again
    c_f.LOGGER.propagate = True

    # Compute and save confusion matrix
    cm = confusion_matrix(pred_labels, true_labels, labels=test_samples_dataset.label)
    with open("confusion_matrix.npy", "wb") as f:
        np.save(f, cm)

    # Display confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=test_samples_dataset.label + 1)
    _, ax = plt.subplots(figsize=(10, 10))
    disp.plot(ax=ax)
    plt.savefig("confusion_matrix.png")
    plt.show()

    # Create the tester
    tester = WithAMPGlobalEmbeddingSpaceTester(
        use_amp=cfg.tester.use_amp,
        batch_size=cfg.tester.batch_size,
        dataloader_num_workers=cfg.tester.dataloader_num_workers,
        visualizer=umap.UMAP(),
        visualizer_hook=visualizer_hook,
        accuracy_calculator=accuracy_calculator.AccuracyCalculator(k=1)
    )

    test_results = {"test": dict()}

    # FAQ: Most of the metrics of Pytorch Metric Learning do not use the true label, but labels predicted via
    # k-means-clustering and thus do not record the metrics the way we would be interested in.
    # See https://github.com/KevinMusgrave/pytorch-metric-learning/discussions/595

    # # Other metrics from Pytorch Metric Learning
    # dataset_dict = {"samples": test_samples_dataset, "test": test_dataset}
    # test_results = tester.test(dataset_dict, epoch, trunk, embedder, splits_to_eval=[('test', ['samples'])])
    #
    # # Assert for some metrics that they were calculated correctly
    # assert np.allclose(np.sum([cm[i][i] for i in range(cm.shape[0])]) / len(test_dataset), test_results["test"]["r_precision_level0"])

    # Record accuracy
    test_results["test"]["accuracy"] = np.sum([cm[i][i] for i in range(cm.shape[0])]) / len(test_dataset)

    # Other metrics that are based on the embeddings and true labels
    embeddings, labels = tester.get_all_embeddings(test_dataset, trunk, embedder, return_as_numpy=True)
    embed_label_metrics = [silhouette_score]
    for embed_label_metric in embed_label_metrics:
        test_results["test"][embed_label_metric.__name__] = embed_label_metric(embeddings, labels)

    # Visualize via umap embedding
    umap_embeddings = umap.UMAP().fit_transform(embeddings)
    visualizer_hook(None, umap_embeddings, labels, "TEST", "TEST")

    # Other metrics that are based on the true and predicted labels
    true_pred_labels_metrics = [adjusted_mutual_info_score, normalized_mutual_info_score]
    for true_pred_labels_metric in true_pred_labels_metrics:
        test_results["test"][true_pred_labels_metric.__name__] = true_pred_labels_metric(true_labels, pred_labels)

    # Change any float32 anf float64 to float so that they can be dumped
    for key in test_results["test"].keys():
        if test_results["test"][key].__class__ in [np.float64, np.float32]:
            test_results["test"][key] = float(test_results["test"][key])

    with open('test_results.json', 'w') as fp:
        json.dump(test_results, fp)


if __name__ == '__main__':
    main()