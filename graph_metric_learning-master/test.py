import logging

from tqdm import tqdm
import json
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader
import hydra
from pytorch_metric_learning.utils import common_functions as c_f
from pytorch_metric_learning.utils import logging_presets, inference
from pytorch_metric_learning.utils import common_functions as c_f
from sklearn.metrics import silhouette_score, confusion_matrix, ConfusionMatrixDisplay
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

    for i in range(num_classes):
        idx = labels == label_set[i]
        plt.scatter(umap_embeddings[idx, 0], umap_embeddings[idx, 1], marker=".", s=1, label=label_set[i] + 1)

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

    epoch, model_suffix = c_f.latest_version(
        cfg.mode.model_folder, "trunk_*.pth", best=cfg.mode.use_best)

    c_f.load_dict_of_models(
        models, model_suffix, cfg.mode.model_folder, device, log_if_successful=True
    )

    dataset_dict = {"samples": test_samples_dataset, "test": test_dataset}

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
        visualizer_hook=visualizer_hook
    )

    # Silhouette score
    embeddings, labels = tester.get_all_embeddings(test_dataset, trunk, embedder, return_as_numpy=True)
    silh_score = silhouette_score(embeddings, labels)

    # TODO: Visualization differs from when computing umap on the embeddings and labels of tester.get_all_embeddings
    # Other metrics
    test_results = tester.test(dataset_dict, epoch, trunk, embedder, splits_to_eval=[('test', ['samples'])])

    # Assert that accuracy was calculated correctly
    assert np.allclose(np.sum([cm[i][i] for i in range(cm.shape[0])]) / len(test_dataset), test_results["test"]["r_precision_level0"])

    # Record other metrics
    silh_score = float(silh_score)
    test_results["test"]["silhouette_score"] = silh_score
    with open('test_results.json', 'w') as fp:
        json.dump(test_results, fp)


if __name__ == '__main__':
    main()