from tqdm import tqdm
import torch
import numpy as np
from torch.utils.data import DataLoader
import hydra
from pytorch_metric_learning.utils import common_functions as c_f
from pytorch_metric_learning.utils import logging_presets, inference
from pytorch_metric_learning.utils import common_functions as c_f
from sklearn.metrics import silhouette_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from train import get_trunk, MLP, get_datasets
from tester.with_autocast_one_shot_tester import WithAMPGlobalEmbeddingSpaceTester

@hydra.main(config_path="config", version_base="1.1", )
def main(cfg):
    # TODO: Copy the original config of the model that I am testing
    # Overall accuracy (compute from confusion matrix)
    # Visualize as plot - I think PML supports it. Otherwise, ask Raphael

    print(cfg)

    # Set the datasets
    data_dir = cfg.dataset.data_dir
    print("Data dir: " + data_dir)

    ds_mem_limits = cfg.dataset.dataset_split.mem_limits
    mem_limits = {"val_samples": ds_mem_limits.val_samples, "val": ds_mem_limits.val, "train": ds_mem_limits.train}

    # TODO: Change the get_datasets such that it also works with that stuff. Use a dummy train dataset.
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
    # TODO: Replace classifier with identity or sth.

    # Package the above stuff into dictionaries.
    models = {"trunk": trunk, "embedder": embedder, "classifier": classifier}

    _, model_suffix = c_f.latest_version(
        cfg.mode.model_folder, "trunk_*.pth", best=cfg.mode.use_best)

    c_f.load_dict_of_models(
        models, model_suffix, cfg.mode.model_folder, device, log_if_successful=True
    )

    dataset_dict = {"samples": test_samples_dataset, "test": test_dataset}
    experiment_name = hydra.core.hydra_config.HydraConfig.get().job.config_name

    record_keeper, _, _ = logging_presets.get_record_keeper("logs", "tensorboard",
                                                            is_new_experiment=cfg.mode.type=="train_from_scratch")
    hooks = logging_presets.get_hook_container(record_keeper)

    # Confusion matrix
    # im = inference.InferenceModel(trunk, embedder)
    # im.train_knn(test_samples_dataset)
    #
    # test_dataloader = DataLoader(test_dataset, cfg.dataset.data_loader.batch_size,
    #                              num_workers=cfg.dataset.data_loader.num_workers, pin_memory=True)
    # true_labels = np.empty(0)
    # pred_labels = []

    # Temporarily disable logger to avoid spamming
    # c_f.LOGGER.propagate = False
    # for input_batch, label_batch in tqdm(test_dataloader):
    #     _, indices = im.get_nearest_neighbors(input_batch, k=1)
    #     true_labels = np.append(true_labels, label_batch.cpu().numpy())
    #     pred_labels += [test_dataset.__getitem__(x[0])[1] for x in indices.cpu().numpy()]

    # Enable logger again
    c_f.LOGGER.propagate = True

    # # Display confusion matrix
    # cm = confusion_matrix(pred_labels, true_labels, labels=test_samples_dataset.label)
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=test_samples_dataset.label)
    # disp.plot()
    # plt.show()

    # Create the tester
    # TODO: Maybe use the visualizer stuff
    tester = WithAMPGlobalEmbeddingSpaceTester(
        use_amp=cfg.tester.use_amp,
        batch_size=cfg.tester.batch_size,
        dataloader_num_workers=cfg.tester.dataloader_num_workers,
        end_of_testing_hook=hooks.end_of_testing_hook
    )

    # # Silhouette score
    # embeddings, labels = tester.get_all_embeddings(test_dataset, trunk, embedder, return_as_numpy=True)
    # silhouette_score(embeddings, labels)

    tester.embedding_filename = data_dir + "/" + experiment_name + ".pkl"

    # Other metrics
    print(tester.test(dataset_dict, 0, trunk, embedder, splits_to_eval=[('test', ['samples'])]))


if __name__ == '__main__':
    main()