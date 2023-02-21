import torch
import hydra
from pytorch_metric_learning.utils import common_functions as c_f

from train import get_trunk, MLP, get_datasets
from tester.with_autocast_one_shot_tester import WithAMPGlobalEmbeddingSpaceTester

@hydra.main(config_path="config", version_base="1.1", )
def main(cfg):
    # Load model
    # Use hydra
    # Copy the original config of the model that I am testing
    # Use PML's inference models
    # Use https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    # Inter- and intraclass distances (look for libraries maybe? https://scikit-learn.org/stable/modules/classes.html#clustering-metrics
    # Overall accuracy (compute from confusion matrix)
    # Visualize as plot - I think PML supports it. Otherwise, ask Raphael

    print(cfg)

    # Set the datasets
    data_dir = cfg.dataset.data_dir
    print("Data dir: " + data_dir)

    ds_mem_limits = cfg.dataset.dataset_split.mem_limits
    mem_limits = {"val_samples": ds_mem_limits.val_samples, "val": ds_mem_limits.val, "train": ds_mem_limits.train}

    # TODO: Change the get_datasets such that it also works with that stuff. Use a dummy train dataset.
    _, val_dataset, val_samples_dataset = get_datasets(cfg.model.model_name,
                                                                   cfg.dataset.data_path, cfg.dataset.label_path,
                                                                   cfg.dataset.dataset_split.val_classes,
                                                                   mem_limits, debug=cfg.dataset.debug,
                                                                   data_kwargs=cfg.dataset.data_kwargs)

    print("Testset: ", len(val_dataset), "Samplesset: ", len(val_samples_dataset))

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
        cfg.mode.model_folder, "trunk_*.pth")

    c_f.load_dict_of_models(
        models, model_suffix, cfg.mode.model_folder, device, log_if_successful=True
    )

    dataset_dict = {"samples": val_samples_dataset, "val": val_dataset}
    experiment_name = hydra.core.hydra_config.HydraConfig.get().job.config_name

    # Create the tester
    tester = WithAMPGlobalEmbeddingSpaceTester(
        cfg.tester.use_amp,
        cfg.tester.batch_size,
        cfg.tester.dataloader_num_workers,
        end_of_testing_hook=hooks.end_of_testing_hook,
        # size_of_tsne=20
    )

    # tester.test
    # tester.get_all_embeddings

    tester.embedding_filename = data_dir + "/" + experiment_name + ".pkl"