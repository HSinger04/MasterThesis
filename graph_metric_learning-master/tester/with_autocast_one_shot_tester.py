import pickle

import torch
from pytorch_metric_learning.testers.base_tester import BaseTester
from pytorch_metric_learning.testers.global_embedding_space import GlobalEmbeddingSpaceTester
from pytorch_metric_learning.utils import common_functions as c_f
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator

def get_eval_dataloader(dataset, batch_size, num_workers, collate_fn):
    """
    Identical to PML's utils.common_functions.get_eval_dataloader, except that pin_memory=True
    """
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=int(batch_size),
        drop_last=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        shuffle=False,
        pin_memory=True,
    )

class WithAMPGlobalEmbeddingSpaceTester(GlobalEmbeddingSpaceTester):
    def __init__(self, use_amp, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_amp = use_amp

    def get_embeddings_for_eval(self, trunk_model, embedder_model, input_imgs):
        input_imgs = c_f.to_device(
            input_imgs, device=self.data_device, dtype=self.dtype
        )
        with torch.autocast(device_type='cuda', enabled=self.use_amp):
            trunk_output = trunk_model(input_imgs)
            if self.use_trunk_output:
                return trunk_output
            return embedder_model(trunk_output)

    def get_all_embeddings(
        self,
        dataset,
        trunk_model,
        embedder_model=None,
        collate_fn=None,
        eval=True,
        return_as_numpy=False,
    ):
        """
        Identical to PML's base_testers get_all_embeddings, except that pin_memory=True for the data loader
        """
        if embedder_model is None:
            embedder_model = c_f.Identity()
        if eval:
            trunk_model.eval()
            embedder_model.eval()
        dataloader = get_eval_dataloader(
            dataset, self.batch_size, self.dataloader_num_workers, collate_fn
        )
        embeddings, labels = self.compute_all_embeddings(
            dataloader, trunk_model, embedder_model
        )
        embeddings = self.maybe_normalize(embeddings)
        if return_as_numpy:
            return embeddings.cpu().numpy(), labels.cpu().numpy()
        return embeddings, labels
