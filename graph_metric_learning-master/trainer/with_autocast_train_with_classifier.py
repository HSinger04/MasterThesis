from pytorch_metric_learning.trainers import TrainWithClassifier
from pytorch_metric_learning.utils import common_functions as c_f
import torch

def get_train_dataloader(dataset, batch_size, sampler, num_workers, collate_fn):
    """
    Identical to PML's utils.common_functions.get_train_dataloader, except that pin_memory=True
    """
    if isinstance(sampler, torch.utils.data.BatchSampler):
        return torch.utils.data.DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=int(batch_size),
        sampler=sampler,
        drop_last=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        shuffle=sampler is None,
        pin_memory=True,
    )

class WithAutocastTrainWithClassifier(TrainWithClassifier):
    """ PML's TrainWithClassifier with support for Automatic Mixed Precision. """
    def __init__(self, use_amp, *args, **kwargs):
        self.set_use_amp(use_amp)
        super().__init__(*args, **kwargs)

    def initialize_dataloader(self):
        """
        Identical to PML's base_trainers initialize_dataloader, except that pin_memory=True for the data loader
        """
        c_f.LOGGER.info("Initializing dataloader")
        self.dataloader = get_train_dataloader(
            self.dataset,
            self.batch_size,
            self.sampler,
            self.dataloader_num_workers,
            self.collate_fn,
        )
        if not self.iterations_per_epoch:
            self.iterations_per_epoch = len(self.dataloader)
        c_f.LOGGER.info("Initializing dataloader iterator")
        self.dataloader_iter = iter(self.dataloader)
        c_f.LOGGER.info("Done creating dataloader iterator")

    def set_use_amp(self, use_amp):
        self.use_amp = use_amp
        self.scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    def calculate_loss(self, curr_batch):
        data, labels = curr_batch
        with torch.autocast(device_type='cuda', enabled=self.use_amp):
            embeddings = self.compute_embeddings(data)
            logits = self.maybe_get_logits(embeddings)
            indices_tuple = self.maybe_mine_embeddings(embeddings, labels)
            self.losses["metric_loss"] = self.maybe_get_metric_loss(
                embeddings, labels, indices_tuple
            )
            self.losses["classifier_loss"] = self.maybe_get_classifier_loss(logits, labels)

    def backward(self):
        self.scaler.scale(self.losses["total_loss"]).backward()

    def clip_gradients(self):
        if self.gradient_clippers is not None:
            for k, v in self.optimizers.items():
                if c_f.regex_replace("_optimizer$", "", k) not in self.freeze_these:
                    self.scaler.unscale_(v)

            for v in self.gradient_clippers.values():
                v()

    def step_optimizers(self):
        for k, v in self.optimizers.items():
            if c_f.regex_replace("_optimizer$", "", k) not in self.freeze_these:
                self.scaler.step(v)
                self.scaler.update()

