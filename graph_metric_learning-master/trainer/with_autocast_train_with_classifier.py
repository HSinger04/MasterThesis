from pytorch_metric_learning.trainers import TrainWithClassifier
from pytorch_metric_learning.utils import common_functions as c_f
import torch

class WithAutocastTrainWithClassifier(TrainWithClassifier):
    def __init__(self, use_amp, *args):
        self.set_use_amp(use_amp)
        super.__init__(*args)

    def set_use_amp(self, use_amp):
        self.use_amp = use_amp
        self.scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    def calculate_loss(self, curr_batch):
        data, labels = curr_batch
        with torch.autocast(device_type='cuda', self.use_amp):
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
                self.scaler.update(v)

