import pickle

import torch
from pytorch_metric_learning.testers.base_tester import BaseTester
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

class WithAMPOneShotTester(BaseTester):

    def __init__(self, use_amp, batch_size, dataloader_num_workers, end_of_testing_hook=None):
        super().__init__(batch_size=batch_size, dataloader_num_workers=dataloader_num_workers)
        self.use_amp = use_amp
        self.max_accuracy = 0.0
        self.embedding_filename = ""
        self.end_of_testing_hook = end_of_testing_hook

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

    # def __get_correct(self, output, target, topk=(1,)):
    #     with torch.no_grad():
    #         maxk = max(topk)
    #         batch_size = target.size(0)
    #
    #         _, pred = output.topk(maxk, 1, True, True)
    #         pred = pred.t()
    #         correct = pred.eq(target.view(1, -1).expand_as(pred))
    # #             print(correct)
    #     return correct
    #
    #
    # def __accuracy(self, output, target, topk=(1,)):
    #     """Computes the accuracy over the k top predictions for the specified values of k"""
    #     with torch.no_grad():
    #         correct = self.__get_correct(output, target, topk)
    #         batch_size = target.size(0)
    #         res = []
    #         for k in topk:
    #             correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
    #             res.append(correct_k.mul_(100.0 / batch_size))
    #         return res


    # def do_knn_and_accuracies(self, accuracies, embeddings_and_labels, split_name, tag_suffix=''):
    #     #print(embeddings_and_labels)
    #     query_embeddings = embeddings_and_labels["val"][0]
    #     query_labels = embeddings_and_labels["val"][1]
    #     reference_embeddings = embeddings_and_labels["samples"][0]
    #     reference_labels = embeddings_and_labels["samples"][1]
    #     knn_indices, knn_distances = utils.stat_utils.get_knn(reference_embeddings, query_embeddings, 1, False)
    #     knn_labels = reference_labels[knn_indices][:,0]
    #
    #     accuracy = accuracy_score(knn_labels, query_labels)
    #     print(accuracy)
    #     with open(self.embedding_filename+"_last", 'wb') as f:
    #         print("Dumping embeddings for new max_acc to file", self.embedding_filename+"_last")
    #         pickle.dump([query_embeddings, query_labels, reference_embeddings, reference_labels, accuracy], f)
    #     accuracies["accuracy"] = accuracy
    #     keyname = self.accuracies_keyname("mean_average_precision_at_r") # accuracy as keyname not working
    #     accuracies[keyname] = accuracy
