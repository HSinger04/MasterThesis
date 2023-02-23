# MasterThesis

## Requirements

```
pip install -r MasterThesis/requirements.txt
```

## Data

### Data Download

### Data Preparation

\# TODO: Where data gets saved for each case.

## Run Code

### Hydra Configs

Training and Testing configurations are handled by [hydra](https://github.com/facebookresearch/hydra). Under MasterThesis/graph_metric_learning-master/config, the main config 
```.yaml``` files as well as directories to the config groups can be found.  

### Training

Change directory to graph_metric_learning-master: ```cd MasterThesis/graph_metric_learning-master```.

Then run e.g. ```python3 train.py --config-name train_sttformer +mode.type=*mode* +mode.model_folder=outputs/train_sttformer/example_saved_models``` where 
```*mode*``` is of value ```train_from_scratch```, ```train_from_latest``` or ```fine-tune```, depending on whether you want the training to start from scratch, 
continue training a model where you left off or fine-tune the best model. 

### Testing

Change directory to graph_metric_learning-master: ```cd MasterThesis/graph_metric_learning-master```. Then run e.g. ```python3 test.py --config-name test_sttformer```.

### Notes On Performance/Memory

The files in the ```trainer``` config group specify ```batch_size```, ```dataloader_num_workers``` and ```use_amp```, the last of which specifies whether to use Automatic Mixed Precision or not.
Changing these parameters can change the speed of your training or help with memory issues (especially with GPU VRAM). Note that while enabling Automatic Mixed Precision can drastically speed up the training, it 
can lead to NaNs and thus to unstable learning and even errors. My recommendation is to try it and see if the loss often results in NaNs or not and if not, to use it.

The files in the ```tester``` config group have the same types of parameters, though ```use_amp``` is recommended to be set to ```false``` as of 23.02.2023 as 
even setting it to ```true``` does not seem to enable Automatic Mixed Precision.

The files in the ```dataset``` config group specify a dictionary called ```mem_limits```. For the data splits (i.e. val_samples, val, train) that you do not have enough RAM
to load the whole data split for, set the respective value to a number higher than ```0``` (ideally the highest value you can load at once, though the specific value is only
relevant for the initial data preparation in ```train.py```, so do not worry about optimizing this value) to use [mmap_mode](https://numpy.org/doc/stable/reference/generated/numpy.memmap.html).
However, note that likely mmap_mode results in slower data loading time during training/testing.

### Debugging

Certain aspects such as e.g. data preparation in ```train.py``` can be annoyingly long when one wants to debug the file. In those cases, use one of the main configs for debugging.

### Evaluation Metrics

During training, we use the accuracy on the validation set (composed of the splits ```val_samples``` and ```val```) in a one-shot setting as the main metric. 
The accuracy is equivalent to ```precision_at_1_level0``` metric when using ```splits_to_eval=[('val', ['samples'])]``` (see also MasterThesis/graph_metric_learning-master/test/test_tester/test_with_autocast_one_shot_tester.py).
The history of metrics can be found under MasterThesis/graph_metric_learning-master/outputs/*config-name*/logs/accuracies_*hashed_value*_SAMPLES.csv.

During testing, we evaluate the following on the test set in a one-shot setting: The [confusion matrix](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html), the accuracy, the mean [Silhouette Coefficient](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html#sklearn.metrics.silhouette_score),
all the [metrics measured by default by Pytorch Metric Learning](https://kevinmusgrave.github.io/pytorch-metric-learning/accuracy_calculation/#explanations-of-the-default-accuracy-metrics).
