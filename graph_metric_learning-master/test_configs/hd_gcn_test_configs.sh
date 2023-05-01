#!/bin/bash

cd ..

models=("1" "2" "21")
modalities=("bone" "joint")
train_types=("with_val" "no_val")
best_or_50=("true" "false")

for model in "${models[@]}"; do
  for modality in "${modalities[@]}"; do
    for train_type in "${train_types[@]}"; do
      if [ $train_type = "with_val" ]; then
        num_labels=90
      else
        num_labels=100
      fi
      for best_xor_50 in "${best_or_50[@]}"; do
        python3 test.py --config-name test_hd-gcn mode.use_best=${best_xor_50} dataset=hd-gcn_${modality}_test \
        model=hd-gcn_com_${model} \
        +mode.old_config=/home/work/Downloads/train_hd-gcn_${modality}_com_${model}_${train_type}/.hydra/config.yaml \
        +mode.model_folder=/home/work/Downloads/train_hd-gcn_${modality}_com_${model}_${train_type}/example_saved_models \
        hydra.run.dir=./outputs/test_hd-gcn_${modality}_com_${model}_${train_type} num_train_labels=${num_labels}
      done
    done
  done
done