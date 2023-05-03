#!/bin/bash

cd ..

modalities=("bone" "joint" "joint_vel" "bone_vel")
train_types=("with_val" "no_val")
best_or_50=("true" "false")

for modality in "${modalities[@]}"; do
  for train_type in "${train_types[@]}"; do

    if [ $train_type = "with_val" ]; then
      num_labels=90
    else
      num_labels=100
    fi

    for best_xor_50 in "${best_or_50[@]}"; do
      if [ $best_xor_50 = "true" ]; then
        used_epoch="best"
      else
        used_epoch="50"
      fi

      outdir="./outputs/test_outputs/test_hyperformer_${modality}_${train_type}_${used_epoch}_epoch"
      echo $outdir

      # Skip directories that already exist
      if [ ! -d ${outdir} ]; then
        python3 test.py --config-name test_hyperformer mode.use_best=${best_xor_50} dataset=hyperformer_${modality}_test \
        +mode.old_config=/home/work/Downloads/train_hyperformer_${modality}_${train_type}/.hydra/config.yaml \
        +mode.model_folder=/home/work/Downloads/train_hyperformer_${modality}_${train_type}/example_saved_models \
        hydra.run.dir=${outdir} num_train_labels=${num_labels}
      fi

    done
  done
done