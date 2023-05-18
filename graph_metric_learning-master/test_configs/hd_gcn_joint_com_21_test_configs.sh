#!/bin/bash

cd ..

model="21"
modality="joint"
train_type="no_val"
epochs=("best" "50" "75" "100" "150" "175" "200")
num_labels=100

for used_epoch in "${epochs[@]}"; do
  if [ $used_epoch = "best" ]; then
    best_xor_50="true"
  else
    best_xor_50="false"
  fi

  outdir="./outputs/test_outputs/test_hd-gcn_${modality}_com_${model}_${train_type}_${used_epoch}_epoch"
  echo $outdir

  # Skip directories that already exist
  if [ ! -d ${outdir} ]; then
    python3 test.py --config-name test_hd-gcn mode.use_best=${best_xor_50} dataset=hd-gcn_${modality}_test \
    model=hd-gcn_com_${model} \
    +mode.old_config=/home/work/Downloads/train_hd-gcn_${modality}_com_${model}_${train_type}/.hydra/config.yaml \
    +mode.model_folder=/home/work/Downloads/train_hd-gcn_${modality}_com_${model}_${train_type}/example_saved_models \
    +mode.load_epoch=${used_epoch} \
    hydra.run.dir=${outdir} num_train_labels=${num_labels}
  fi
done
