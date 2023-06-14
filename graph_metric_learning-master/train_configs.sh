python3 train.py --config-name train_hd_gcn_joint_com_21_no_val +mode.type=train_from_scratch trainer.num_epochs=200 +trainer.save_epochs=[75,100,150,200] +mode.model_folder=example_saved_models

python3 train.py --config-name train_hyperformer_bone_vel_with_val
python3 train.py --config-name train_hyperformer_bone_with_val
python3 train.py --config-name train_hyperformer_joint_vel_with_val
python3 train.py --config-name train_hyperformer_joint_with_val
#python3 train.py --config-name train_sttformer_joint_no_val +mode.type=train_from_scratch +trainer.save_epochs=[49]
#python3 train.py --config-name train_sttformer_bone_no_val +mode.type=train_from_scratch +trainer.save_epochs=[23]
#python3 train.py --config-name train_sttformer_joint_motion_no_val +mode.type=train_from_scratch +trainer.save_epochs=[45]