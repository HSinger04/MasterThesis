python3 train.py --config-name train_sttformer_bone_with_val +mode.type=train_from_scratch +trainer.save_epochs=[]
python3 train.py --config-name train_sttformer_joint_motion_with_val +mode.type=train_from_scratch +trainer.save_epochs=[]

python3 train.py --config-name train_sttformer_joint_no_val +mode.type=train_from_scratch +trainer.save_epochs=[33]

# TODO
python3 train.py --config-name train_hd_gcn_joint_com_1_no_val +mode.type=train_from_scratch +trainer.save_epochs=[?]
python3 train.py --config-name train_hd_gcn_joint_com_2_no_val +mode.type=train_from_scratch +trainer.save_epochs=[?]
python3 train.py --config-name train_hd_gcn_joint_com_21_no_val +mode.type=train_from_scratch +trainer.save_epochs=[?]

python3 train.py --config-name train_hd_gcn_bone_com_2_no_val +mode.type=train_from_scratch +trainer.save_epochs=[37]
python3 train.py --config-name train_hd_gcn_bone_com_21_no_val +mode.type=train_from_scratch +trainer.save_epochs=[24]