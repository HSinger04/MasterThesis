import argparse
import pickle
from tqdm import tqdm
import sys

training_subjects = [
    1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38
]
training_cameras = [2, 3]
test_classes = [1, 7, 13, 19, 25, 31, 37, 43, 49, 55, 61, 67, 73, 79, 85, 91, 97, 103, 109, 115]  # one shot protocol
max_body_true = 2
max_body_kinect = 4
num_joint = 25
max_frame = 300

import numpy as np
import os


def read_skeleton_filter(file):
    with open(file, 'r') as f:
        skeleton_sequence = {}
        skeleton_sequence['numFrame'] = int(f.readline())
        skeleton_sequence['frameInfo'] = []
        # num_body = 0
        for t in range(skeleton_sequence['numFrame']):
            frame_info = {}
            frame_info['numBody'] = int(f.readline())
            frame_info['bodyInfo'] = []

            for m in range(frame_info['numBody']):
                body_info = {}
                body_info_key = [
                    'bodyID', 'clipedEdges', 'handLeftConfidence',
                    'handLeftState', 'handRightConfidence', 'handRightState',
                    'isResticted', 'leanX', 'leanY', 'trackingState'
                ]
                body_info = {
                    k: float(v)
                    for k, v in zip(body_info_key, f.readline().split())
                }
                body_info['numJoint'] = int(f.readline())
                body_info['jointInfo'] = []
                for v in range(body_info['numJoint']):
                    joint_info_key = [
                        'x', 'y', 'z', 'depthX', 'depthY', 'colorX', 'colorY',
                        'orientationW', 'orientationX', 'orientationY',
                        'orientationZ', 'trackingState'
                    ]
                    joint_info = {
                        k: float(v)
                        for k, v in zip(joint_info_key, f.readline().split())
                    }
                    body_info['jointInfo'].append(joint_info)
                frame_info['bodyInfo'].append(body_info)
            skeleton_sequence['frameInfo'].append(frame_info)

    return skeleton_sequence


def get_nonzero_std(s):  # tvc
    index = s.sum(-1).sum(-1) != 0  # select valid frames
    s = s[index]
    if len(s) != 0:
        s = s[:, :, 0].std() + s[:, :, 1].std() + s[:, :, 2].std()  # three channels
    else:
        s = 0
    return s


def read_xyz(file, max_body=4, num_joint=25):  # 取了前两个body
    seq_info = read_skeleton_filter(file)
    data = np.zeros((max_body, seq_info['numFrame'], num_joint, 3))
    for n, f in enumerate(seq_info['frameInfo']):
        for m, b in enumerate(f['bodyInfo']):
            for j, v in enumerate(b['jointInfo']):
                if m < max_body and j < num_joint:
                    data[m, n, j, :] = [v['x'], v['y'], v['z']]
                else:
                    pass

    # select two max energy body
    energy = np.array([get_nonzero_std(x) for x in data])
    index = energy.argsort()[::-1][0:max_body_true]
    data = data[index]

    data = data.transpose(3, 1, 2, 0)
    return data


def gendata(sixty_path, onetwenty_path, out_path, ignored_sample_path=None, reference_sample_path=None,
            benchmark='xview', part='eval'):
    """

    :param sixty_path: Path to data
    :param out_path: Save path
    :param ignored_sample_path:
    :param reference_sample_path: Path to text file containing the exemplars for the one-shot part
    :param benchmark:
    :param part: val means validation for one-shot, sample means exemplar for one-shot and training means training
    :return:
    """
    if ignored_sample_path != None:
        with open(ignored_sample_path, 'r') as f:
            ignored_samples = [
                line.strip() + '.skeleton' for line in f.readlines()
            ]
    else:
        ignored_samples = []

    if reference_sample_path != None:
        with open(reference_sample_path, 'r') as f:
            reference_samples = [
                line.strip() + '.skeleton' for line in f.readlines()
            ]
    else:
        reference_samples = []
    sample_name = []
    sample_label = []
    j = None

    print(reference_samples)
    for data_path in [sixty_path, onetwenty_path]:
        if onetwenty_path == data_path:
            j = len(sample_name)
        for filename in os.listdir(data_path):
            # Only process not-to-ignore and not-OS samples
            if filename in ignored_samples:
                continue
            if part is not "sample":
                if filename in reference_samples:
                    continue

            action_class = int(
                filename[filename.find('A') + 1:filename.find('A') + 4])
            subject_id = int(
                filename[filename.find('P') + 1:filename.find('P') + 4])
            camera_id = int(
                filename[filename.find('C') + 1:filename.find('C') + 4])

            if benchmark == 'xview':
                istraining = (camera_id in training_cameras)
            elif benchmark == 'xsub':
                istraining = (subject_id in training_subjects)
            elif benchmark == 'one_shot':
                # Check if datum belongs to one-shot data or not
                istraining = (action_class not in test_classes)
            else:
                raise ValueError()

            # print(filename)

            if part == 'train':
                issample = istraining
            elif part == 'val':
                issample = not (istraining)
            elif part == "sample":
                issample = filename in reference_samples
            else:
                raise ValueError()

            if issample:
                sample_name.append(filename)
                if part == "sample" or part == "val":
                    sample_label.append(action_class // 6)
                else:
                    sample_label.append(action_class - 1)

    with open('{}/{}_label.pkl'.format(out_path, part), 'wb') as f:
        pickle.dump((sample_name, list(sample_label)), f)

    fp = np.zeros((len(sample_label), 3, max_frame, num_joint, max_body_true), dtype=np.float32)

    data_path = sixty_path
    for i, s in enumerate(tqdm(sample_name)):
        if i == j:
            data_path = onetwenty_path
        data = read_xyz(os.path.join(data_path, s), max_body=max_body_kinect, num_joint=num_joint)
        # TODO: Unclear: Data gets added here
        fp[i, :, 0:data.shape[1], :, :] = data

    del sample_name
    del sample_label

    fp = pre_normalization(fp)
    np.save('{}/{}_data_joint.npy'.format(out_path, part), fp)


sixty_path = "/kaggle/input/ntu120-4/nturgbd_skeletons_s001_to_s017/nturgb+d_skeletons"
onetwenty_path = "/kaggle/input/ntu120-4/nturgbd_skeletons_s018_to_s032"
ignored_sample_path = "/kaggle/input/ntupreseqstatistics/NTU_RGBD120_samples_with_missing_skeletons.txt"
reference_samples_path = "/kaggle/input/graph-metric-learning-gendata/one_shot_samples.txt"
out_folder = "./"

# benchmark = ['xsub', 'xview', 'one_shot']
benchmark = ['one_shot']
# part = ['train', 'val', 'sample']
# part = ['val', 'sample'].
# TODO
part = ['train']

for b in benchmark:
    for p in part:
        out_path = os.path.join(out_folder, b)
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        print(b, p)
        gendata(
            sixty_path,
            onetwenty_path,
            out_path,
            ignored_sample_path,
            reference_samples_path,
            benchmark=b,
            part=p)
