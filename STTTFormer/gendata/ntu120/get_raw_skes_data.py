# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import os.path as osp
import os
import numpy as np
import pickle
import logging
import argparse


def get_raw_bodies_data(skes_path, ske_name, frames_drop_skes, frames_drop_logger):
    """
    Get raw bodies data from a skeleton sequence.

    Each body's data is a dict that contains the following keys:
      - joints: raw 3D joints positions. Shape: (num_frames x 25, 3)
      - colors: raw 2D color locations. Shape: (num_frames, 25, 2)
      - interval: a list which stores the frame indices of this body.
      - motion: motion amount (only for the sequence with 2 or more bodyIDs).

    Return:
      a dict for a skeleton sequence with 3 key-value pairs:
        - name: the skeleton filename.
        - data: a dict which stores raw data of each body.
        - num_frames: the number of valid frames.
    """
    if int(ske_name[1:4]) >= 18:
        skes_path = '../nturgbd_raw/nturgb+d_skeletons120/'
    ske_file = osp.join(skes_path, ske_name + '.skeleton')
    assert osp.exists(ske_file), 'Error: Skeleton file %s not found' % ske_file
    # Read all data from .skeleton file into a list (in string format)
    print('Reading data from %s' % ske_file[-29:])
    with open(ske_file, 'r') as fr:
        str_data = fr.readlines()

    # The very first value of the .skeleton file. E.g. "103"
    num_frames = int(str_data[0].strip('\r\n'))
    frames_drop = []
    bodies_data = dict()
    # E.g. 0 for the first dude,
    valid_frames = -1  # 0-based index
    current_line = 1

    for f in range(num_frames):
        # Starts with the 2nd line in .skeleton file e.g. "1". The next one would be line 30's "1"
        num_bodies = int(str_data[current_line].strip('\r\n'))
        current_line += 1

        if num_bodies == 0:  # no data in this frame, drop it
            frames_drop.append(f)  # 0-based index
            continue

        valid_frames += 1
        joints = np.zeros((num_bodies, 25, 3), dtype=np.float32)
        colors = np.zeros((num_bodies, 25, 2), dtype=np.float32)

        for b in range(num_bodies):
            # Read bodyID from one line after the previous. E.g. '72057594037931101'
            bodyID = str_data[current_line].strip('\r\n').split()[0]
            current_line += 1
            # Read from one line after the previous. E.g. '25'
            num_joints = int(str_data[current_line].strip('\r\n'))  # 25 joints
            current_line += 1

            for j in range(num_joints):
                temp_str = str_data[current_line].strip('\r\n').split()
                # The first three values from next line. E.g. 0.2181153 0.1725972 3.785547
                joints[b, j, :] = np.array(temp_str[:3], dtype=np.float32)
                # TODO: What is "colors"? Also see variable "colors" above
                # The 6th and 7th values from same line. E.g. 1036.233 519.1677
                colors[b, j, :] = np.array(temp_str[5:7], dtype=np.float32)
                current_line += 1

            if bodyID not in bodies_data:  # Add a new body's data
                body_data = dict()
                body_data['joints'] = joints[b]  # ndarray: (25, 3)
                body_data['colors'] = colors[b, np.newaxis]  # ndarray: (1, 25, 2)
                body_data['interval'] = [valid_frames]  # the index of the first frame
            else:  # Update an already existed body's data
                body_data = bodies_data[bodyID]
                # Stack each body's data of each frame along the frame order
                # body_data['joints'] is of dimension # (25 * how many times the body ID has occured, 3 dimensions of joint locations)
                body_data['joints'] = np.vstack((body_data['joints'], joints[b]))
                body_data['colors'] = np.vstack((body_data['colors'], colors[b, np.newaxis]))
                pre_frame_idx = body_data['interval'][-1]
                body_data['interval'].append(pre_frame_idx + 1)  # add a new frame index

            bodies_data[bodyID] = body_data  # Update bodies_data

    num_frames_drop = len(frames_drop)
    assert num_frames_drop < num_frames, \
        'Error: All frames data (%d) of %s is missing or lost' % (num_frames, ske_name)
    if num_frames_drop > 0:
        frames_drop_skes[ske_name] = np.array(frames_drop, dtype=np.int)
        frames_drop_logger.info('{}: {} frames missed: {}\n'.format(ske_name, num_frames_drop,
                                                                    frames_drop))

    # Calculate motion (only for the sequence with 2 or more bodyIDs)
    if len(bodies_data) > 1:
        for body_data in bodies_data.values():
            # First, calculate the variance (from e.g. (1950, 3), we get output of dim (3,)) and then the sum of it to get
            # a scalar value. Seems to be used to determine primary and secondary actor (see doc string of get_raw_denoised_data)
            body_data['motion'] = np.sum(np.var(body_data['joints'], axis=0))

    return {'name': ske_name, 'data': bodies_data, 'num_frames': num_frames - num_frames_drop}


def get_raw_skes_data():
    # # save_path = './data'
    # # skes_path = '/data/pengfei/NTU/nturgb+d_skeletons/'
    # stat_path = osp.join(save_path, 'statistics')
    #
    # skes_name_file = osp.join(stat_path, 'skes_available_name.txt')
    # save_data_pkl = osp.join(save_path, 'raw_skes_data.pkl')
    # frames_drop_pkl = osp.join(save_path, 'frames_drop_skes.pkl')
    #
    # frames_drop_logger = logging.getLogger('frames_drop')
    # frames_drop_logger.setLevel(logging.INFO)
    # frames_drop_logger.addHandler(logging.FileHandler(osp.join(save_path, 'frames_drop.log')))
    # frames_drop_skes = dict()

    skes_name = np.loadtxt(skes_name_file, dtype=str)

    num_files = skes_name.size
    print('Found %d available skeleton files.' % num_files)

    raw_skes_data = []
    frames_cnt = np.zeros(num_files, dtype=np.int)

    for (idx, ske_name) in enumerate(skes_name):
        bodies_data = get_raw_bodies_data(skes_path, ske_name, frames_drop_skes, frames_drop_logger)
        raw_skes_data.append(bodies_data)
        frames_cnt[idx] = bodies_data['num_frames']
        if (idx + 1) % 1000 == 0:
            print('Processed: %.2f%% (%d / %d)' % \
                  (100.0 * (idx + 1) / num_files, idx + 1, num_files))

    with open(save_data_pkl, 'wb') as fw:
        pickle.dump(raw_skes_data, fw, pickle.HIGHEST_PROTOCOL)
    np.savetxt(osp.join(save_path, 'raw_data', 'frames_cnt.txt'), frames_cnt, fmt='%d')

    print('Saved raw bodies data into %s' % save_data_pkl)
    print('Total frames: %d' % np.sum(frames_cnt))

    with open(frames_drop_pkl, 'wb') as fw:
        pickle.dump(frames_drop_skes, fw, pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--skes_path', help='../nturgbd_raw/nturgb+d_skeletons/')
    parser.add_argument('--save_path', help="./")

    args = parser.parse_args()
    args_dict = vars(parser.parse_args())

    # As the name implies, describes where the data will get saved.
    save_path = args_dict['save_path']

    skes_path = args_dict['skes_path']
    # Basically there to identify where the skes_available_name.txt is. Can be thus hard-coded.
    stat_path = osp.join(save_path, 'statistics')

    if not osp.exists(osp.join(save_path, 'raw_data')):
        os.makedirs(osp.join(save_path, 'raw_data'))

    skes_name_file = osp.join(stat_path, 'skes_available_name.txt')
    save_data_pkl = osp.join(save_path, 'raw_data', 'raw_skes_data.pkl')
    frames_drop_pkl = osp.join(save_path, 'raw_data', 'frames_drop_skes.pkl')

    frames_drop_logger = logging.getLogger('frames_drop')
    frames_drop_logger.setLevel(logging.INFO)
    frames_drop_logger.addHandler(logging.FileHandler(osp.join(save_path, 'raw_data', 'frames_drop.log')))
    # Keeps track of what frames to drop. No actual dropping happens here
    frames_drop_skes = dict()

    get_raw_skes_data()

    with open(frames_drop_pkl, 'wb') as fw:
        pickle.dump(frames_drop_skes, fw, pickle.HIGHEST_PROTOCOL)
        
