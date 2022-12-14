# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import os
import os.path as osp
import numpy as np
import cv2
import pickle
import logging
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', help='./')
parser.add_argument('--make_video', default=False, type=bool)

args = parser.parse_args()
args_dict = vars(parser.parse_args())

make_video = args_dict['make_video']
root_path = args_dict['root_path']
raw_data_file = osp.join(root_path, 'raw_data', 'raw_skes_data.pkl')
save_path = osp.join(root_path, 'denoised_data')

if not osp.exists(save_path):
    os.mkdir(save_path)

rgb_ske_path = osp.join(save_path, 'rgb+ske')
if not osp.exists(rgb_ske_path):
    os.mkdir(rgb_ske_path)

actors_info_dir = osp.join(save_path, 'actors_info')
if not osp.exists(actors_info_dir):
    os.mkdir(actors_info_dir)

video_path = osp.join(save_path, 'denoised_vids')
if not osp.exists(video_path):
    os.mkdir(video_path)

missing_count = 0
noise_len_thres = 11
noise_spr_thres1 = 0.8
noise_spr_thres2 = 0.69754
noise_mot_thres_lo = 0.089925
noise_mot_thres_hi = 2

noise_len_logger = logging.getLogger('noise_length')
noise_len_logger.setLevel(logging.INFO)
noise_len_logger.addHandler(logging.FileHandler(osp.join(save_path, 'noise_length.log')))
noise_len_logger.info('{:^20}\t{:^17}\t{:^8}\t{}'.format('Skeleton', 'bodyID', 'Motion', 'Length'))

noise_spr_logger = logging.getLogger('noise_spread')
noise_spr_logger.setLevel(logging.INFO)
noise_spr_logger.addHandler(logging.FileHandler(osp.join(save_path, 'noise_spread.log')))
noise_spr_logger.info('{:^20}\t{:^17}\t{:^8}\t{:^8}'.format('Skeleton', 'bodyID', 'Motion', 'Rate'))

noise_mot_logger = logging.getLogger('noise_motion')
noise_mot_logger.setLevel(logging.INFO)
noise_mot_logger.addHandler(logging.FileHandler(osp.join(save_path, 'noise_motion.log')))
noise_mot_logger.info('{:^20}\t{:^17}\t{:^8}'.format('Skeleton', 'bodyID', 'Motion'))

fail_logger_1 = logging.getLogger('noise_outliers_1')
fail_logger_1.setLevel(logging.INFO)
fail_logger_1.addHandler(logging.FileHandler(osp.join(save_path, 'denoised_failed_1.log')))

fail_logger_2 = logging.getLogger('noise_outliers_2')
fail_logger_2.setLevel(logging.INFO)
fail_logger_2.addHandler(logging.FileHandler(osp.join(save_path, 'denoised_failed_2.log')))

missing_skes_logger = logging.getLogger('missing_frames')
missing_skes_logger.setLevel(logging.INFO)
missing_skes_logger.addHandler(logging.FileHandler(osp.join(save_path, 'missing_skes.log')))
missing_skes_logger.info('{:^20}\t{}\t{}'.format('Skeleton', 'num_frames', 'num_missing'))

missing_skes_logger1 = logging.getLogger('missing_frames_1')
missing_skes_logger1.setLevel(logging.INFO)
missing_skes_logger1.addHandler(logging.FileHandler(osp.join(save_path, 'missing_skes_1.log')))
missing_skes_logger1.info('{:^20}\t{}\t{}\t{}\t{}\t{}'.format('Skeleton', 'num_frames', 'Actor1',
                                                              'Actor2', 'Start', 'End'))

missing_skes_logger2 = logging.getLogger('missing_frames_2')
missing_skes_logger2.setLevel(logging.INFO)
missing_skes_logger2.addHandler(logging.FileHandler(osp.join(save_path, 'missing_skes_2.log')))
missing_skes_logger2.info('{:^20}\t{}\t{}\t{}'.format('Skeleton', 'num_frames', 'Actor1', 'Actor2'))


def denoising_by_length(ske_name, bodies_data):
    """
    Denoising data based on the frame length for each bodyID.
    Filter out the bodyID which length is less or equal than the predefined threshold.
    Probable motivation: If a body only showed up for a very short amount of frames, it's likely that the
    body was just through a misdetection and not of an actual person

    """
    noise_info = str()
    new_bodies_data = bodies_data.copy()
    for (bodyID, body_data) in new_bodies_data.items():
        length = len(body_data['interval'])
        if length <= noise_len_thres:
            noise_info += 'Filter out: %s, %d (length).\n' % (bodyID, length)
            noise_len_logger.info('{}\t{}\t{:.6f}\t{:^6d}'.format(ske_name, bodyID,
                                                                  body_data['motion'], length))
            del bodies_data[bodyID]
    if noise_info != '':
        noise_info += '\n'

    return bodies_data, noise_info


def get_valid_frames_by_spread(points):
    """
    Find the valid (or reasonable) frames (index) based on the spread of X and Y. Pre-proc from NTU paper.

    points: joints or colors
    """
    num_frames = points.shape[0]
    valid_frames = []
    for i in range(num_frames):
        # x and y are 25 dimensional
        x = points[i, :, 0]
        y = points[i, :, 1]
        if (x.max() - x.min()) <= noise_spr_thres1 * (y.max() - y.min()):  # 0.8
            valid_frames.append(i)
    return valid_frames


def denoising_by_spread(ske_name, bodies_data):
    """
    Denoising data based on the spread of Y value and X value.
    Filter out the bodyID which the ratio of noisy frames is higher than the predefined
    threshold.

    bodies_data: contains at least 2 bodyIDs
    """
    noise_info = str()
    denoised_by_spr = False  # mark if this sequence has been processed by spread.

    new_bodies_data = bodies_data.copy()
    # for (bodyID, body_data) in bodies_data.items():
    for (bodyID, body_data) in new_bodies_data.items():
        if len(bodies_data) == 1:
            break
        # e.g. (1950, 3) --> (78, 25, 3)
        valid_frames = get_valid_frames_by_spread(body_data['joints'].reshape(-1, 25, 3))
        num_frames = len(body_data['interval'])
        num_noise = num_frames - len(valid_frames)
        if num_noise == 0:
            continue

        # Only get here if there are invalid frames
        ratio = num_noise / float(num_frames)
        motion = body_data['motion']
        # If too many noisy frames, delete the body entirely
        if ratio >= noise_spr_thres2:  # 0.69754
            del bodies_data[bodyID]
            denoised_by_spr = True
            noise_info += 'Filter out: %s (spread rate >= %.2f).\n' % (bodyID, noise_spr_thres2)
            noise_spr_logger.info('%s\t%s\t%.6f\t%.6f' % (ske_name, bodyID, motion, ratio))
        else:  # Update motion
            joints = body_data['joints'].reshape(-1, 25, 3)[valid_frames]
            # Recalculate motion based on the valid frames. Take the motion according to the minimum between
            # old and new motion data
            body_data['motion'] = min(motion, np.sum(np.var(joints.reshape(-1, 3), axis=0)))
            noise_info += '%s: motion %.6f -> %.6f\n' % (bodyID, motion, body_data['motion'])
            # TODO: Consider removing noisy frames for each bodyID

    if noise_info != '':
        noise_info += '\n'

    return bodies_data, noise_info, denoised_by_spr


def denoising_by_motion(ske_name, bodies_data, bodies_motion):
    """
    Filter out the bodyID which motion is out of the range of predefined interval

    """
    # Sort bodies based on the motion, return a list of tuples
    # bodies_motion = sorted(bodies_motion.items(), key=lambda x, y: cmp(x[1], y[1]), reverse=True)
    bodies_motion = sorted(bodies_motion.items(), key=lambda x: x[1], reverse=True)

    # Reserve the body data with the largest motion
    denoised_bodies_data = [(bodies_motion[0][0], bodies_data[bodies_motion[0][0]])]
    noise_info = str()

    for (bodyID, motion) in bodies_motion[1:]:
        if (motion < noise_mot_thres_lo) or (motion > noise_mot_thres_hi):
            noise_info += 'Filter out: %s, %.6f (motion).\n' % (bodyID, motion)
            noise_mot_logger.info('{}\t{}\t{:.6f}'.format(ske_name, bodyID, motion))
        else:
            denoised_bodies_data.append((bodyID, bodies_data[bodyID]))
    if noise_info != '':
        noise_info += '\n'

    return denoised_bodies_data, noise_info


def denoising_bodies_data(bodies_data):
    """
    Denoising data based on some heuristic methods, not necessarily correct for all samples.

    Return:
      denoised_bodies_data (list): tuple: (bodyID, body_data).
    """
    ske_name = bodies_data['name']
    bodies_data = bodies_data['data']

    # Step 1: Denoising based on frame length.
    bodies_data, noise_info_len = denoising_by_length(ske_name, bodies_data)

    if len(bodies_data) == 1:  # only has one bodyID left after step 1
        return bodies_data.items(), noise_info_len

    # Step 2: Denoising based on spread.
    bodies_data, noise_info_spr, denoised_by_spr = denoising_by_spread(ske_name, bodies_data)

    if len(bodies_data) == 1: # only has one bodyID left after step 2
        return bodies_data.items(), noise_info_len + noise_info_spr

    bodies_motion = dict()  # get body motion
    for (bodyID, body_data) in bodies_data.items():
        bodies_motion[bodyID] = body_data['motion']
    # Sort bodies based on the motion
    # bodies_motion = sorted(bodies_motion.items(), key=lambda x, y: cmp(x[1], y[1]), reverse=True)
    bodies_motion = sorted(bodies_motion.items(), key=lambda x: x[1], reverse=True)
    denoised_bodies_data = list()
    for (bodyID, _) in bodies_motion:
        denoised_bodies_data.append((bodyID, bodies_data[bodyID]))

    return denoised_bodies_data, noise_info_len + noise_info_spr

    # TODO: Consider denoising further by integrating motion method

    # if denoised_by_spr:  # this sequence has been denoised by spread
    #     bodies_motion = sorted(bodies_motion.items(), lambda x, y: cmp(x[1], y[1]), reverse=True)
    #     denoised_bodies_data = list()
    #     for (bodyID, _) in bodies_motion:
    #         denoised_bodies_data.append((bodyID, bodies_data[bodyID]))
    #     return denoised_bodies_data, noise_info

    # Step 3: Denoising based on motion
    # bodies_data, noise_info = denoising_by_motion(ske_name, bodies_data, bodies_motion)

    # return bodies_data, noise_info


def get_one_actor_points(body_data, num_frames):
    """
    Get joints and colors for only one actor.
    For joints, each frame contains 75 X-Y-Z coordinates.
    For colors, each frame contains 25 x 2 (X, Y) coordinates.
    """
    joints = np.zeros((num_frames, 75), dtype=np.float32)
    colors = np.ones((num_frames, 1, 25, 2), dtype=np.float32) * np.nan
    # start = first valid frame, end = last valid frame
    start, end = body_data['interval'][0], body_data['interval'][-1]
    # [0:3] is x, y, z (not sure about order of coordinates) of the first joint
    joints[start:end + 1] = body_data['joints'].reshape(-1, 75)
    colors[start:end + 1, 0] = body_data['colors']

    return joints, colors


def remove_missing_frames(ske_name, joints, colors):
    """
    Cut off missing frames which all joints positions OF ALL ACTORS are 0s

    For the sequence with 2 actors' data, also record the number of missing frames for
    actor1 and actor2, respectively (for debug).
    """
    num_frames = joints.shape[0]
    num_bodies = colors.shape[1]  # 1 or 2

    if num_bodies == 2:  # DEBUG
        missing_indices_1 = np.where(joints[:, :75].sum(axis=1) == 0)[0]
        missing_indices_2 = np.where(joints[:, 75:].sum(axis=1) == 0)[0]
        cnt1 = len(missing_indices_1)
        cnt2 = len(missing_indices_2)

        start = 1 if 0 in missing_indices_1 else 0
        end = 1 if num_frames - 1 in missing_indices_1 else 0
        # Note down info about missing frames
        if max(cnt1, cnt2) > 0:
            if cnt1 > cnt2:
                info = '{}\t{:^10d}\t{:^6d}\t{:^6d}\t{:^5d}\t{:^3d}'.format(ske_name, num_frames,
                                                                            cnt1, cnt2, start, end)
                missing_skes_logger1.info(info)
            else:
                info = '{}\t{:^10d}\t{:^6d}\t{:^6d}'.format(ske_name, num_frames, cnt1, cnt2)
                missing_skes_logger2.info(info)

    # Find valid frame indices that the data is not missing or lost
    # For two-subjects action, this means both data of actor1 and actor2 is missing.
    valid_indices = np.where(joints.sum(axis=1) != 0)[0]  # 0-based index
    # Compare missing_indices to missing_indices_1 and 2: Same formula, except that for the former, we consider
    # joints of both actors while for the latter 2, we look at the joints of each actor individually
    missing_indices = np.where(joints.sum(axis=1) == 0)[0]
    num_missing = len(missing_indices)

    if num_missing > 0:  # Update joints and colors
        joints = joints[valid_indices]
        colors[missing_indices] = np.nan
        global missing_count
        missing_count += 1
        missing_skes_logger.info('{}\t{:^10d}\t{:^11d}'.format(ske_name, num_frames, num_missing))

    return joints, colors


def get_bodies_info(bodies_data):
    """

    :param bodies_data: A dictionary containing ske_name, the joint, color, interval and motion info of bodies and
    the number of frames
    :return: A string describing the bodyIDs, Intervals and Motions of the actors
    """
    bodies_info = '{:^17}\t{}\t{:^8}\n'.format('bodyID', 'Interval', 'Motion')
    for (bodyID, body_data) in bodies_data.items():
        start, end = body_data['interval'][0], body_data['interval'][-1]
        bodies_info += '{}\t{:^8}\t{:f}\n'.format(bodyID, str([start, end]), body_data['motion'])

    return bodies_info + '\n'


def get_two_actors_points(bodies_data):
    """
    Get the first and second actor's joints positions and colors locations.

    # Arguments:
        bodies_data (dict): 3 key-value pairs: 'name', 'data', 'num_frames'.
        bodies_data['data'] is also a dict, while the key is bodyID, the value is
        the corresponding body_data which is also a dict with 4 keys:
          - joints: raw 3D joints positions. Shape: (num_frames x 25, 3)
          - colors: raw 2D color locations. Shape: (num_frames, 25, 2)
          - interval: a list which records the frame indices.
          - motion: motion amount

    # Return:
        joints, colors.
    """
    ske_name = bodies_data['name']
    label = int(ske_name[-2:])
    num_frames = bodies_data['num_frames']
    bodies_info = get_bodies_info(bodies_data['data'])

    bodies_data, noise_info = denoising_bodies_data(bodies_data)  # Denoising data
    bodies_info += noise_info

    bodies_data = list(bodies_data)
    if len(bodies_data) == 1:  # Only left one actor after denoising
        if label >= 50:  # DEBUG: Denoising failed for two-subjects action
            fail_logger_2.info(ske_name)

        # Treat this as if it's only one actor / the 2nd actor was just an artifact due to noise
        bodyID, body_data = bodies_data[0]
        joints, colors = get_one_actor_points(body_data, num_frames)
        bodies_info += 'Main actor: %s' % bodyID
    else:
        if label < 50:  # DEBUG: Denoising failed for one-subject action
            fail_logger_1.info(ske_name)

        joints = np.zeros((num_frames, 150), dtype=np.float32)
        colors = np.ones((num_frames, 2, 25, 2), dtype=np.float32) * np.nan

        bodyID, actor1 = bodies_data[0]  # the 1st actor with largest motion
        start1, end1 = actor1['interval'][0], actor1['interval'][-1]
        # Fill joint and colors of first actor
        joints[start1:end1 + 1, :75] = actor1['joints'].reshape(-1, 75)
        colors[start1:end1 + 1, 0] = actor1['colors']
        actor1_info = '{:^17}\t{}\t{:^8}\n'.format('Actor1', 'Interval', 'Motion') + \
                      '{}\t{:^8}\t{:f}\n'.format(bodyID, str([start1, end1]), actor1['motion'])
        del bodies_data[0]

        actor2_info = '{:^17}\t{}\t{:^8}\n'.format('Actor2', 'Interval', 'Motion')
        start2, end2 = [0, 0]  # initial interval for actor2 (virtual)

        # TODO: Understand. Would have to watch a video with 3 bodies
        while len(bodies_data) > 0:
            bodyID, actor = bodies_data[0]
            start, end = actor['interval'][0], actor['interval'][-1]
            # TODO: I am not understanding how this formula relates to showing there is no overlap. However, if there is
            # overlap, one might want to say that it's action of the other actor as there can be at most two
            # and the other one can't appear twice in the same frames.
            # Another question I have: Why can we say that it necessarily belongs to the first actor? What if it's a case
            # that it does not overlap with neither first nor second actor?
            if min(end1, end) - max(start1, start) <= 0:  # no overlap with actor1
                # Add joint and colors of first actor
                joints[start:end + 1, :75] = actor['joints'].reshape(-1, 75)
                colors[start:end + 1, 0] = actor['colors']
                actor1_info += '{}\t{:^8}\t{:f}\n'.format(bodyID, str([start, end]), actor['motion'])
                # Update the interval of actor1
                # TODO: What's the purpose of this? My assumption: Actors who appear stay for the whole duration
                # of the video from start to finish --> since we assume that the frames given by [start, end] are part
                # of actor 1, we expand their screentime.
                start1 = min(start, start1)
                end1 = max(end, end1)
            elif min(end2, end) - max(start2, start) <= 0:  # no overlap with actor2
                joints[start:end + 1, 75:] = actor['joints'].reshape(-1, 75)
                colors[start:end + 1, 1] = actor['colors']
                actor2_info += '{}\t{:^8}\t{:f}\n'.format(bodyID, str([start, end]), actor['motion'])
                # Update the interval of actor2
                start2 = min(start, start2)
                end2 = max(end, end2)
            del bodies_data[0]

        bodies_info += ('\n' + actor1_info + '\n' + actor2_info)

    with open(osp.join(actors_info_dir, ske_name + '.txt'), 'w') as fw:
        fw.write(bodies_info + '\n')

    return joints, colors


def save_video(ske_name, colors):
    """

    :param ske_name:
    :param colors: (num_frames, num_bodies, num_joints=25, color_coord=2)
    :return:
    """
    pass
    # size = 1920, 1080
    # fps = 31
    # out = cv2.VideoWriter(osp.join(video_path, ske_name + '.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), fps, (size[1], size[0]), False)
    # for i, frame in enumerate(colors):
    #     data = np.zeros(size, dtype='uint8')
    #     data.fill(255)
    #     coords = frame.reshape(-1, 2).astype(int)
    #     coords_2 = coords[:, 0] * size[1] + coords[:, 1]
    #     try:
    #         np.put(data, coords_2, 0)
    #     except IndexError:
    #         continue
    #     out.write(data)
    # out.release()

def get_raw_denoised_data():
    """
    Get denoised data (joints positions and color locations) from raw skeleton sequences.

    For each frame of a skeleton sequence, an actor's 3D positions of 25 joints represented
    by an 2D array (shape: 25 x 3) is reshaped into a 75-dim vector by concatenating each
    3-dim (x, y, z) coordinates along the row dimension in joint order. Each frame contains
    two actor's joints positions constituting a 150-dim vector. If there is only one actor,
    then the last 75 values are filled with zeros. Otherwise, select the main actor and the
    second actor based on the motion amount. Each 150-dim vector as a row vector is put into
    a 2D numpy array where the number of rows equals the number of valid frames. All such
    2D arrays are put into a list and finally the list is serialized into a cPickle file.

    For the skeleton sequence which contains two or more actors (mostly corresponds to the
    last 11 classes), the filename and actors' information are recorded into log files.
    For better understanding, also generate RGB+skeleton videos for visualization.
    """

    with open(raw_data_file, 'rb') as fr:  # load raw skeletons data
        raw_skes_data = pickle.load(fr)

    num_skes = len(raw_skes_data)
    print('Found %d available skeleton sequences.' % num_skes)

    raw_denoised_joints = []
    raw_denoised_colors = []
    frames_cnt = []

    for (idx, bodies_data) in enumerate(raw_skes_data):
        ske_name = bodies_data['name']
        print('Processing %s' % ske_name)
        num_bodies = len(bodies_data['data'])

        if num_bodies == 1:  # only 1 actor
            num_frames = bodies_data['num_frames']
            # bodies_data['data'] is the bodies data (i.e. a mapping from bodies to their joints, colors and interval)
            body_data = list(bodies_data['data'].values())[0]
            joints, colors = get_one_actor_points(body_data, num_frames)
        else:  # more than 1 actor, select two main actors
            joints, colors = get_two_actors_points(bodies_data)
            # Remove missing frames
            # TODO: Why is remove_missing_frames only applied to num_bodies == 1?
            joints, colors = remove_missing_frames(ske_name, joints, colors)
            num_frames = joints.shape[0]  # Update
            # Visualize selected actors' skeletons on RGB videos.
            if make_video:
                save_video(ske_name, colors)

        raw_denoised_joints.append(joints)
        raw_denoised_colors.append(colors)
        frames_cnt.append(num_frames)

        if (idx + 1) % 1000 == 0:
            print('Processed: %.2f%% (%d / %d), ' % \
                  (100.0 * (idx + 1) / num_skes, idx + 1, num_skes) + \
                  'Missing count: %d' % missing_count)

    raw_skes_joints_pkl = osp.join(save_path, 'raw_denoised_joints.pkl')
    with open(raw_skes_joints_pkl, 'wb') as f:
        pickle.dump(raw_denoised_joints, f, pickle.HIGHEST_PROTOCOL)

    raw_skes_colors_pkl = osp.join(save_path, 'raw_denoised_colors.pkl')
    with open(raw_skes_colors_pkl, 'wb') as f:
        pickle.dump(raw_denoised_colors, f, pickle.HIGHEST_PROTOCOL)

    frames_cnt = np.array(frames_cnt, dtype=np.int)
    np.savetxt(osp.join(save_path, 'frames_cnt.txt'), frames_cnt, fmt='%d')

    print('Saved raw denoised positions of {} frames into {}'.format(np.sum(frames_cnt),
                                                                     raw_skes_joints_pkl))
    print('Found %d files that have missing data' % missing_count)

if __name__ == '__main__':
    get_raw_denoised_data()
