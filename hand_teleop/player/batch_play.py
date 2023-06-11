import multiprocessing as mp
import pickle
from pathlib import Path
from time import time

import numpy as np
from natsort import natsorted

from hand_teleop.env.rl_env.relocate_env import RelocateRLEnv
from hand_teleop.kinematics.retargeting_optimizer import PositionRetargeting
from hand_teleop.player.player import RelocateObjectEnvPlayer


def batch_play_data(task_name, robot_name, dir_path, num_cpu, version_num):
    data_dir = Path(dir_path)
    all_path_list = []
    for file in data_dir.iterdir():
        if file.suffix != ".pickle":
            continue
        all_path_list.append(str(file.resolve()))

    all_path_list = natsorted(all_path_list)
    print(all_path_list)
    if num_cpu == 1:
        results = [play_one_data(task_name, robot_name, all_path_list[i]) for i in range(len(all_path_list))]
    else:
        pool = mp.Pool(processes=num_cpu, maxtasksperchild=1)
        results = [pool.apply(play_one_data, args=(task_name, robot_name, all_path_list[i])) for i in
                   range(len(all_path_list))]

    output_file = data_dir / f"{task_name}-{robot_name}-v{version_num}.pkl"
    with output_file.open("wb") as f:
        pickle.dump(results, f)


def play_one_data(task_name, robot_name, filename):
    tic = time()
    all_data = np.load(filename, allow_pickle=True)
    meta_data = all_data["meta_data"]
    data = all_data["data"]
    env = RelocateRLEnv(**meta_data["env_kwargs"], robot_name=robot_name)
    env.reset()
    player = RelocateObjectEnvPlayer(meta_data, data, env, zero_joint_pos=meta_data["zero_joint_pos"])

    # Retargeting
    if robot_name == "adroit_free":
        link_names = ["palm", "thtip", "fftip", "mftip", "rftip", "lftip"] + ["thmiddle", "ffmiddle", "mfmiddle",
                                                                              "rfmiddle", "lfmiddle"]
        joint_names = [joint.get_name() for joint in env.robot.get_active_joints()]
        retargeting = PositionRetargeting(env.robot, joint_names, link_names, has_global_pose_limits=False,
                                          has_joint_limits=True)
        method = "tip_middle"
        indices = None
    elif "allegro_hand" in robot_name:
        link_names = ["palm", "link_15.0_tip", "link_3.0_tip", "link_7.0_tip", "link_11.0_tip", "link_14.0",
                      "link_2.0", "link_6.0", "link_10.0"]
        joint_names = [joint.get_name() for joint in env.robot.get_active_joints()]
        retargeting = PositionRetargeting(env.robot, joint_names, link_names, has_global_pose_limits=False,
                                          has_joint_limits=True)
        method = "tip_middle"
        indices = [0, 1, 2, 3, 5, 6, 7, 8]
    elif robot_name == "svh_free":
        link_names = ["right_hand_e1", "right_hand_c", "right_hand_t", "right_hand_s", "right_hand_r",
                      "right_hand_q"]
        link_names += ["right_hand_b", "right_hand_p", "right_hand_o", "right_hand_n", "right_hand_i"]
        joint_names = [joint.get_name() for joint in env.robot.get_active_joints()]
        retargeting = PositionRetargeting(env.robot, joint_names, link_names, has_global_pose_limits=False,
                                          has_joint_limits=True)
        method = "tip_middle"
        indices = None
    elif robot_name == "ar10_free":
        link_names = ["palm", "thumbtip", "fingertip4", "fingertip3", "fingertip2", "fingertip1"]
        joint_names = [joint.get_name() for joint in env.robot.get_active_joints()]
        retargeting = PositionRetargeting(env.robot, joint_names, link_names, has_global_pose_limits=False,
                                          has_joint_limits=True)
        method = "tip"
        indices = None
    else:
        raise NotImplementedError
    baked_data = player.bake_demonstration(retargeting, method=method, indices=indices)

    print(f"Finish {filename} in {time() - tic}s")
    return dict(observations=baked_data["obs"], actions=baked_data["action"])


def batch_replay_test():
    task_name = "relocate"
    robot_name = "adroit_free"
    dir_path = "./data/teleop/relocate-potted_meat_can"
    version_num = 4
    batch_play_data(task_name, robot_name, dir_path=dir_path, version_num=version_num, num_cpu=1)


if __name__ == '__main__':
    batch_replay_test()
