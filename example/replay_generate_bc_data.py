from pathlib import Path

import numpy as np
import sapien.core as sapien

from hand_teleop.env.rl_env.mug_flip_env import MugFlipRLEnv
from hand_teleop.env.rl_env.relocate_env import RelocateRLEnv
from hand_teleop.env.rl_env.table_door_env import TableDoorRLEnv
from hand_teleop.kinematics.retargeting_optimizer import PositionRetargeting
from hand_teleop.player.player import RelocateObjectEnvPlayer, TableDoorEnvPlayer, FlipMugEnvPlayer


def bake_demonstration_test(data_path, robot_name, visualize=True):
    assert robot_name in ["allegro_hand_free", "svh_hand_free", "adroit_hand_free"]

    # Recorder
    path = Path(data_path)
    all_data = np.load(str(path.resolve()), allow_pickle=True)
    meta_data = all_data["meta_data"]
    data = all_data["data"]

    env_class = meta_data["env_class"].lower()
    if "relocate" in env_class:
        env = RelocateRLEnv(**meta_data["env_kwargs"], robot_name=robot_name, use_gui=True)
        player = RelocateObjectEnvPlayer(meta_data, data, env, zero_joint_pos=meta_data["zero_joint_pos"])
    elif "door" in env_class:
        env = TableDoorRLEnv(**meta_data["env_kwargs"], robot_name=robot_name, use_gui=True)
        player = TableDoorEnvPlayer(meta_data, data, env, zero_joint_pos=meta_data["zero_joint_pos"])
    elif "flip" in env_class:
        env = MugFlipRLEnv(**meta_data["env_kwargs"], robot_name=robot_name, use_gui=True)
        player = FlipMugEnvPlayer(meta_data, data, env, zero_joint_pos=meta_data["zero_joint_pos"])
    else:
        raise ValueError(env_class)

    # Retargeting

    if robot_name == "adroit_hand_free":
        link_names = ["palm", "thtip", "fftip", "mftip", "rftip", "lftip"] + ["thmiddle", "ffmiddle", "mfmiddle",
                                                                              "rfmiddle", "lfmiddle"]
        joint_names = [joint.get_name() for joint in env.robot.get_active_joints()]
        retargeting = PositionRetargeting(env.robot, joint_names, link_names, has_global_pose_limits=False,
                                          has_joint_limits=True)
        indices = None
    elif "allegro_hand" in robot_name:
        link_names = ["palm", "link_15.0_tip", "link_3.0_tip", "link_7.0_tip", "link_11.0_tip", "link_14.0",
                      "link_2.0", "link_6.0", "link_10.0"]
        joint_names = [joint.get_name() for joint in env.robot.get_active_joints()]
        retargeting = PositionRetargeting(env.robot, joint_names, link_names, has_global_pose_limits=False,
                                          has_joint_limits=True)
        indices = [0, 1, 2, 3, 5, 6, 7, 8]
    elif robot_name == "svh_hand_free":
        link_names = ["right_hand_e1", "right_hand_c", "right_hand_t", "right_hand_s", "right_hand_r",
                      "right_hand_q"]
        link_names += ["right_hand_b", "right_hand_p", "right_hand_o", "right_hand_n", "right_hand_i"]
        joint_names = [joint.get_name() for joint in env.robot.get_active_joints()]
        retargeting = PositionRetargeting(env.robot, joint_names, link_names, has_global_pose_limits=False,
                                          has_joint_limits=True)
        indices = None
    else:
        raise NotImplementedError

    # Save the baked data for your imitation learning purpose
    baked_data = player.bake_demonstration(retargeting, method="tip_middle", indices=indices)
    print(baked_data.keys())

    # Visualize the baked data, only for debug purpose
    if visualize:
        player.scene.remove_articulation(player.human_robot_hand.robot)
        if "door" not in env_class:
            actor_id = env.manipulated_object.get_id()
        else:
            art_id = env.table_door.get_links()[0].get_id()
        if "relocate" in env_class:
            target_pose_vec = baked_data["target_pose"]
            env.target_object.set_pose(sapien.Pose(target_pose_vec[:3], target_pose_vec[3:]))
        for obs, qpos, state, action in zip(baked_data["obs"], baked_data["robot_qpos"], baked_data["state"],
                                            baked_data["action"]):
            env.robot.set_qpos(obs[:env.robot.dof])
            if "door" not in env_class:
                object_pose = state["actor"][actor_id]["pose"]
                env.manipulated_object.set_pose(sapien.Pose(object_pose[:3], object_pose[3:7]))
            else:
                qpos = state["articulation"][art_id]["qpos"]
                env.table_door.set_qpos(qpos)
                door_pose = state["articulation"][art_id]["pose"]
                env.table_door.set_pose(sapien.Pose(door_pose[:3], door_pose[3:7]))

            for _ in range(2):
                env.render()


if __name__ == '__main__':
    robot_name = ["allegro_hand_free", "svh_hand_free", "adroit_hand_free"][2]
    pkl_path = Path(__file__).parent / "example_teleop_data/relocate_tomato_soup_can.pkl"
    # pkl_path = Path(__file__).parent / "example_teleop_data/flip_mug.pkl"
    # pkl_path = Path(__file__).parent / "example_teleop_data/open_door.pkl"
    bake_demonstration_test(data_path=str(pkl_path), robot_name=robot_name, visualize=True)
