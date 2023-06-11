from functools import cached_property
from typing import Optional

import numpy as np
import sapien.core as sapien
import transforms3d
from sapien.utils import Viewer

from hand_teleop.env.rl_env.base import BaseRLEnv
from hand_teleop.env.sim_env.constructor import add_default_scene_light
from hand_teleop.env.sim_env.table_door_env import TableDoorEnv
from hand_teleop.utils.common_robot_utils import generate_free_robot_hand_info, generate_arm_robot_hand_info


class TableDoorRLEnv(TableDoorEnv, BaseRLEnv):
    def __init__(self, use_gui=False, frame_skip=5, robot_name="adroit_hand_free", friction=1, **renderer_kwargs):
        super().__init__(use_gui, frame_skip, friction=friction, **renderer_kwargs)
        self.setup(robot_name)
        if "arm" in robot_name:
            init_pose = sapien.Pose(np.array([-0.65, 0, -0.01]), transforms3d.euler.euler2quat(0, 0, 0))
            self.robot.set_pose(init_pose)

        # Parse link name
        if self.is_robot_free:
            info = generate_free_robot_hand_info()[robot_name]
        else:
            info = generate_arm_robot_hand_info()[robot_name]
        self.palm_link_name = info.palm_name
        self.palm_link = [link for link in self.robot.get_links() if link.get_name() == self.palm_link_name][0]
        self.handle_link = [link for link in self.table_door.get_links() if link.get_name() == "handle"][0]
        self.is_contact = False
        self.is_unlock = False

    def get_oracle_state(self):
        robot_qpos_vec = self.robot.get_qpos()
        door_qpos_vec = self.table_door.get_qpos()
        handle_pose = self.handle_link.get_pose()
        palm_pose = self.palm_link.get_pose()
        handle_in_palm = handle_pose.p - palm_pose.p
        palm_v = self.palm_link.get_velocity()
        palm_w = self.palm_link.get_angular_velocity()
        self.is_contact = self.check_contact(self.robot.get_links(), [self.handle_link])
        self.is_unlock = door_qpos_vec[1] > 1.1
        return np.concatenate(
            [robot_qpos_vec, door_qpos_vec, palm_v, palm_w, handle_in_palm,
             [int(self.is_contact), int(self.is_unlock)]])

    def get_robot_state(self):
        robot_qpos_vec = self.robot.get_qpos()
        palm_pose = self.palm_link.get_pose()
        return np.concatenate([robot_qpos_vec, palm_pose.p])

    def get_reward(self, action):
        door_qpos_vec = self.table_door.get_qpos()
        handle_pose = self.handle_link.get_pose()
        palm_pose = self.palm_link.get_pose()
        is_contact = self.is_contact

        reward = -0.1 * min(np.linalg.norm(palm_pose.p - handle_pose.p), 0.5)
        if is_contact:
            reward += 0.1
            openness = door_qpos_vec[1]
            reward += openness * 0.5
            if openness > 1.1:
                reward += 0.5
                reward += door_qpos_vec[0] * 5

        return reward

    def reset(self, *, seed: Optional[int] = None, return_info: bool = False, options: Optional[dict] = None):
        super().reset(seed=seed)
        if not self.is_robot_free:
            qpos = np.zeros(self.robot.dof)
            xarm_qpos = self.robot_info.arm_init_qpos
            qpos[:self.arm_dof] = xarm_qpos
            self.robot.set_qpos(qpos)
        self.reset_internal()
        self.is_contact = False
        return self.get_observation()

    @cached_property
    def obs_dim(self):
        return self.robot.dof + 2 + 6 + 3 + 2

    def is_done(self):
        return self.current_step >= self.horizon

    @cached_property
    def horizon(self):
        return 250


def main_env():
    env = TableDoorRLEnv(use_gui=True, robot_name="svh_hand_free",
                         frame_skip=10, use_visual_obs=False)
    base_env = env
    robot_dof = env.robot.dof
    env.seed(0)
    env.reset()
    viewer = Viewer(base_env.renderer)
    viewer.set_scene(base_env.scene)
    base_env.viewer = viewer

    viewer.toggle_pause(True)
    for i in range(5000):
        action = np.zeros(robot_dof)
        action[2] = 0.01
        obs, reward, done, _ = env.step(action)
        env.render()

    while not viewer.closed:
        env.render()


if __name__ == '__main__':
    main_env()
