from abc import abstractmethod
from typing import List

import numpy as np
import sapien.core as sapien

from hand_teleop.kinematics.optimizer import PositionOptimizer


class RetargetingBase:
    @abstractmethod
    def retarget(self, target_pos, fixed_qpos):
        pass


class PositionRetargeting(RetargetingBase):
    def __init__(self, robot: sapien.Articulation, target_joint_names: List[str], target_link_names: List[str],
                 has_joint_limits=True, has_global_pose_limits=True):
        self.optimizer = PositionOptimizer(robot, target_joint_names, target_link_names, huber_delta=0.02)
        self.has_global_pose_limits = has_global_pose_limits

        # Joint limit
        self.has_joint_limits = has_joint_limits
        joint_limits = np.ones_like(robot.get_qlimits())
        joint_limits[:, 0] = -1e4  # a large value is equivalent to no limit
        joint_limits[:, 1] = 1e4
        if has_joint_limits:
            joint_limits[6:] = robot.get_qlimits()[6:]
        if has_global_pose_limits:
            joint_limits[:6] = robot.get_qlimits()[:6]
        if has_joint_limits or has_global_pose_limits:
            self.optimizer.set_joint_limit(joint_limits[self.optimizer.target_joint_indices])
        self.joint_limits = joint_limits

        # Temporal information
        self.last_qpos = joint_limits.mean(1)[self.optimizer.target_joint_indices]

    # @timeit
    def retarget(self, target_pos, fixed_qpos):
        qpos = self.optimizer.retarget(target_pos=target_pos.astype(np.float32),
                                       fixed_qpos=fixed_qpos.astype(np.float32),
                                       last_qpos=self.last_qpos.astype(np.float32), verbose=False)
        self.last_qpos = qpos
        robot_qpos = np.zeros(self.optimizer.robot.dof)
        robot_qpos[self.optimizer.fixed_joint_indices] = fixed_qpos
        robot_qpos[self.optimizer.target_joint_indices] = qpos
        return robot_qpos
