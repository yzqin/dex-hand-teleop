from functools import cached_property
from typing import List, Dict

import numpy as np
import sapien.core as sapien
from sapien.core import Pose
from transforms3d.euler import euler2quat, axangle2euler
from transforms3d.quaternions import mat2quat

from hand_teleop.utils.common_robot_utils import LPFilter, PIDController
from hand_teleop.utils.model_utils import build_free_root, rot_from_connected_link, build_ball_joint, fix_link_inertia, \
    create_visual_material

PALM_THICKNESS = 0.011
FINGER_RADIUS = 0.01
INVERSE_ALONG_AXIS = 0
INVERSE_FACE_AXIS = 1
LITTLE_TO_THUMB_AXIS = 2


class MANORobotHand:
    def __init__(self, scene: sapien.Scene, renderer: sapien.VulkanRenderer, init_joint_pos: np.ndarray,
                 control_interval: float = 0.01, hand_type="right", free_root=True, position_control=True, scale=1):
        # Visual material
        # self.finger_viz_mat_kwargs = dict(specular=0.07, metallic=0.2, roughness=0.8, base_color=(0.9, 0.7, 0.5, 1))
        self.finger_viz_mat_kwargs = dict(specular=0.5, metallic=0.0, roughness=0.1, base_color=(0.1, 0.1, 0.1, 1))
        self.tip_viz_mat_kwargs = dict(specular=0.07, metallic=0.2, roughness=0.5, base_color=(0.9, 0.9, 0.9, 1))

        # Create robot
        self.hand_type = hand_type
        self.free_root = free_root
        self.scale = scale
        self.part_names = ["palm", "thumb", "index", "middle", "ring", "little"]
        self.robot = create_mano_full_dof_robot_hand(scene, renderer, init_joint_pos, free_root=free_root,
                                                     finger_viz_mat_kwargs=self.finger_viz_mat_kwargs,
                                                     tip_viz_mat_kwargs=self.tip_viz_mat_kwargs,
                                                     hand_type=self.hand_type, scale=scale)
        self.robot.set_qpos(np.zeros(self.robot.dof))  # Set qpos to update the articulation architecture
        self.robot.set_name("mano_robot")

        self.link_index_dict = {i: [] for i in range(6)}
        for index, link in enumerate(self.robot.get_links()):
            part_name = link.get_name().split("_")[0]
            if part_name not in self.part_names:
                continue
            self.link_index_dict[self.part_names.index(part_name)].append(index)

        # robot Dynamics
        joints = self.robot.get_active_joints()
        self.position_control = position_control
        if position_control:
            root_translation_control_params = np.array([100, 4, 5]) * 10
            root_rotation_control_params = np.array([20, 1, 1]) * 10
            finger_control_params = np.array([20, 1, 0.2]) * 10
            self.pid = PIDController(1, 0, 0.00, control_interval, [-0.2, 0.2])
        else:
            root_translation_control_params = [0, 200, 50]
            root_rotation_control_params = [0, 100, 25]
            finger_control_params = [0, 10, 2]
            self.pid = PIDController(1, 0, 0.00, control_interval, [-0.2, 0.2])

        if free_root:
            for i in range(3):
                joints[i].set_drive_property(*root_translation_control_params)
            for i in range(3, 6):
                joints[i].set_drive_property(*root_rotation_control_params)
        for i in range(int(free_root) * 6, self.robot.dof):
            joints[i].set_drive_property(*finger_control_params)

        # Controllers and control cache
        self.filter = LPFilter(50, 5)
        self.static_finger_indices = []

        # Cache
        self.scene = scene
        self.time_step = scene.get_timestep()

    @staticmethod
    def compute_qpos(pose_param: np.ndarray):
        if pose_param.size != 45:
            raise ValueError(f"pose_param should be in shape of 45")
        smplx_to_panoptic = np.array([12, 13, 14, 0, 1, 2, 3, 4, 5, 9, 10, 11, 6, 7, 8])
        pose_param = pose_param.reshape([15, 3])[smplx_to_panoptic, :]
        qpos = []
        for i in range(15):
            vec = pose_param[i]
            angle = np.linalg.norm(vec)
            if np.isclose(angle, 0):
                qpos.append(np.zeros(3))
            else:
                axis = vec / (angle + 1e-6)
                euler = axangle2euler(axis, angle, "rxyz")
                qpos.append(euler)
        return np.concatenate(qpos)

    def control_robot(self, target_qpos, confidence=1, lock_indices: List[int] = ()):
        target_qpos = self.filter.next(target_qpos)
        self.robot.set_qf(self.robot.compute_passive_force(external=False))
        current_qpos = self.robot.get_qpos()
        delta_qpos = (target_qpos - current_qpos) * confidence
        if self.position_control:
            pid_delta_qpos = self.pid.control(delta_qpos)
            drive_target = pid_delta_qpos + current_qpos
            if lock_indices:
                joint_indices = self.finger_indices2joint_indices(lock_indices)
                drive_target[joint_indices] = self.robot.get_drive_target()[joint_indices]
            self.robot.set_drive_target(drive_target)
        else:
            pid_qvel = self.pid.control(delta_qpos)
            control_qvel = self.filter.next(pid_qvel)
            self.robot.set_drive_velocity_target(control_qvel)

    @staticmethod
    def finger_indices2joint_indices(finger_indices):
        if len(finger_indices) == 0:
            return np.array([])
        indices = []
        for index in finger_indices:
            indices.append(np.arange(index * 9 - 3, index * 9 + 6))
        return np.concatenate(indices)

    def check_contact_finger(self, actors: List[sapien.Actor]) -> List[int]:
        link_set = set(self.robot.get_links())
        actor_set = set(actors)
        contact_finger_indices = set()
        for contact in self.scene.get_contacts():
            contact_actors = {contact.actor0, contact.actor1}
            if len(link_set & contact_actors) > 0 and len(actor_set & contact_actors) > 0:
                impulse = [point.impulse for point in contact.points]
                if np.sum(np.abs(impulse)) < 1e-6:
                    continue
                if contact.actor0 in link_set:
                    finger_name = contact.actor0.get_name().split("_")[0]
                else:
                    finger_name = contact.actor1.get_name().split("_")[0]
                finger_index = self.part_names.index(finger_name)
                contact_finger_indices.add(finger_index)

        return list(contact_finger_indices)

    def highlight_finger_color(self, finger_indices):
        links = self.robot.get_links()
        for index in np.arange(1, 6):
            link_indices = self.link_index_dict[index]
            for link_index in link_indices:
                link = links[link_index]
                for geometry in link.get_visual_bodies():
                    for shape in geometry.get_render_shapes():
                        mat = shape.material
                        if index in finger_indices:
                            mat.set_base_color([0, 0, 0.8, 1])
                        elif "2_2" in link.get_name():
                            mat.set_base_color(self.tip_viz_mat_kwargs["base_color"])
                        else:
                            mat.set_base_color(self.finger_viz_mat_kwargs["base_color"])

    def clear_finger_color(self):
        for link in self.robot.get_links():
            for geometry in link.get_visual_bodies():
                for shape in geometry.get_render_shapes():
                    mat = shape.material
                    if "2_2" in link.get_name():
                        mat.set_base_color(self.tip_viz_mat_kwargs["base_color"])
                    else:
                        mat.set_base_color(self.finger_viz_mat_kwargs["base_color"])
        return

    @cached_property
    def finger_tips(self):
        # TODO: deal with non full dof hand
        finger_names = ["thumb", "index", "middle", "ring", "little"]
        tip_names = [f"{name}_tip_link" for name in finger_names]
        finger_tips = {link.get_name(): link for link in self.robot.get_links() if link.get_name() in tip_names}
        return [finger_tips[name] for name in tip_names]

    @cached_property
    def palm(self):
        return [link for link in self.robot.get_links() if link.get_name() == "palm"][0]

    @cached_property
    def finger_middles(self):
        finger_names = ["thumb", "index", "middle", "ring", "little"]
        middle_names = [f"{name}_1_2_link" for name in finger_names]
        finger_middles = {link.get_name(): link for link in self.robot.get_links() if link.get_name() in middle_names}
        return [finger_middles[name] for name in finger_middles]


def create_mano_full_dof_robot_hand(scene: sapien.Scene, renderer: sapien.VulkanRenderer, joint_pos: np.ndarray,
                                    finger_viz_mat_kwargs: Dict, tip_viz_mat_kwargs: Dict, hand_type="right",
                                    free_root=True, scale=1):
    # Compute shape related params
    joint_pos = joint_pos - joint_pos[:1, :]
    joint_pos = joint_pos * scale
    finger_palm_width = np.abs(joint_pos[17, LITTLE_TO_THUMB_AXIS] - joint_pos[5, LITTLE_TO_THUMB_AXIS]) / 3

    # Build palm and four palm finger geom
    mat = scene.create_physical_material(1, 0.5, 0.01)
    friction_dict = {"material": mat, "patch_radius": 0.04, "min_patch_radius": 0.02}
    robot_builder = scene.create_articulation_builder()
    palm = _create_palm(robot_builder, free_root, robot_name="human_hand")

    palm_half_width = finger_palm_width / 2 * scale
    palm_thickness = PALM_THICKNESS * scale
    palm_viz_mat = create_visual_material(renderer, **finger_viz_mat_kwargs)
    use_visual = False if renderer is None else True

    # Hand type indicator
    if hand_type == "right":
        along_axis_sign = 1
    else:
        along_axis_sign = -1
    for i in range(4):
        finger_palm_length = np.abs(joint_pos[5 + 4 * i, INVERSE_ALONG_AXIS])
        pos = np.array(
            [-finger_palm_length * along_axis_sign / 2, 0, joint_pos[5, LITTLE_TO_THUMB_AXIS] - i * finger_palm_width])
        if use_visual:
            palm.add_box_visual(pose=Pose(pos),
                                half_size=np.array([finger_palm_length / 2, palm_thickness, palm_half_width]),
                                material=palm_viz_mat)
        palm.add_box_collision(pose=Pose(pos),
                               half_size=np.array([finger_palm_length / 2, palm_thickness, finger_palm_width / 2]),
                               **friction_dict)

    # Build four finger
    finger_names = ["thumb", "index", "middle", "ring", "little"]
    radius = FINGER_RADIUS * scale
    for i in range(5):
        finger_name = finger_names[i]
        finger_viz_mat = create_visual_material(renderer, **finger_viz_mat_kwargs)
        tip_viz_mat = create_visual_material(renderer, **tip_viz_mat_kwargs)
        # Link 0
        _, _, link0 = build_ball_joint(robot_builder, palm, name=f"{finger_name}_0", pose=Pose(joint_pos[1 + 4 * i]))
        pos_link1_in_link0 = joint_pos[2 + 4 * i] - joint_pos[1 + 4 * i]
        length_link1_in_link0 = np.linalg.norm(pos_link1_in_link0)
        quat_link1_in_link0 = mat2quat(rot_from_connected_link(joint_pos[1 + 4 * i], joint_pos[2 + 4 * i]))
        if use_visual:
            link0.add_capsule_visual(pose=Pose(pos_link1_in_link0 / 2, quat_link1_in_link0), radius=radius,
                                     half_length=length_link1_in_link0 / 2, material=finger_viz_mat)
        link0.add_capsule_collision(pose=Pose(pos_link1_in_link0 / 2, quat_link1_in_link0), radius=radius,
                                    half_length=length_link1_in_link0 / 2, **friction_dict)

        # Link 1
        _, _, link1 = build_ball_joint(robot_builder, link0, name=f"{finger_name}_1", pose=Pose(pos_link1_in_link0))
        pos_link2_in_link1 = joint_pos[3 + 4 * i] - joint_pos[2 + 4 * i]
        length_link2_in_link1 = np.linalg.norm(pos_link2_in_link1)
        quat_link2_in_link1 = mat2quat(rot_from_connected_link(joint_pos[2 + 4 * i], joint_pos[3 + 4 * i]))
        if use_visual:
            link1.add_capsule_visual(pose=Pose(pos_link2_in_link1 / 2, quat_link2_in_link1), radius=radius,
                                     half_length=length_link2_in_link1 / 2, material=finger_viz_mat)
        link1.add_capsule_collision(pose=Pose(pos_link2_in_link1 / 2, quat_link2_in_link1), radius=radius,
                                    half_length=length_link2_in_link1 / 2, **friction_dict)

        # Link 2
        _, _, link2 = build_ball_joint(robot_builder, link1, name=f"{finger_name}_2", pose=Pose(pos_link2_in_link1))
        finger_tip_middle = joint_pos[4 + 4 * i] + np.array([0, -1, 0]) * FINGER_RADIUS / 2
        pos_link3_in_link2 = finger_tip_middle - joint_pos[3 + 4 * i]
        length_link3_in_link2 = np.linalg.norm(pos_link3_in_link2)
        quat_link3_in_link2 = mat2quat(rot_from_connected_link(joint_pos[3 + 4 * i], finger_tip_middle))
        if use_visual:
            link2.add_capsule_visual(pose=Pose(pos_link3_in_link2 / 2, quat_link3_in_link2), radius=radius,
                                     half_length=length_link3_in_link2 / 2, material=tip_viz_mat)
        link2.add_capsule_collision(pose=Pose(pos_link3_in_link2 / 2, quat_link3_in_link2), radius=radius,
                                    half_length=length_link3_in_link2 / 2, **friction_dict)

        # Link 3
        link3 = robot_builder.create_link_builder(link2)
        link3.set_name(f"{finger_name}_tip_link")
        link3.set_joint_name(f"{finger_name}_tip_joint")
        link3.set_joint_properties("fixed", limits=np.array([]),
                                   pose_in_parent=Pose(joint_pos[4 + 4 * i] - joint_pos[3 + 4 * i]))

    for link_builder in robot_builder.get_link_builders():
        link_builder.set_collision_groups(0, 1, 2, 2)
    fix_link_inertia(robot_builder)
    robot = robot_builder.build(fix_root_link=True)
    return robot


def _create_palm(robot_builder: sapien.ArticulationBuilder, free_root: bool, robot_name=""):
    if free_root:
        palm = build_free_root(robot_builder, robot_name)
    else:
        root = robot_builder.create_link_builder()
        palm = robot_builder.create_link_builder(root)
        palm.set_joint_properties("fixed", limits=np.array([]),
                                  pose_in_parent=Pose(q=euler2quat(0, np.pi / 2, np.pi / 2)))
    palm.set_name("palm")
    return palm
