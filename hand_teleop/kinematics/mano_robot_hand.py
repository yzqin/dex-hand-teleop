from functools import cached_property
from typing import List, Dict

import numpy as np
import sapien.core as sapien
from sapien.core import Pose
from transforms3d.euler import euler2quat, axangle2euler
from transforms3d.quaternions import mat2quat, quat2axangle

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


def main():
    from sapien.core.pysapien import renderer as R
    from sapien.utils import Viewer
    from hand_teleop.utils.render_scene_utils import add_mesh_to_renderer, add_line_set_to_renderer
    import smplx
    from pathlib import Path
    import torch

    # Hand type
    hand_type = "left"

    # Create SMPLX model
    smplx_model_path = Path(
        __file__).parent.parent.parent / "hand_detector" / "extra_data" / "smpl" / "SMPLX_NEUTRAL.pkl"
    smplx_model = smplx.create(str(smplx_model_path.resolve()), model_type="smplx", batch_size=1,
                               gender='neutral', num_betas=10, use_pca=False, ext='pkl').cuda()

    # Add shape parameters
    shape_params = np.array([[-0.6953, -0.0172, 0.1187, -0.0556, -0.0531, -0.0848, 0.0136, -0.0182,
                              -0.0680, -0.0583]], dtype=np.float32) + np.random.randn(1, 10) * 0.3
    shape_params = torch.from_numpy(shape_params.astype(np.float32)).cuda()
    body_pose = torch.zeros((1, 63)).float().cuda()
    smplx_hand_to_panoptic = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]

    # Hand type based information
    hand_info_pkl_file = Path(
        __file__).parent.parent.parent / "hand_detector/extra_data/hand_module/SMPLX_HAND_INFO.pkl"
    hand_info = np.load(hand_info_pkl_file, allow_pickle=True)
    hand_vertex_index = np.array(hand_info[f'{hand_type}_hand_verts_idx'])
    if hand_type == 'right':
        wrist_idx, hand_start_idx = 21, 21
        hand_idxs = [21] + list(range(40, 55)) + list(range(71, 76))  # 21 for right wrist. 20 finger joints
        hand_pose = -smplx_model.right_hand_mean
    else:
        wrist_idx, hand_start_idx = 20, 20
        hand_idxs = [20] + list(range(25, 40)) + list(range(66, 71))  # 20 for left wrist. 20 finger joints
        hand_pose = -smplx_model.left_hand_mean

    def forward_smplx(pose_param):
        if hand_type == 'right':
            kwargs = dict(right_hand_pose=pose_param)
        else:
            kwargs = dict(left_hand_pose=pose_param)

        with torch.no_grad():
            output = smplx_model(body_pose=body_pose, betas=shape_params, return_verts=True, **kwargs)
        joints = output.joints
        hand_joints = joints[:, hand_idxs, :][:, smplx_hand_to_panoptic, :]
        joint_pos = hand_joints - joints[:, hand_start_idx:hand_start_idx + 1, :]
        joint_pos = joint_pos.detach().cpu().numpy()[0]
        origin = joint_pos[0:1, :]
        vertices = output.vertices
        vertices = vertices[:, torch.from_numpy(hand_vertex_index).cuda(), :]
        vertices = (vertices - joints[:, hand_start_idx:hand_start_idx + 1, :]).cpu().numpy()[0] - origin
        joint_pos -= origin
        print(f"Finger tip: {joint_pos[np.array([4, 8, 12, 16, 20])]}")
        return vertices, joint_pos

    # Setup
    engine = sapien.Engine()
    renderer = sapien.VulkanRenderer(offscreen_only=False)
    engine.set_renderer(renderer)
    config = sapien.SceneConfig()
    config.gravity = np.array([0, 0, 0])
    scene = engine.create_scene(config=config)
    scene.set_timestep(1 / 125)

    # Lighting
    scene.set_ambient_light(np.array([0.6, 0.6, 0.6]))
    scene.add_directional_light(np.array([1, -1, -1]), np.array([1, 1, 1]))
    scene.add_point_light(np.array([2, 2, 2]), np.array([2, 2, 2]))
    scene.add_point_light(np.array([2, -2, 2]), np.array([2, 2, 2]))
    scene.add_point_light(np.array([-2, 0, 2]), np.array([2, 2, 2]))

    # Viewer
    viewer = Viewer(renderer)
    viewer.set_scene(scene)
    viewer.set_camera_xyz(-0.5, 0.2, 0)
    viewer.set_camera_rpy(np.pi, 0, 0)
    viewer.toggle_axes(0)

    # Renderer
    _, joint_pos = forward_smplx(hand_pose)
    nodes = []
    context: R.Context = renderer._internal_context
    mat_hand = context.create_material(np.zeros(4), np.array([0.96, 0.75, 0.69, 1]), 0.0, 0.8, 0)
    mano_robot = MANORobotHand(scene, renderer, joint_pos, free_root=True, control_interval=0.1, hand_type=hand_type)
    faces = np.load(str(Path(__file__).parent / "smplx_faces.npy"))
    parent = np.array([0, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19])
    mano_robot.robot.set_qpos(np.zeros([mano_robot.robot.dof]))

    # while not viewer.closed:
    #     viewer.render()

    render_scene: R.Scene = scene.get_renderer_scene()._internal_scene
    while not viewer.closed:
        # Clear node
        for i in range(len(nodes)):
            node = nodes.pop()
            render_scene.remove_node(node)

        mean_pose = smplx_model.right_hand_mean if hand_type == "right" else smplx_model.left_hand_mean
        hand_qpos = mano_robot.compute_qpos((hand_pose + mean_pose).cpu().numpy().flatten())
        hand_verts, joint_pos = forward_smplx(hand_pose)
        # Rendering
        obj = add_mesh_to_renderer(scene, renderer, hand_verts, faces, material=mat_hand)
        line_set = add_line_set_to_renderer(scene, renderer, joint_pos, np.stack([parent, np.arange(21)], axis=1),
                                            color=np.array([1, 0, 0, 1]))
        line = add_line_set_to_renderer(scene, renderer, joint_pos, np.stack([parent, np.arange(21)], axis=1),
                                        color=np.array([1, 0, 0, 1]))
        obj.set_position(np.array([0, 0.2, 0]))
        line.set_position(np.array([0, 0.2, 0]))
        obj.set_rotation(euler2quat(0, np.pi / 2, np.pi / 2))
        line_set.set_rotation(euler2quat(0, np.pi / 2, np.pi / 2))
        line.set_rotation(euler2quat(0, np.pi / 2, np.pi / 2))
        nodes.extend([obj, line_set, line])

        relative_pose = (mano_robot.robot.get_links()[9].get_pose().inv() * mano_robot.robot.get_links()[10].get_pose())
        print(f"Relative {relative_pose}")
        print(f"Axis angle {np.multiply(*quat2axangle(relative_pose.q))}")

        qpos = np.concatenate([np.zeros(6), hand_qpos])
        for i in range(5):
            mano_robot.control_robot(qpos)
            scene.step()
            scene.update_render()
            viewer.render()
        hand_pose += torch.randn([45], dtype=torch.float32).cuda() * 0.02


if __name__ == '__main__':
    main()
