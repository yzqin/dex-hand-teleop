import numpy as np
import sapien.core as sapien
from transforms3d.euler import euler2quat
from transforms3d.quaternions import quat2axangle
from hand_teleop.kinematics.mano_robot_hand import MANORobotHand


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
        __file__).parent.parent / "hand_detector" / "extra_data" / "smpl" / "SMPLX_NEUTRAL.pkl"
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
        __file__).parent.parent / "hand_detector/extra_data/hand_module/SMPLX_HAND_INFO.pkl"
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
