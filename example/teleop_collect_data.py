from pathlib import Path

import numpy as np
import sapien.core as sapien
import transforms3d.euler

from hand_detector.hand_monitor import Record3DSingleHandMotionControl
from hand_teleop.env.sim_env.constructor import add_default_scene_light
from hand_teleop.env.sim_env.mug_flip_env import MugFlipEnv
from hand_teleop.env.sim_env.relocate_env import RelocateEnv
from hand_teleop.env.sim_env.table_door_env import TableDoorEnv
from hand_teleop.gui.teleop_gui import GUIBase, DEFAULT_TABLE_TOP_CAMERAS
from hand_teleop.kinematics.mano_robot_hand import MANORobotHand
from hand_teleop.player.recorder import DataRecorder


def main():
    # Choose a task: relocate, open_door, flip
    task_name = ["open_door", "relocate", "flip"][0]

    # Choose object if you are working relocate
    object_name = ['tomato_soup_can', 'bleach_cleanser', 'mug', "mustard_bottle", "potted_meat_can"][3]

    # Setup
    frame_skip = 5
    object_scale = 0.8
    if task_name == "relocate":
        task_full_name = f"relocate-{object_name}"
        env_dict = dict(frame_skip=frame_skip, object_name=object_name, object_scale=object_scale)
    elif task_name == "open_door":
        task_full_name = "table_door"
        env_dict = dict(frame_skip=frame_skip)
    elif task_name == "flip":
        task_full_name = "flip"
        env_dict = dict(frame_skip=frame_skip)
    else:
        raise NotImplementedError

    # Specify the demonstration file path and name
    demo_data_root_path = Path(__file__).parent / "teleop_collected_data"
    demo_data_root_path.mkdir(exist_ok=True)
    demo_index = "0000"  # demo_index, used only in the name of the demonstration file
    path = Path(demo_data_root_path) / task_full_name
    path = path / f"{demo_index}.pickle"

    if task_name == "open_door":
        env = TableDoorEnv(**env_dict, use_gui=True)
    elif task_name == "relocate":
        env = RelocateEnv(**env_dict, use_gui=True)
    elif task_name == "flip":
        env = MugFlipEnv(**env_dict, use_gui=True)
    else:
        raise NotImplementedError
    env.reset_env()
    env.seed(int(demo_index))

    # Setup viewer and camera
    add_default_scene_light(env.scene, env.renderer)
    gui = GUIBase(env.scene, env.renderer)
    for name, params in DEFAULT_TABLE_TOP_CAMERAS.items():
        gui.create_camera(**params)
    gui.viewer.set_camera_rpy(0, -0.7, 0.01)
    gui.viewer.set_camera_xyz(-0.4, 0, 0.45)
    scene = env.scene
    viz_mat_hand_init = gui.context.create_material(np.array([0, 0, 0, 0]), np.array([0.96, 0.75, 0.69, 1]), 0.0, 0.8,
                                                    0)

    # Perception
    motion_control = Record3DSingleHandMotionControl(hand_mode="right_hand", show_hand=True)

    # Recorder
    recorder = DataRecorder(filename=str(path.resolve()), scene=scene)

    # Init
    create_robot = False
    steps = 0
    env_init_pos = np.array([-0.4, 0, 0.2])
    rgb, depth = motion_control.camera.fetch_rgb_and_depth()
    locked_indices = []
    scene.step()

    # Press "q" on the keyboard to exit the teleoperation when you finish
    # The demonstration data will be automatically saved
    while not gui.closed:
        for _ in range(frame_skip):
            scene.step()
        gui.render(additional_views=[rgb[..., ::-1]])
        steps += 1

        if not motion_control.initialized:
            success, motion_data = motion_control.step()
            rgb = motion_data["rgb"]
            if not success:
                continue

            viz_mat_hand_init.set_base_color(motion_control.init_process_color)
            rotate_pose = sapien.Pose(q=[0.9238, 0, 0.3826, 0], p=[0.2, 0, -0.1])
            gui.update_mesh(motion_data["vertices"], motion_data["faces"], viz_mat=viz_mat_hand_init,
                            clear_context=True, pose=sapien.Pose(env_init_pos) * rotate_pose)
        else:
            if not create_robot:
                zero_joint_pos = motion_control.compute_hand_zero_pos()
                mano_robot = MANORobotHand(env.scene, env.renderer, init_joint_pos=zero_joint_pos,
                                           control_interval=frame_skip * scene.get_timestep(), scale=1)
                robot = mano_robot.robot
                robot.set_pose(sapien.Pose(env_init_pos, transforms3d.euler.euler2quat(0, np.pi / 2, 0)))
                create_robot = True

                # Lock means that the finger will not move regardless of the hand pose detection results
                # It can save you sometime when you already grasp something and do not want to release it
                # You can press "z" on the keyboard to lock the hand and then press "x" to unlock it
                def change_locked():
                    locked_indices.clear()
                    contact_finger_indices = mano_robot.check_contact_finger([env.target_object])
                    locked_indices.extend(contact_finger_indices)
                    mano_robot.highlight_finger_color(contact_finger_indices)

                def clear_locked():
                    locked_indices.clear()
                    mano_robot.clear_finger_color()

                gui.register_keydown_action('z', change_locked)
                gui.register_keydown_action('x', clear_locked)

                # Clear colored hand visualization during initialization
                for i in range(len(gui.nodes)):
                    node = gui.nodes.pop()
                    gui.render_scene.remove_node(node)

            success, motion_data = motion_control.step()
            rgb = motion_data["rgb"]

            # Data recording.py
            record_data = motion_data.copy()
            record_data.update({"success": success})

            # Remove the pop code if you want to save the camera image in into the dataset. It can be large.
            record_data.pop("rgb")
            record_data.pop("depth")

            recorder.step(record_data)
            if not success:
                continue

            root_joint_qpos = motion_control.compute_operator_space_root_qpos(motion_data)
            root_joint_qpos *= 1

            finger_joint_qpos = mano_robot.compute_qpos(motion_data["pose_params"][3:])
            robot_qpos = np.concatenate([root_joint_qpos, finger_joint_qpos])

            if np.abs(robot.get_qpos().mean()) < 1e-5:
                robot.set_qpos(robot_qpos)

            mano_robot.control_robot(robot_qpos, confidence=motion_data["confidence"], lock_indices=locked_indices)

            # Create SAPIEN mesh for rendering
            # gui.update_mesh(motion_data["vertices"], motion_data["faces"], viz_mat=viz_mat_hand_init,
            #                 clear_context=True,
            #                 pose=sapien.Pose(root_joint_qpos[:3] + np.array([0, -0.5, 0]) + env_init_pos))

    print(len(recorder.data_list))
    meta_data = dict(env_class=env.__class__.__name__, env_kwargs=env_dict,
                     shape_param=motion_control.calibrated_shape_params,
                     zero_joint_pos=motion_control.compute_hand_zero_pos())
    recorder.dump(meta_data)


if __name__ == '__main__':
    main()
