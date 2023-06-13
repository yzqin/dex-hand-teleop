from pathlib import Path

import numpy as np

from hand_teleop.env.sim_env.constructor import add_default_scene_light
from hand_teleop.env.sim_env.mug_flip_env import MugFlipEnv
from hand_teleop.env.sim_env.relocate_env import RelocateEnv
from hand_teleop.env.sim_env.table_door_env import TableDoorEnv
from hand_teleop.gui.teleop_gui import GUIBase, DEFAULT_TABLE_TOP_CAMERAS
from hand_teleop.kinematics.mano_robot_hand import MANORobotHand


def replay(path_to_pickle: str):
    path = Path(path_to_pickle)
    all_data = np.load(str(path.resolve()), allow_pickle=True)
    meta_data = all_data["meta_data"]
    data = all_data["data"]
    env_class = meta_data["env_class"].lower()
    if "relocate" in env_class:
        env = RelocateEnv(**meta_data["env_kwargs"])
    elif "door" in env_class:
        env = TableDoorEnv(**meta_data["env_kwargs"])
    elif "flip" in env_class:
        env = MugFlipEnv(**meta_data["env_kwargs"])
    else:
        raise ValueError(env_class)

    # Setup viewer and camera
    add_default_scene_light(env.scene, env.renderer)
    gui = GUIBase(env.scene, env.renderer)
    for name, params in DEFAULT_TABLE_TOP_CAMERAS.items():
        gui.create_camera(**params)
    gui.viewer.set_camera_rpy(0, -0.7, 0.01)
    gui.viewer.set_camera_xyz(-0.4, 0, 0.45)
    scene = env.scene

    # Create the mano robot hand during teleoperation
    mano_robot = MANORobotHand(env.scene, env.renderer, init_joint_pos=meta_data["zero_joint_pos"],
                               control_interval=env.frame_skip * scene.get_timestep(), scale=1)

    for i in range(meta_data["data_len"] - 10):
        scene.unpack(data[i]["simulation"])

        for _ in range(2):
            gui.render(render_all_views=False)


if __name__ == '__main__':
    # pkl_path = Path(__file__).parent / "example_teleop_data/relocate_tomato_soup_can.pkl"
    # pkl_path = Path(__file__).parent / "example_teleop_data/flip_mug.pkl"
    pkl_path = Path(__file__).parent / "example_teleop_data/open_door.pkl"
    replay(str(pkl_path))
