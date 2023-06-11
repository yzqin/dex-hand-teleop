import numpy as np
import sapien.core as sapien
import transforms3d

from hand_teleop.env.sim_env.base import BaseSimulationEnv
from hand_teleop.real_world import lab
from hand_teleop.utils.render_scene_utils import set_entity_color
from hand_teleop.utils.ycb_object_utils import load_ycb_object, YCB_SIZE, YCB_ORIENTATION


class RelocateEnv(BaseSimulationEnv):
    def __init__(self, use_gui=True, frame_skip=5, object_category="YCB", object_name="tomato_soup_can",
                 object_scale=1.0, randomness_scale=1, friction=1, use_visual_obs=False, **renderer_kwargs):
        super().__init__(use_gui=use_gui, frame_skip=frame_skip, use_visual_obs=use_visual_obs, **renderer_kwargs)

        # Object info
        self.object_category = object_category
        self.object_name = object_name
        self.object_scale = object_scale
        self.object_height = object_scale * YCB_SIZE[self.object_name][2] / 2
        self.target_pose = sapien.Pose()

        # Dynamics info
        self.randomness_scale = randomness_scale
        self.friction = friction

        # Construct scene
        self.scene = self.engine.create_scene()
        self.scene.set_timestep(0.005)

        # Dummy camera creation to initial geometry object
        if self.renderer and not self.no_rgb:
            cam = self.scene.add_camera("init_not_used", width=10, height=10, fovy=1, near=0.1, far=1)
            self.scene.remove_camera(cam)

        # Load table
        self.tables = self.create_table(table_height=0.6, table_half_size=[0.65, 0.65, 0.025])

        # Load object
        if self.object_category.lower() == "ycb":
            self.manipulated_object = load_ycb_object(self.scene, object_name)
            self.target_object = load_ycb_object(self.scene, object_name, visual_only=True)
            if self.use_visual_obs:
                self.target_object.hide_visual()
        else:
            raise NotImplementedError
        if self.use_gui:
            set_entity_color([self.target_object], [0, 1, 0, 0.6])

    def generate_random_object_pose(self, randomness_scale):
        pos = self.np_random.uniform(low=-0.1, high=0.1, size=2) * randomness_scale
        ycb_orientation = YCB_ORIENTATION[self.object_name]
        position = np.array([pos[0], pos[1], self.object_height])
        pose = sapien.Pose(position, ycb_orientation)
        return pose

    def generate_random_target_pose(self, randomness_scale):
        pos = self.np_random.uniform(low=-0.2, high=0.2, size=2) * randomness_scale
        height = 0.25
        position = np.array([pos[0], pos[1], height])
        euler = self.np_random.uniform(low=np.deg2rad(-15), high=np.deg2rad(15), size=2)
        ycb_orientation = YCB_ORIENTATION[self.object_name]
        quaternion = transforms3d.euler.euler2quat(euler[0], euler[1], 0)
        pose = sapien.Pose(position, transforms3d.quaternions.qmult(ycb_orientation, quaternion))
        return pose

    def reset_env(self):
        pose = self.generate_random_object_pose(self.randomness_scale)
        self.manipulated_object.set_pose(pose)

        # Target pose
        pose = self.generate_random_target_pose(self.randomness_scale)
        self.target_object.set_pose(pose)
        self.target_pose = pose


def env_test():
    from sapien.utils import Viewer
    from constructor import add_default_scene_light
    env = RelocateEnv(object_category="ycb", object_name="mustard_bottle")
    env.reset_env()
    viewer = Viewer(env.renderer)
    viewer.set_scene(env.scene)
    add_default_scene_light(env.scene, env.renderer)
    env.viewer = viewer

    while not viewer.closed:
        env.simple_step()
        env.render()


if __name__ == '__main__':
    env_test()
