import numpy as np
import sapien.core as sapien
import transforms3d.euler

from hand_teleop.env.sim_env.base import BaseSimulationEnv
from hand_teleop.utils.ycb_object_utils import YCB_SIZE, load_ycb_object


class MugFlipEnv(BaseSimulationEnv):
    def __init__(self, use_gui=True, frame_skip=5, object_scale=1, randomness_scale=1, friction=0.3,
                 **renderer_kwargs):
        super().__init__(use_gui=use_gui, frame_skip=frame_skip, **renderer_kwargs)

        # Construct scene
        scene_config = sapien.SceneConfig()
        self.scene = self.engine.create_scene(config=scene_config)
        self.scene.set_timestep(0.004)
        self.friction = friction
        self.object_scale = object_scale
        self.randomness_scale = randomness_scale

        # Load table
        self.table = self.create_table(table_height=0.6, table_half_size=[0.35, 0.35, 0.025])

        # Load object
        material = self.engine.create_physical_material(self.friction, self.friction * 1.5, 0.1)
        self.manipulated_object = load_ycb_object(self.scene, "mug", material=material)
        self.original_object_pos = np.zeros(3)

    def reset_env(self):
        pose = self.generate_random_init_pose(self.randomness_scale)
        self.manipulated_object.set_pose(pose)
        self.original_object_pos = pose.p

    def generate_random_init_pose(self, randomness_scale):
        pos = self.np_random.uniform(low=-0.1, high=0.1, size=2) * randomness_scale
        ycb_height = YCB_SIZE["mug"][1] / 2 * self.object_scale
        random_z_rotate = self.np_random.uniform(0, np.pi)
        orientation = transforms3d.euler.euler2quat(-np.pi / 2, 0, random_z_rotate)
        position = np.array([pos[0], pos[1], ycb_height])
        pose = sapien.Pose(position, orientation)
        return pose


def env_test():
    from sapien.utils import Viewer
    from constructor import add_default_scene_light
    env = MugFlipEnv()
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
