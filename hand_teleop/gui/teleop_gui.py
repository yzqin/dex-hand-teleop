from typing import List, Dict, Optional, Callable, Union

import cv2
import numpy as np
import sapien.core as sapien
import torch.utils.dlpack
from sapien.core import Pose
from sapien.core.pysapien import renderer as R
from sapien.utils import Viewer
from hand_teleop.utils.render_scene_utils import add_mesh_to_renderer

DEFAULT_TABLE_TOP_CAMERAS = {
    "left": dict(position=np.array([0, 1, 0.6]), look_at_dir=np.array([0, -1, -0.6]), right_dir=np.array([-1, 0, 0]),
                 name="left_view", ),
    "bird": dict(position=np.array([0, 0, 1.0]), look_at_dir=np.array([0, 0, -1]), right_dir=np.array([0, -1, 0]),
                 name="bird_view", ),
}


class GUIBase:
    def __init__(self, scene: sapien.Scene, renderer: Union[sapien.VulkanRenderer, sapien.KuafuRenderer],
                 resolution=(640, 480), window_scale=0.5):
        use_ray_tracing = isinstance(renderer, sapien.KuafuRenderer)
        self.scene = scene
        self.renderer = renderer
        self.cams: List[sapien.CameraEntity] = []
        self.cam_mounts: List[sapien.ActorBase] = []

        # Context
        self.use_ray_tracing = use_ray_tracing
        if not use_ray_tracing:
            self.context: R.Context = renderer._internal_context
            self.render_scene: R.Scene = scene.get_renderer_scene()._internal_scene
            self.nodes: List[R.Node] = []
        self.sphere_nodes: Dict[str, List[R.Node]] = {}
        self.sphere_model: Dict[str, R.Model] = {}

        # Viewer
        if not use_ray_tracing:
            self.viewer = Viewer(renderer)
            self.viewer.set_scene(scene)
            self.viewer.toggle_axes(False)
            self.viewer.toggle_camera_lines(False)
            self.viewer.set_camera_xyz(-0.3, 0, 0.5)
            self.viewer.set_camera_rpy(0, -1.4, 0)
        self.resolution = resolution
        self.window_scale = window_scale

        # Key down action map
        self.keydown_map: Dict[str, Callable] = {}

        # Common material
        self.viz_mat_hand = self.renderer.create_material()
        self.viz_mat_hand.set_base_color(np.array([0.96, 0.75, 0.69, 1]))
        self.viz_mat_hand.set_specular(0)
        self.viz_mat_hand.set_metallic(0.8)
        self.viz_mat_hand.set_roughness(0)

    def create_camera(self, position, look_at_dir, right_dir, name):
        builder = self.scene.create_actor_builder()
        builder.set_mass_and_inertia(1e-2, Pose(np.zeros(3)), np.ones(3) * 1e-4)
        mount = builder.build_static(name=f"{name}_mount")
        cam = self.scene.add_mounted_camera(name, mount, Pose(), width=self.resolution[0], height=self.resolution[1],
                                            fovy=0.9, fovx=0.9, near=0.1, far=10)

        # Construct camera pose
        look_at_dir = look_at_dir / np.linalg.norm(look_at_dir)
        right_dir = right_dir - np.sum(right_dir * look_at_dir).astype(np.float64) * look_at_dir
        right_dir = right_dir / np.linalg.norm(right_dir)
        up_dir = np.cross(look_at_dir, -right_dir)
        rot_mat_homo = np.stack([look_at_dir, -right_dir, up_dir, position], axis=1)
        pose_mat = np.concatenate([rot_mat_homo, np.array([[0, 0, 0, 1]])])

        # Add camera to the scene
        mount.set_pose(Pose.from_transformation_matrix(pose_mat))
        self.cams.append(cam)
        self.cam_mounts.append(mount)

    @property
    def closed(self):
        return self.viewer.closed

    def _fetch_all_views(self, use_bgr=False) -> List[np.ndarray]:
        views = []
        for cam in self.cams:
            cam.take_picture()
            rgb_tensor = torch.clamp(torch.utils.dlpack.from_dlpack(cam.get_dl_tensor("Color"))[..., :3], 0, 1) * 255
            if use_bgr:
                rgb_tensor = torch.flip(rgb_tensor, [-1])
            rgb = rgb_tensor.type(torch.uint8).cpu().numpy()
            views.append(rgb)
        return views

    def take_single_view(self, camera_name: str, use_bgr=False) -> np.ndarray:
        import cupy
        for cam in self.cams:
            if cam.get_name() == camera_name:
                cam.take_picture()
                dlpack = cam.get_dl_tensor("Color")
                rgb = np.clip(cupy.asnumpy(cupy.from_dlpack(dlpack))[..., :3], 0, 1) * 255
                rgb = rgb.astype(np.uint8)
                if use_bgr:
                    rgb = np.flip(rgb, [-1])
                return rgb
        raise RuntimeError(f"Camera name not found: {camera_name}")

    def render(self, render_all_views=True, additional_views: Optional[List[np.ndarray]] = None, horizontal=True):
        self.scene.update_render()
        self.viewer.render()
        if not self.viewer.closed:
            for key, action in self.keydown_map.items():
                if self.viewer.window.key_down(key):
                    action()
        if (additional_views is not None or len(self.cams) > 0) and render_all_views:
            views = self._fetch_all_views(use_bgr=True)
            if additional_views is not None:
                views.extend(additional_views)

            if horizontal:
                pad = np.ones([views[0].shape[0], 200, 3], dtype=np.uint8) * 255
            else:
                pad = np.ones([200, views[0].shape[1], 3], dtype=np.uint8) * 255

            final_views = [views[0]]
            for i in range(1, len(views)):
                final_views.append(pad)
                final_views.append(views[i])
            axis = 1 if horizontal else 0
            final_views = np.concatenate(final_views, axis=axis)
            target_shape = final_views.shape
            target_shape = (int(target_shape[1] * self.window_scale), int(target_shape[0] * self.window_scale))
            final_views = cv2.resize(final_views, target_shape)
            cv2.imshow("Monitor", final_views)
            cv2.waitKey(1)

    def update_mesh(self, v, f, viz_mat: R.Material, pose: Pose, use_shadow=True, clear_context=False):
        if clear_context:
            for i in range(len(self.nodes)):
                node = self.nodes.pop()
                self.render_scene.remove_node(node)
        node = add_mesh_to_renderer(self.scene, self.renderer, v, f, viz_mat)
        node.set_position(pose.p)
        node.set_rotation(pose.q)
        if use_shadow:
            node.shading_mode = 0
            node.cast_shadow = True
        self.nodes.append(node)

    def register_keydown_action(self, key, action: Callable):
        if key in self.keydown_map:
            raise RuntimeError(f"Key {key} has already been registered")
        self.keydown_map[key] = action

    def add_sphere_visual(self, label, pos: np.ndarray, rgba: np.ndarray = np.array([1, 0, 0, 1]),
                          radius: float = 0.01):
        if label not in self.sphere_model:
            mesh = self.context.create_uvsphere_mesh()
            material = self.context.create_material(emission=np.zeros(4), base_color=rgba, specular=0.4, metallic=0,
                                                    roughness=0.1)
            self.sphere_model[label] = self.context.create_model([mesh], [material])
            self.sphere_nodes[label] = []
        model = self.sphere_model[label]
        node = self.render_scene.add_object(model, parent=None)
        node.set_scale(np.ones(3) * radius)
        self.sphere_nodes[label].append(node)
        node.set_position(pos)
        return node
