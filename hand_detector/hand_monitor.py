from functools import cached_property
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np
import torch
import transforms3d
from matplotlib.cm import get_cmap
from smplx import SMPLX
from timingdecorator.timeit import timeit
import scipy.stats

from hand_detector.hand_mode_detector import SingleHandDetector, HandMocap
from hand_detector.record3d_app import CameraApp
from hand_detector.record3d_app_realsense import RealsenseApp
from hand_teleop.utils.mesh_utils import compute_smooth_shading_normal_np


def frame_cam2operator(point_array: np.ndarray):
    point_array_operator = -point_array[:, [2, 0, 1]]
    point_array_operator[:, 1] = -point_array_operator[:, 1]
    return np.ascontiguousarray(point_array_operator)


def rot_mano2operator(mano_joint_rotation: np.ndarray):
    opencv2sim = np.array([[0, 1, 0], [0, 0, -1], [-1, 0, 0]]).T
    operator2mano = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]])
    return opencv2sim @ mano_joint_rotation @ operator2mano


def mano_rotation_order2joint_order(pred_hand_pose: np.ndarray):
    return


def depth2point_cloud(depth: np.ndarray, intrinsic: np.ndarray):
    v, u = np.indices(depth.shape)  # [H, W], [H, W]
    uv1 = np.stack([u + 0.5, v + 0.5, np.ones_like(depth)], axis=-1)
    points_camera = uv1 @ np.linalg.inv(intrinsic).T * depth[..., None]  # [H, W, 3]
    return points_camera


COLOR_MAP = get_cmap("RdYlGn")


class Record3DSingleHandMotionControl:
    SUPPORT_HAND_MODE = ["right_hand", "left_hand"]

    def __init__(self, hand_mode: str, show_hand=True, virtual_video_file="", need_init=True):
        if hand_mode not in self.SUPPORT_HAND_MODE:
            raise ValueError(
                f"Mode {hand_mode} is invalid. Current {len(self.SUPPORT_HAND_MODE)} mode are supported: "
                f"{self.SUPPORT_HAND_MODE} ")

        # Camera app
        self.camera = CameraApp(file=virtual_video_file)
        # self.camera = RealsenseApp(file=virtual_video_file)
        self.camera.connect_to_device()
        self.camera_mat = self.camera.camera_intrinsics
        self.focal_length = self.camera.camera_intrinsics[0, 0]
        print("Camera Intrinsics:", self.camera_mat)

        # Flag
        self.show_hand = show_hand

        # Init cache
        self.init_root_pose_list = []
        self.init_shape_param_list = []
        self.need_init = need_init
        self.hand_mode = hand_mode
        if need_init:
            self.step = self.init_step
            self.init_process = 0
        else:
            self.step = self.normal_step
            self.init_process = 1.0

        # Init result
        self.shape_dist_var = 0.2
        self.calibrated_offset = np.zeros([3])
        self.calibrated_rotation = np.eye(3)
        self.calibrated_shape_params = np.zeros([10])
        self.calibrated_shape_norm_dist = scipy.stats.norm(np.zeros(10), np.ones(10))

        # Hand detection
        mediapipe_hand_type = hand_mode.split("_")[0].capitalize()
        self.bbox_detector = SingleHandDetector(hand_type=mediapipe_hand_type)

        # Hand joint regression
        hand_detector_dir = Path(__file__).parent
        default_checkpoint_hand = "./extra_data/hand_module/pretrained_weights/pose_shape_best.pth"
        default_checkpoint_body_smpl = './extra_data/smpl'
        self.hand_mocap = HandMocap(str(hand_detector_dir / default_checkpoint_hand),
                                    str(hand_detector_dir / default_checkpoint_body_smpl))

        # Detection cache
        self.previous_bbox = {"left_hand": None, "right_hand": None}

        # Offset based bbox estimation
        self.previous_offset = {"left_hand": np.zeros(3, dtype=np.float32), "right_hand": np.zeros(3, dtype=np.float32)}

    def compute_3d_offset(self, mocap_data: Dict, depth: np.ndarray):
        height, width = depth.shape
        # Image space vertices
        mask_int = np.rint(mocap_data["pred_vertices_img"][:, :2]).astype(int)
        mask_int = np.clip(mask_int, [0, 0], [width - 1, height - 1])
        depth_vertices = depth[mask_int[:, 1], mask_int[:, 0]]
        depth_median = np.nanmedian(depth_vertices)
        depth_valid_mask = np.nonzero(np.abs(depth_vertices - depth_median) < 0.2)[0]
        valid_vertex_depth = depth_vertices[depth_valid_mask]

        # Hand frame vertices
        v_smpl = mocap_data["pred_vertices_smpl"][depth_valid_mask]
        z_smpl = v_smpl[:, 2]
        z_near_to_far_order = np.argsort(z_smpl)

        # Filter depth with same pixel pos to the front position
        valid_mask_int = mask_int[depth_valid_mask, :][z_near_to_far_order, :]
        mask_int_encoding = valid_mask_int[:, 0] * 1e5 + valid_mask_int[:, 1]
        _, unique_indices = np.unique(mask_int_encoding, return_index=True)
        front_indices = z_near_to_far_order[unique_indices]

        # Calculate mean depth from image space and hand frame
        mean_depth_image = np.mean(valid_vertex_depth[front_indices])
        mean_depth_smpl = np.mean(z_smpl[front_indices])
        depth_offset = mean_depth_image - mean_depth_smpl

        offset_img = mocap_data["pred_joints_img"][0, 0:2] - self.camera_mat[0:2, 2]
        offset = np.concatenate([offset_img / self.focal_length * depth_offset, [depth_offset]])

        return offset

    # @timeit
    def normal_step(self) -> Tuple[bool, Dict]:
        rgb, depth = self.camera.fetch_rgb_and_depth()
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        # Detection
        hand_bbox_list = [{"left_hand": None, "right_hand": None}]
        num_bbox, hand_boxes = self.bbox_detector.detect_hand_bbox(rgb)
        if num_bbox < 1:
            return False, dict(rgb=rgb, depth=depth)
        else:
            hand_bbox_list[0][self.hand_mode] = hand_boxes[0]
            self.previous_bbox = hand_bbox_list[0].copy()

        # Joint regression
        pred_output = self.hand_mocap.regress(bgr, hand_bbox_list, add_margin=False)[0]
        mocap_data = pred_output[self.hand_mode]
        offset = self.compute_3d_offset(mocap_data, depth)
        self.previous_offset[self.hand_mode] = offset

        # Output
        pose_params = mocap_data["pred_hand_pose"]
        pose_params[3:] += self.mean_hand_pose
        output = dict(rgb=rgb, depth=depth, origin=self.latest_root_offset,
                      joint=self.compute_operator_space_joint_pos(mocap_data["pred_joints_smpl"]),
                      pose_params=pose_params, bbox=hand_bbox_list[0])
        if self.show_hand:
            vertices, normals = self.compute_operator_space_vertices(mocap_data, self.latest_root_offset)
            output.update({"vertices": vertices, "faces": mocap_data["faces"], "normals": normals})

        # Confidence
        confidence = self.compute_shape_confidence(mocap_data["pred_shape_params"][0])
        output.update({"confidence": confidence})

        shape_error = np.linalg.norm(mocap_data["pred_shape_params"][0] - self.calibrated_shape_params)
        return True, output

    @timeit
    def init_step(self):
        rgb, depth = self.camera.fetch_rgb_and_depth()
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        # Detection
        hand_bbox_list = [{"left_hand": None, "right_hand": None}]
        num_bbox, hand_boxes = self.bbox_detector.detect_hand_bbox(rgb)
        if num_bbox < 1:
            return False, dict(rgb=rgb, depth=depth)
        else:
            hand_bbox_list[0][self.hand_mode] = hand_boxes[0]

        # Joint regression
        pred_output = self.hand_mocap.regress(bgr, hand_bbox_list, add_margin=False)[0]
        mocap_data = pred_output[self.hand_mode]
        offset = self.compute_3d_offset(mocap_data, depth)
        has_init_offset = np.sum(np.abs(self.previous_offset[self.hand_mode])) > 1e-2

        # Stop initialization process and clear data
        if np.linalg.norm(offset - self.previous_offset[self.hand_mode]) > 0.05 and has_init_offset:
            self.init_process = 0
            self.init_root_pose_list.clear()
            self.init_shape_param_list.clear()
            self.previous_offset[self.hand_mode] = offset
        else:
            # Continue init if offset not vary too much
            self.previous_offset[self.hand_mode] = offset
            hand_pose = mocap_data["pred_hand_pose"][3:] + self.hand_mocap.mean_pose
            hand_pose = np.reshape(hand_pose, [15, 3])
            if np.linalg.norm(hand_pose, axis=1).mean() < 0.50:
                self.init_process += 0.02
                self.init_root_pose_list.append(np.concatenate([offset, mocap_data["pred_hand_pose"][:3]]))
                self.init_shape_param_list.append(mocap_data["pred_shape_params"][0])
            else:
                self.init_process = 0
                self.init_root_pose_list.clear()
                self.init_shape_param_list.clear()

        # Compute initialization cache if process reach 100%
        if self.init_process >= 1:
            init_collect_data = np.stack(self.init_root_pose_list)
            num_data = init_collect_data.shape[0]
            weight = (np.arange(num_data) + 1) / np.sum(np.arange(num_data) + 1)
            root_axis_angle = init_collect_data[-1, 3:]
            angle = np.linalg.norm(root_axis_angle)
            axis = root_axis_angle / (angle + 1e-6)
            self.calibrated_offset = np.sum(weight[:, None] * init_collect_data[:, :3], axis=0).astype(np.float32)
            self.calibrated_rotation = rot_mano2operator(transforms3d.axangles.axangle2mat(axis, angle)).T
            self.calibrated_shape_params = np.sum(weight[:, None] * self.init_shape_param_list, axis=0)
            self.calibrated_shape_norm_dist = scipy.stats.norm(self.calibrated_shape_params, self.shape_dist_var)
            print(f"Estimated shape params during init of the operator: {self.calibrated_shape_params}")
            print(f"The variance of shape params: {np.std(self.init_shape_param_list, axis=0)}")

            # Switch step function
            self.step = self.normal_step

        # Output
        pose_params = mocap_data["pred_hand_pose"]
        pose_params[3:] += self.mean_hand_pose
        output = dict(rgb=rgb, depth=depth, origin=offset, joint=mocap_data["pred_joints_smpl"],
                      pose_params=pose_params, bbox=hand_bbox_list[0])
        if self.show_hand:
            vertices, normals = self.compute_operator_space_vertices(mocap_data, np.zeros(3))
            output.update({"vertices": vertices, "faces": mocap_data["faces"], "normals": normals})

        return True, output

    @staticmethod
    def compute_operator_space_vertices(mocap_data: Dict, offset: np.ndarray):
        v_smpl = mocap_data["pred_vertices_smpl"]
        vertices_camera = v_smpl + offset
        vertices = frame_cam2operator(vertices_camera)
        faces = mocap_data["faces"]
        vertex_normals = compute_smooth_shading_normal_np(vertices, faces)
        return vertices, vertex_normals

    @property
    def init_process_color(self):
        return np.array(COLOR_MAP(self.init_process)).astype(np.float32)

    @property
    def initialized(self):
        return self.init_process >= 1

    # Note that the init offset will only influence operator space computation
    @property
    def latest_root_offset(self):
        return self.previous_offset[self.hand_mode] - self.calibrated_offset

    def compute_shape_confidence(self, shape_params: np.ndarray):
        confidence = self.calibrated_shape_norm_dist.pdf(shape_params)
        final_confidence = np.prod(np.clip(confidence, 0, 1))
        return final_confidence

    def compute_operator_space_root_pose(self, motion_data: Dict):
        root_position = frame_cam2operator(self.latest_root_offset[None, :])[0]
        root_axis_angle = motion_data["pose_params"][:3]
        angle = np.linalg.norm(root_axis_angle)
        axis = root_axis_angle / (angle + 1e-6)
        root_rotation = self.calibrated_rotation @ rot_mano2operator(transforms3d.axangles.axangle2mat(axis, angle))
        # TODO: use init frame rotation, should consider the joints for retargeting
        return root_position, root_rotation

    def compute_operator_space_root_qpos(self, motion_data: Dict):
        position, rotation = self.compute_operator_space_root_pose(motion_data)
        euler = transforms3d.euler.mat2euler(rotation, "rxyz")
        root_joint_qpos = np.concatenate([position, euler])
        return root_joint_qpos

    def compute_operator_space_joint_pos(self, joint_pos: np.ndarray):
        human_hand_joints = frame_cam2operator(joint_pos + self.latest_root_offset)
        return human_hand_joints

    def compute_hand_zero_pos(self):
        if not self.initialized:
            raise RuntimeError(f"Can not perform hand shape based computation before initialization")
        shape_params = torch.from_numpy(self.calibrated_shape_params.astype(np.float32))[None, :].cuda()
        smplx: SMPLX = self.hand_mocap.model.smplx
        hand_index = [21] + list(range(40, 55)) + list(range(71, 76))  # 21 for right wrist. 20 finger joints
        smplx_hand_to_panoptic = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]
        body_pose = torch.zeros((1, 63)).float().cuda()
        with torch.no_grad():
            output = smplx(body_pose=body_pose, right_hand_pose=-smplx.right_hand_mean, betas=shape_params,
                           return_verts=True)
        joints = output.joints
        hand_joints = joints[:, hand_index, :][:, smplx_hand_to_panoptic, :]
        joint_pos = hand_joints - hand_joints[:, 0:1, :]
        joint_pos = joint_pos.detach().cpu().numpy()[0]
        return joint_pos

    @cached_property
    def mean_hand_pose(self):
        return self.hand_mocap.model.smplx.right_hand_mean.cpu().numpy()
