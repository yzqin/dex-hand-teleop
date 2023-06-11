import os
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
import torch
from mediapipe.python.solution_base import SolutionBase
from torchvision.transforms import transforms

from hand_detector.handmocap.hand_modules.h3dw_model import H3DWModel
from hand_detector.handmocap.hand_modules.test_options import TestOptions
from hand_detector.mocap_utils.coordconv import convert_smpl_to_bbox, convert_bbox_to_oriIm


class MediapipeBBoxHand(SolutionBase):
    def __init__(self,
                 static_image_mode=False,
                 max_num_hands=2,
                 model_complexity=1,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        """
        See original mp.solutions.hands.Hands for documentations.
        This class only do one more thing than the original Hands class:
         add hand detection bounding box results into the pipeline with "hand_rects"
        """

        _BINARYPB_FILE_PATH = 'mediapipe/modules/hand_landmark/hand_landmark_tracking_cpu.binarypb'
        super().__init__(
            binary_graph_path=_BINARYPB_FILE_PATH,
            side_inputs={
                'model_complexity': model_complexity,
                'num_hands': max_num_hands,
                'use_prev_landmarks': not static_image_mode,
            },
            calculator_params={
                'palmdetectioncpu__TensorsToDetectionsCalculator.min_score_thresh':
                    min_detection_confidence,
                'handlandmarkcpu__ThresholdingCalculator.threshold':
                    min_tracking_confidence,
            },
            outputs=[
                'multi_hand_landmarks', 'multi_hand_world_landmarks',
                'multi_handedness', 'hand_rects'
            ])

    def process(self, image: np.ndarray) -> NamedTuple:
        """
        See original mp.solutions.hands.Hands for documentations.
        """

        return super().process(input_data={'image': image})


class SingleHandDetector:
    def __init__(self, hand_type="Right", min_detection_confidence=0.8, min_tracking_confidence=0.8, max_num_hand=1,
                 selfie=False):
        self.hand_detector = MediapipeBBoxHand(
            static_image_mode=False,
            max_num_hands=max_num_hand,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence)
        self.selfie = selfie
        inverse_hand_dict = {"Right": "Left", "Left": "Right"}
        self.detected_hand_type = hand_type if selfie else inverse_hand_dict[hand_type]

    @staticmethod
    def _mediapipe_bbox_to_numpy(image, hand_rect):
        image_size = np.array(image.shape[:2][::-1])
        center = np.array([hand_rect.x_center, hand_rect.y_center])
        size = np.array([hand_rect.width, hand_rect.height])
        bbox = np.zeros(4)
        bbox[:2] = (center - size / 2) * image_size
        bbox[2:] = size * image_size
        return bbox

    def detect_hand_bbox(self, rgb):
        results = self.hand_detector.process(rgb)
        if not results.multi_hand_landmarks:
            return 0, None

        desired_hand_num = -1
        for i in range(len(results.multi_hand_landmarks)):
            label = results.multi_handedness[i].ListFields()[0][1][0].label
            if label == self.detected_hand_type:
                desired_hand_num = i
                break
        if desired_hand_num < 0:
            return 0, None

        bbox = results.hand_rects[desired_hand_num]
        num_box = len(results.multi_hand_landmarks)
        return num_box, self._mediapipe_bbox_to_numpy(rgb, bbox)[None, :]


class HandMocap:
    def __init__(self, checkpoint_path, smpl_dir):
        # For image transform
        transform_list = [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        self.normalize_transform = transforms.Compose(transform_list)

        # Load Hand network
        self.opt = TestOptions().parse([])

        # Default options
        self.opt.single_branch = True
        self.opt.main_encoder = "resnet50"

        hand_detector_dir = Path(__file__).parent
        self.opt.model_root = str(hand_detector_dir / "./extra_data")
        self.opt.smplx_model_file = os.path.join(smpl_dir, 'SMPLX_NEUTRAL.pkl')

        self.opt.batchSize = 1
        self.opt.phase = "test"
        self.opt.nThreads = 0
        self.opt.which_epoch = -1
        self.opt.checkpoint_path = checkpoint_path

        self.opt.serial_batches = True  # no shuffle
        self.opt.no_flip = True  # no flip
        self.opt.process_rank = -1

        self.mean_pose = np.array([0.1117, -0.0429, 0.4164, 0.1088, 0.0660,
                                   0.7562, -0.0964, 0.0909, 0.1885, -0.1181, -0.0509, 0.5296, -0.1437,
                                   -0.0552, 0.7049, -0.0192, 0.0923, 0.3379, -0.4570, 0.1963, 0.6255,
                                   -0.2147, 0.0660, 0.5069, -0.3697, 0.0603, 0.0795, -0.1419, 0.0859,
                                   0.6355, -0.3033, 0.0579, 0.6314, -0.1761, 0.1321, 0.3734, 0.8510,
                                   -0.2769, 0.0915, -0.4998, -0.0266, -0.0529, 0.5356, -0.0460, 0.2774])

        # self.opt.which_epoch = str(epoch)
        self.model = H3DWModel(self.opt)
        if not self.model.success_load:
            raise RuntimeError(f"Check points {self.opt.checkpoint_path} does not exist")
        self.model.eval()

    def __pad_and_resize(self, img, hand_bbox, add_margin, final_size=224):
        ori_height, ori_width = img.shape[:2]
        min_x, min_y = hand_bbox[:2].astype(np.int32)
        width, height = hand_bbox[2:].astype(np.int32)
        max_x = min_x + width
        max_y = min_y + height

        if width > height:
            margin = (width - height) // 2
        else:
            margin = (height - width) // 2
        min_y = max(min_y - margin, 0)
        max_y = min(max_y + margin, ori_height)
        min_x = max(min_x - margin, 0)
        max_x = min(max_x + margin, ori_width)

        # add additional margin
        if add_margin:
            margin = int(0.3 * (max_y - min_y))  # if use loose crop, change 0.3 to 1.0
            min_y = max(min_y - margin, 0)
            max_y = min(max_y + margin, ori_height)
            min_x = max(min_x - margin, 0)
            max_x = min(max_x + margin, ori_width)

        img_cropped = img[int(min_y):int(max_y), int(min_x):int(max_x), :]
        new_size = max(max_x - min_x, max_y - min_y)
        new_img = np.zeros((new_size, new_size, 3), dtype=np.uint8)
        new_img[:(max_y - min_y), :(max_x - min_x), :] = img_cropped

        # resize to 224 * 224
        new_img = cv2.resize(new_img, (final_size, final_size))

        ratio = final_size / new_size
        return new_img, ratio, (min_x, min_y, max_x - min_x, max_y - min_y)

    def __process_hand_bbox(self, raw_image, hand_bbox, hand_type, add_margin=True):
        """
        args:
            original image,
            bbox: (x0, y0, w, h)
            hand_type ("left_hand" or "right_hand")
            add_margin: If the input hand bbox is a tight bbox, then set this value to True, else False
        output:
            img_cropped: 224x224 cropped image (original colorvalues 0-255)
            norm_img: 224x224 cropped image (normalized color values)
            bbox_scale_ratio: scale factor to convert from original to cropped
            bbox_top_left_origin: top_left corner point in original image cooridate
        """
        # print("hand_type", hand_type)

        assert hand_type in ['left_hand', 'right_hand']
        img_cropped, bbox_scale_ratio, bbox_processed = self.__pad_and_resize(raw_image, hand_bbox, add_margin)

        # horizontal Flip to make it as right hand
        if hand_type == 'left_hand':
            img_cropped = np.ascontiguousarray(img_cropped[:, ::-1, :], img_cropped.dtype)

        # img normalize
        norm_img = self.normalize_transform(img_cropped).float()
        return img_cropped, norm_img, bbox_scale_ratio, bbox_processed

    def regress(self, img_original, hand_bbox_list, add_margin=False):
        """
            args:
                img_original: original raw image (BGR order by using cv2.imread)
                hand_bbox_list: [
                    dict(
                        left_hand = [x0, y0, w, h] or None
                        right_hand = [x0, y0, w, h] or None
                    )
                    ...
                ]
                add_margin: whether to do add_margin given the hand bbox
            outputs:
                To be filled
            Note:
                Output element can be None. This is to keep the same output size with input bbox
        """
        pred_output_list = list()

        for hand_bboxes in hand_bbox_list:
            if hand_bboxes is None:  # Should keep the same size with bbox size
                pred_output_list.append(None)
                continue

            pred_output = dict(
                left_hand=None,
                right_hand=None
            )

            for hand_type in hand_bboxes:
                bbox = hand_bboxes[hand_type]

                if bbox is None:
                    continue
                else:
                    img_cropped, norm_img, bbox_scale_ratio, bbox_processed = \
                        self.__process_hand_bbox(img_original, bbox, hand_type, add_margin)

                    with torch.no_grad():
                        self.model.set_input_imgonly({'img': norm_img.unsqueeze(0)})
                        self.model.test()
                        pred_res = self.model.get_pred_result()

                        cam = pred_res['cams'][0, :]  # scale, tranX, tranY
                        pred_vertex_origin = pred_res['pred_verts'][0]
                        faces = self.model.right_hand_faces_local
                        pred_pose = pred_res['pred_pose_params'].copy()
                        pred_joints = pred_res['pred_joints_3d'].copy()[0]

                        if hand_type == 'left_hand':
                            cam[1] *= -1
                            pred_vertex_origin[:, 0] *= -1
                            faces = faces[:, ::-1]
                            pred_pose[:, 1::3] *= -1
                            pred_pose[:, 2::3] *= -1
                            pred_joints[:, 0] *= -1

                        # Different from original implementation, here we set the root of hand to be origin (0,0,0)
                        pred_output[hand_type] = dict()
                        pred_output[hand_type]['pred_vertices_smpl'] = pred_vertex_origin - pred_joints[0:1, :]
                        pred_output[hand_type]['pred_joints_smpl'] = pred_joints - pred_joints[0:1, :]
                        pred_output[hand_type]['pred_shape_params'] = pred_res["pred_shape_params"]
                        pred_output[hand_type]['faces'] = faces

                        pred_output[hand_type]['bbox_scale_ratio'] = bbox_scale_ratio
                        pred_output[hand_type]['bbox_top_left'] = np.array(bbox_processed[:2])
                        pred_output[hand_type]['pred_camera'] = cam

                        # pred hand pose & shape params & hand joints 3d
                        pred_output[hand_type]['pred_hand_pose'] = pred_pose[0]  # (48)
                        pred_output[hand_type]['pred_hand_betas'] = pred_res['pred_shape_params'][0]  # (10)

                        # Convert vertices into bbox & image space
                        cam_scale = cam[0]
                        cam_trans = cam[1:]
                        vert_smplcoord = pred_vertex_origin.copy()
                        joints_smplcoord = pred_joints.copy()

                        vert_bboxcoord = convert_smpl_to_bbox(
                            vert_smplcoord, cam_scale, cam_trans, bAppTransFirst=True)  # SMPL space -> bbox space
                        joints_bboxcoord = convert_smpl_to_bbox(
                            joints_smplcoord, cam_scale, cam_trans, bAppTransFirst=True)  # SMPL space -> bbox space

                        hand_boxScale_o2n = pred_output[hand_type]['bbox_scale_ratio']
                        hand_bboxTopLeft = pred_output[hand_type]['bbox_top_left']

                        vert_imgcoord = convert_bbox_to_oriIm(
                            vert_bboxcoord, hand_boxScale_o2n, hand_bboxTopLeft,
                            img_original.shape[1], img_original.shape[0])
                        pred_output[hand_type]['pred_vertices_img'] = vert_imgcoord

                        joints_imgcoord = convert_bbox_to_oriIm(
                            joints_bboxcoord, hand_boxScale_o2n, hand_bboxTopLeft,
                            img_original.shape[1], img_original.shape[0])
                        pred_output[hand_type]['pred_joints_img'] = joints_imgcoord

                        # offset = (joints_smplcoord[0]) * cam_scale * 112
                        # offset = offset / hand_boxScale_o2n
                        # print(1 / offset)

                        depth = 595.85 * hand_boxScale_o2n / cam_scale / 112

            pred_output_list.append(pred_output)

        return pred_output_list


def main():
    from matplotlib import pyplot as plt
    hand_detector_dir = Path(__file__).parent
    default_checkpoint_hand = "./extra_data/hand_module/pretrained_weights/pose_shape_best.pth"
    default_checkpoint_body_smpl = './extra_data/smpl'
    detector = SingleHandDetector()
    hand_mocap = HandMocap(str(hand_detector_dir / default_checkpoint_hand),
                           str(hand_detector_dir / default_checkpoint_body_smpl))

    vid = cv2.VideoCapture("rgb_0001.mp4")
    while True:
        ret, image_bgr = vid.read()
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        _, bbox = detector.detect_hand_bbox(image_rgb)
        hand_bbox_list = [{"left_hand": None, "right_hand": None}]
        hand_bbox_list[0]["right_hand"] = bbox[0]
        pred_output = hand_mocap.regress(image_bgr, hand_bbox_list, add_margin=False)[0]
        # print(pred_output["right_hand"]["pred_joints_img"])
        plt.imshow(image_rgb)


if __name__ == '__main__':
    main()
