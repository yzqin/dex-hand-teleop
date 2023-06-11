import os
import time
from threading import Event

import cv2
import imageio
import numpy as np
from PIL import Image
from record3d import Record3DStream


class CameraApp:
    def __init__(self, file: str = ""):
        self.rgb_video_file = file
        self.event = Event()
        self.session = None

        self.rgb_stream = None
        self.depth_stream = None
        self.read_count = 0

    def on_new_frame(self):
        """
        This method is called from non-main thread, therefore cannot be used for presenting UI.
        """
        self.event.set()  # Notify the main thread to stop waiting and process new frame.

    def on_stream_stopped(self):
        print('Stream stopped')

    def connect_to_device(self, dev_idx=0):
        if not os.path.exists(self.rgb_video_file):
            print('Searching for devices')
            devs = Record3DStream.get_connected_devices()
            print('{} device(s) found'.format(len(devs)))
            for dev in devs:
                print('\tID: {}\n\tUDID: {}\n'.format(dev.product_id, dev.udid))

            if len(devs) <= dev_idx:
                raise RuntimeError('Cannot connect to device #{}, try different index.'.format(dev_idx))

            dev = devs[dev_idx]
            self.session = Record3DStream()
            self.session.on_new_frame = self.on_new_frame
            self.session.on_stream_stopped = self.on_stream_stopped
            self.session.connect(dev)  # Initiate connection and start capturing

            # Wait for camera to connected, here we use a hack to check whether we can really get the info from camera
            time_out = 2
            while self.camera_intrinsics.sum() < 10 and time_out >= 0:
                time.sleep(1e-3)
                time_out -= 1e-3
        else:
            print(f"Streaming from video file {os.path.abspath(self.rgb_video_file)}")
            self.rgb_stream = cv2.VideoCapture(self.rgb_video_file)
            self.depth_stream = np.load(self.rgb_video_file.replace("rgb", "depth").replace("mp4", "npz"))['depth']

    @staticmethod
    def _get_intrinsic_mat_from_coeffs(coeffs):
        return np.array([[coeffs.fx, 0, coeffs.tx],
                         [0, coeffs.fy, coeffs.ty],
                         [0, 0, 1]])

    @property
    def camera_intrinsics(self):
        if self.session is None:
            intrinsic_mat = np.array([[804.5928, 0.0, 357.6741333]
                                      [0.0, 804.5928, 474.83026123]
                                      [0., 0., 1.]])
        else:
            intrinsic_mat = self._get_intrinsic_mat_from_coeffs(self.session.get_intrinsic_mat())
        return intrinsic_mat

    def fetch_rgb_and_depth(self):
        if not self.rgb_video_file:
            self.event.wait(1)
            depth = np.transpose(self.session.get_depth_frame(), [1, 0])
            rgb = np.transpose(self.session.get_rgb_frame(), [1, 0, 2])

            is_true_depth = depth.shape[0] == 480
            if is_true_depth:
                depth = cv2.flip(depth, 1)
                rgb = cv2.flip(rgb, 1)

            return rgb, depth
        else:
            while self.rgb_stream.isOpened():
                success_rgb, bgr_frame = self.rgb_stream.read()
                depth_frame = self.depth_stream[self.read_count]
                self.read_count += 1

                if not success_rgb:
                    print("Can't receive frame (stream end?). Exiting ...")
                    break
                return cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB), depth_frame

    def record(self, rgb_filename):
        if len(self.rgb_video_file) > 0:
            raise RuntimeError(f"When recoding video, the input file must be empty string")
        saved_rgb = []
        saved_depth = []

        while True:
            rgb, depth = self.fetch_rgb_and_depth()
            cv2.imshow('frame', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            saved_rgb.append(Image.fromarray(rgb))
            saved_depth.append(depth)
            time.sleep(1.0/20)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        imageio.mimsave(rgb_filename, saved_rgb[5:-5], "mp4", fps=15)
        np.savez(rgb_filename.replace("rgb", "depth").replace("mp4", "npz"), depth=saved_depth)


if __name__ == '__main__':
    from time import time as get_time

    app = CameraApp(file="")
    app.connect_to_device()
    filename = f"video/rgb_{1:0>4d}.mp4"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    app.record(filename)
    # for _ in range(50000):
    #     tic = get_time()
    #     for _ in range(1):
    #         rgb_image, depth = app.fetch_rgb_and_depth()
    #     print(f"Time for one frame: {get_time() - tic}s")
    #     # time.sleep(0.2)
    #     bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    #     cv2.imshow("record3d", bgr)
    #     cv2.imwrite("temp.png", bgr)
    #     # cv2.imshow("record3", rgb_image)
    #     cv2.waitKey(1)
