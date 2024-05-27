import os
import cv2
import time
import imageio
import numpy as np
from PIL import Image
import pyrealsense2 as pyrs
from threading import Thread


class RealsenseApp:
    def __init__(self, file: str = ""):
        self.stopped = False
        self.rgb_video_file = file
        self.rgb_stream, self.depth_stream, self.read_count = None, None, 0
        self.align, self.stream, self.cam_k, self.frame, self.depth_scale = None, None, None, None, 1

    def connect_to_device(self):
        if not os.path.exists(self.rgb_video_file):
            cfg = pyrs.config()
            cfg.enable_stream(pyrs.stream.color, 640, 480, pyrs.format.rgb8, 30)
            cfg.enable_stream(pyrs.stream.depth, 640, 480, pyrs.format.z16, 30)
            self.align = pyrs.align(pyrs.stream.color)
            self.stream = pyrs.pipeline()
            dev = self.stream.start(cfg)
            cam_int = dev.get_stream(pyrs.stream.color).as_video_stream_profile().get_intrinsics()
            self.depth_scale = dev.get_device().first_depth_sensor().get_depth_scale()
            self.cam_k = np.array([[cam_int.fx, 0, cam_int.ppx],
                                   [0, cam_int.fy, cam_int.ppy],
                                   [0, 0, 1]])
            self.frame = self.grab_realsense()
            Thread(target=self.update, args=()).start()
        else:
            print(f"Streaming from video file {os.path.abspath(self.rgb_video_file)}")
            self.rgb_stream = cv2.VideoCapture(self.rgb_video_file)
            self.depth_stream = np.load(self.rgb_video_file.replace("rgb", "depth").replace("mp4", "npz"))['depth']

    @property
    def camera_intrinsics(self):
        if self.stream is None:
            return np.array([[614.450317, 0., 332.668884],
                             [0., 614.965996, 246.103592],
                             [0., 0., 1.]])
        else:
            return self.cam_k

    def stop(self):
        self.stopped = True

    def update(self):
        while True:
            if self.stopped:
                return
            self.frame = self.grab_realsense()

    def grab_realsense(self):
        frames = self.align.process(self.stream.wait_for_frames())
        color = frames.get_color_frame().get_data()
        depth = frames.get_depth_frame().get_data()
        if not color or not depth:
            return
        color = np.asarray(color)
        depth = np.asarray(depth) * self.depth_scale
        return color, depth

    def fetch_rgb_and_depth(self):
        if not self.rgb_video_file:
            return self.frame
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
        saved_rgb = []
        saved_depth = []

        while True:
            tick = time.time()
            rgb, depth = self.fetch_rgb_and_depth()
            cv2.imshow('frame', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            saved_rgb.append(Image.fromarray(rgb))
            saved_depth.append(depth.copy())
            sleep = (1.0/15) - (time.time() - tick)
            if sleep > 0:
                time.sleep(sleep)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.stop()
        imageio.mimsave(rgb_filename, saved_rgb[5:-5], "mp4", fps=15)
        np.savez(rgb_filename.replace("rgb", "depth").replace("mp4", "npz"), depth=saved_depth)


if __name__ == '__main__':
    app = RealsenseApp(file="")
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
