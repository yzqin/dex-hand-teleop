from pathlib import Path

from setuptools import setup, find_packages

root_dir = Path(__file__).parent
long_description = (root_dir / "README.md").read_text(encoding="utf-8")

setup(
    name="HandTeleop",
    version="1.0.0",
    author="Yuzhe Qin",
    author_email="y1qin@ucsd.edu",
    keywords="dexterous-manipulation data-collection teleoperation",
    description="From One Hand to Multiple Hands: Imitation Learning for Dexterous Manipulation from Single-Camera Teleoperation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://yzqin.github.io/dex-teleop-imitation/",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "sapien==2.1.0",
        "natsort",
        "numpy",
        "transforms3d",
        "gym==0.25.2",
        "open3d>=0.15.2",
        "imageio",
        "torch>=1.11.0",
        "nlopt",
        "smplx",
        "opencv-python",
        "mediapipe",
        "torchvision",
        "record3d",
        "pyrealsense2"
    ],
    extras_require={"tests": ["pytest", "black", "isort"]},
)
