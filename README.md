# From One Hand to Multiple Hands: Imitation Learning for Dexterous Manipulation from Single-Camera Teleoperation

[[Project Page]](https://yzqin.github.io/dex-teleop-imitation/) [[Paper]](https://arxiv.org/abs/2204.12490) [[Slides]](https://docs.google.com/presentation/d/186iJVvY9B0D_KSKKQFV1ketIKiTK1L_A/edit?usp=sharing&ouid=108317450590466198031&rtpof=true&sd=true)
-----

[From One Hand to Multiple Hands: Imitation Learning for Dexterous Manipulation from Single-Camera Teleoperation](https://yzqin.github.io/dex-teleop-imitation/)

Yuzhe Qin, Hao Su*, Xiaolong Wang*, RA-L & IROS 2022.

Here we provide a simple system for teleoperation with dexterous hand in simulation, as well as a conversion between
MANO/SMPL hand format and simulatable robot model (.urdf) format

![Teaser](docs/teleop_teaser.png)

## Installation

```shell
git clone git@github.com:yzqin/dex-hand-teleop.git
cd dex-hand-teleop 
conda create --name dexteleop python=3.8
conda activate dexteleop
pip install -e .
```

Download data file for the scene
from [Google Drive Link](https://drive.google.com/file/d/1Xe3jgcIUZm_8yaFUsHnO7WJWr8cV41fE/view?usp=sharing).
Place the `day.ktx` at `assets/misc/ktx/day.ktx`.

```shell
pip install gdown
gdown https://drive.google.com/uc?id=1Xe3jgcIUZm_8yaFUsHnO7WJWr8cV41fE
```

### Download additional weights for hand detection

Follow the guidelines provided by
the [FrankMocap](https://github.com/facebookresearch/frankmocap/blob/main/docs/INSTALL.md)
project to download the weight files for the SMPLX hand model and hand pose detector.

Please note that you only need to obtain a subset of the files utilized in the original FrankMocap repository. If the
process is completed successfully, the final file structure should appear as follows within
the `hand_detector/extra_data`
directory.

```shell
├── extra_data
│   ├── hand_module
│   │   ├── mean_mano_params.pkl
│   │   ├── pretrained_weights
│   │   │   └── pose_shape_best.pth
│   │   └── SMPLX_HAND_INFO.pkl
│   └── smpl
│       └── SMPLX_NEUTRAL.pkl
```

## File Structure

- `hand_teleop`: main content for the environment, utils, and other staff needs for simulation. It utilizes the same
  code
  structure as [DexPoint](https://github.com/yzqin/dexpoint-release).
- `hand_detector`: perception code and model to detect hand bbox and regress hand pose in SMPLX format
- `assets`: robot and object models, and other static files
- `example`: entry files to learn how to use the teleoperation and the customized robot hand

## Quick Start

1. Use the `customized robot hand` proposed in the paper:

Run [example/customized_robot_hand.py](example/customized_robot_hand.py) to learn how to construct and control a robot
hand based on SMPLX hand parameterization.

Run [example/teleop_collect_data.py](example/teleop_collect_data.py) to learn how to utilize the teleoperation system to
perform a manipulation data and collect the data as a pickle file.

## Acknowledgements

We would like to thank the following people for providing valuable feedback when testing the system.

[Jiarui Xu](https://jerryxu.net/), [Crystina Zhang](https://crystina-z.github.io/),
[Binghao Huang](https://binghao-huang.github.io/), [Jiashun Wang](https://jiashunwang.github.io/),
[Ruihan Yang](https://rchalyang.github.io/), [Jianglong Ye](https://jianglongye.com/),
[Yang Fu](https://oasisyang.github.io/), [Jiteng Mu](https://jitengmu.github.io/).

## Bibtex

```
@misc{qin2022from,
  author         = {From One Hand to Multiple Hands: Imitation Learning for Dexterous Manipulation from Single-Camera Teleoperation},
  title          = {Qin, Yuzhe and Su, Hao and Wang, Xiaolong},
  archivePrefix  = {arXiv},
  year           = {2022},
}

```


