import numpy as np
from sapien.core import Pose

CAM2ROBOT = Pose.from_transformation_matrix(np.array(
    [[0.60346958, 0.36270068, -0.7101216, 0.962396],
     [0.7960018, -0.22156729, 0.56328419, -0.35524235],
     [0.04696384, -0.90518294, -0.42241951, 0.31896536],
     [0., 0., 0., 1.]]
))

DESK2ROBOT_Z_AXIS = -0.1352233

# Relocate
RELOCATE_BOUND = [0.2, 0.8, -0.4, 0.4, DESK2ROBOT_Z_AXIS + 0.005, 0.6]

ROBOT2BASE = Pose(p=np.array([-0.55, 0., -DESK2ROBOT_Z_AXIS]))

# Table size
TABLE_XY_SIZE = np.array([0.6, 1.2])
TABLE_ORIGIN = np.array([0, -0.15])

