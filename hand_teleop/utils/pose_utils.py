import numpy as np
import nlopt
from pytorch3d import transforms
import torch
import transforms3d


class EulerOptimizer:
    def __init__(self, norm_lambda=1e-3, hybrid=True, hybrid_threshold=0.1):
        self.norm_lambda = norm_lambda
        self.last_value = np.zeros(3)
        self.opt = nlopt.opt(nlopt.LD_SLSQP, 3)
        self.opt.set_ftol_abs(1e-5)
        self.hybrid = hybrid
        self.hybrid_threshold = hybrid_threshold

    def init_euler_angle(self, euler_angle: np.ndarray):
        if euler_angle.shape != (3,):
            raise ValueError(f"Euler angle should be in shape (3,), but given f{euler_angle.shape}")
        self.last_value = euler_angle

    def _get_objective_function(self, target_rot_mat: np.ndarray):
        def objective(x: np.ndarray, grad: np.ndarray) -> float:
            require_grad = grad.size > 0
            torch_euler = torch.from_numpy(x.astype(np.float64))[None, :]
            torch_euler.requires_grad_(require_grad)
            torch_mat = transforms.euler_angles_to_matrix(torch_euler, "XYZ")  # pytorch3d convention: moving frame

            torch_target_mat = torch.from_numpy(target_rot_mat.astype(np.float64))[None, :]
            torch_target_mat.requires_grad_(False)
            torch_last_value = torch.from_numpy(self.last_value.astype(np.float64))[None, :]
            torch_last_value.requires_grad_(False)

            rot_distance_cos = -transforms.so3_relative_angle(torch_mat, torch_target_mat, cos_angle=True)
            norm_distance = self.norm_lambda * torch.norm(torch_last_value - torch_euler)
            loss = (rot_distance_cos + norm_distance).sum()
            result = loss.cpu().detach().item()

            if require_grad:
                loss.backward()
                grad_euler = torch_euler.grad.cpu().numpy()[0]
                grad[:] = grad_euler[:]

            return result

        return objective

    def optimize(self, target_rot_mat: np.ndarray, save_result=True, verbose=False):
        if target_rot_mat.shape != (3, 3):
            raise ValueError(f"Rotation matrix should be in shape (3, 3), but given f{target_rot_mat.shape}")
        need_opt = True
        if self.hybrid:
            analytical_euler_angle = np.array(transforms3d.euler.mat2euler(target_rot_mat, "rxyz"))
            if np.linalg.norm(analytical_euler_angle - self.last_value) < self.hybrid_threshold:
                need_opt = False
                euler_angle = analytical_euler_angle

        if need_opt:
            last_value = self.last_value
            last_value = list(last_value)
            objective_fn = self._get_objective_function(target_rot_mat)
            self.opt.set_min_objective(objective_fn)
            try:
                euler_angle = self.opt.optimize(last_value)
            except RuntimeError as e:
                print(e, self.opt.get_errmsg())
                euler_angle = self.last_value
            if verbose:
                min_value = self.opt.last_optimum_value()
                print(f"Last distance: {min_value}")

        if save_result:
            self.last_value = np.array(euler_angle)
        return euler_angle


def test_euler_optimizer():
    import time
    seq_len = 1000
    hybrid = False
    opt = EulerOptimizer(hybrid=hybrid)
    gt_euler_seq = np.array([0.01, 0.02, 0.03])[None,] * np.arange(seq_len)[:, None]

    pred_euler_seq = []
    analytical_euler_seq = []
    tic = time.time()
    for i in range(seq_len):
        rot_mat = transforms3d.euler.euler2mat(*gt_euler_seq[i], "rxyz")
        pred_euler = opt.optimize(rot_mat)
        pred_euler_seq.append(pred_euler)
        analytical_euler_seq.append(transforms3d.euler.mat2euler(rot_mat, "rxyz"))
        if np.linalg.norm(pred_euler_seq[i] - gt_euler_seq[i]) > 0.01:
            print(f"Diff: {pred_euler_seq[i] - gt_euler_seq[i]}, original: {gt_euler_seq[i]}")
    print(f"{seq_len} iterations takes {time.time() - tic}s")

    pred_euler_seq = np.stack(pred_euler_seq)
    analytical_euler_seq = np.stack(analytical_euler_seq)

    print(f"Pred-gt diff: {np.linalg.norm(pred_euler_seq - gt_euler_seq, axis=1).mean()}")
    print(f"analytical-gt diff: {np.linalg.norm(analytical_euler_seq - gt_euler_seq, axis=1).mean()}")


if __name__ == '__main__':
    test_euler_optimizer()
