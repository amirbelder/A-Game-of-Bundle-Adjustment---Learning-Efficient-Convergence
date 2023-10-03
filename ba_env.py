# Core Library
import random
import time
from typing import Any, Dict, List, Tuple

# Third party
import gym
from gym import spaces
from scipy.spatial.transform import Rotation as R

from optimizer import *

import numpy as np
import numpy.linalg

def get_chance(x: float) -> float:
    """Get probability that a banana will be sold at price x."""
    e = math.exp(1)
    return (1.0 + e) / (1.0 + math.exp(x + 1))


def _cost(feature_vec, point, camera_quat, camera_tr):
    return feature_vec.cross((Quat(camera_quat)(point) + camera_tr).normalized())


def get_camera_real_world_cord_via_quat_inverse(camera_quat, camera_tr):
    """
    In order to extract x,y,z coordinates of the camera in the world we need to use the inverse quat.
    Where quat reprents the Rotation portion of the camera's 6DOF and the tr and translation portion.
    The formula to get the v (which is the x,y,z), where q is quat and t is translation:
    v = q^-1 (w - t) = q^-1 * w - (q^-1 * t)
    As w is (0,0,0) in the camera's cartesian coordinates.
    v = q^-1 (w - t) = - (q^-1 * t)

    So we use the scipy Rotation to invert the q portion via scipy.
    Steps:
    1. Get rotation matrix from quat (q) via scipy
    2. Inverse the rotation matrix we received in 1
    3. Go from the inverted rotation matrix back into a quat form that will be q^-1
    4. Calc and return: v = - (q^-1 * t)
    """

    rot_matrix = R.from_quat(camera_quat.data()).as_matrix()
    inv_rot_matrix = np.linalg.inv(rot_matrix)
    inv_quat = R.from_matrix(inv_rot_matrix).as_quat()
    inv_quat = Quat(inv_quat)
    # v = - 1.0 * (inv_quat * camera_tr)
    v = -1.0 * (inv_quat(camera_tr))
    return v


def calc_relative_error(P, Q):
    """
    Claculates the relative error between 2 point clouds.
    Also returns: rotation, translation and scale between them.
    """
    # Rigidly (+scale) aligns two point clouds with know point-to-point correspondences
    # with least-squares error.
    # Returns (scale factor c, rotation matrix R, translation vector t) such that
    #   Q = P*cR + t
    # if they align perfectly, or such that
    #   SUM over point i ( | P_i*cR + t - Q_i |^2 )
    # is minimised if they don't align perfectly.
    P = np.asarray(P)
    Q = np.asarray(Q)
    common_len = min(P.shape[0], Q.shape[0])
    P = P[:common_len, :]
    Q = Q[:common_len, :]
    n, dim = P.shape

    centeredP = P - P.mean(axis=0)
    centeredQ = Q - Q.mean(axis=0)

    common_length = min(centeredP.shape[0], centeredQ.shape[0])
    centeredP = centeredP[:common_length, :]
    centeredQ = centeredQ[:common_length, :]
    C = np.dot(np.transpose(centeredP), centeredQ) / n

    V, S, W = np.linalg.svd(C)
    d = (np.linalg.det(V) * np.linalg.det(W)) < 0.0

    if d:
        S[-1] = -S[-1]
        V[:, -1] = -V[:, -1]

    R = np.dot(V, W)

    varP = np.var(P, axis=0).sum()
    c = 1 / varP * np.sum(S)  # scale factor

    t = Q.mean(axis=0) - P.mean(axis=0).dot(c * R)
    err = ((P.dot(c * R) + t - Q) ** 2).sum()
    return c, R, t, err


class BaEnv(gym.Env):
    """
    The environment defines which actions can be taken at which point and
    when the agent receives which reward.
    """

    def __init__(self, json_path=None) -> None:
        self.__version__ = "0.1.0"

        # General variables defining the environment - actions, states and done
        self.done = False

        # Action space - Lambda can be between 0.0 and 1.0
        self.action_space = spaces.Box(
            low=np.array([0.0]), high=np.array([np.inf]), dtype=np.float32
        )

        # State space - will represent the last error, between 0 (best case) and 1000
        self.MAX_ERROR = 1000.0
        low = np.array([0.0])  # remaining_tries
        high = np.array([self.MAX_ERROR])  # remaining_tries
        # TODO: change into an list of states
        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        # Adding a big positive value to the reward of the final stage
        self.finish_reward = 10

        self.iteration = 0
        self.last_problematic_iteration = 0
        self.last_improvement = 0

        self.mult_damping = None 
        json_path = json_path 
        self.optimizer = self._init_optimizer(json_path=json_path)
        self.print_flag = False
        self.calc_hessian_thrs = np.inf
        self.use_cameras_error_only = False
        self.mean_camera_error_thrs = 0.07
        self.points_and_cameras_error_thrs = 1e-10
        self.max_iter_num = 100
        return

    def _init_optimizer(self, iter=0, json_path=None):
        # Optimizer initialization
        optimizer = Optimizer()
        seed = int(np.random.rand(1) * 10) + iter
        np.random.seed(seed=seed)
        num_points = 10
        num_cameras = 10
        print("num_points ", num_points, " num cameras ", num_cameras)
        points = [
            Vec(v) for v in (np.random.rand(num_points, 3) * 5.0 + [-2.5, -2.5, 1.0])
        ]
        cameras = [
            (Quat(1.0, 0.0, 0.0, 0.0), Vec(x, 0.0, 0.0))
            for x in np.linspace(-2, 2, num_cameras)
        ]

        # Cloud points
       
        noisy_points = [p + (np.random.normal(size=3) - 0.5) * 0.01 for p in points]
        noisy_cameras = [
            (
                Quat(q.coeffs() + (np.random.rand(4) - 0.5) * 0.01, normalize=True),
                t + (np.random.normal(size=3) - 0.5) * 0.05,
            )
            for q, t in cameras
        ]

        if json_path is None:
            for j, (nq, nt) in enumerate(noisy_cameras):
                v_j = get_camera_real_world_cord_via_quat_inverse(nq, nt)
                optimizer.initial_cameras_locations.append(v_j)
                for i, npt in enumerate(noisy_points):
                    feat_vec = (cameras[j][0](points[i]) + cameras[j][1]).normalized()
                    optimizer.add_elim_cost(partial(_cost, feat_vec), npt, nq, nt)

        elif str(json_path).__contains__('bal'):
            for j, (nq, nt) in enumerate(noisy_cameras):
                v_j = get_camera_real_world_cord_via_quat_inverse(nq, nt)
                optimizer.initial_cameras_locations.append(v_j)
                for i, npt in enumerate(noisy_points):
                    if cameras_to_points_dict[j].__contains__(i):
                        feat_vec = (cameras[j][0](points[i]) + cameras[j][1]).normalized()
                        optimizer.add_elim_cost(partial(_cost, feat_vec), npt, nq, nt)


        optimizer.setup()
        return optimizer

    def reset(self, iter=0, json_path=None):
        """
        Reset the state of the environment and returns an initial observation.

        Returns
        -------
        observation: List[int]
            The initial observation of the space.
        """
        self.done = False

        self.iteration = 0
        self.last_problematic_iteration = 0
        self.last_improvement = 0

        json_path = json_path
        self.optimizer = self._init_optimizer(iter=iter, json_path=json_path)
        self.calc_hessian_thrs = np.inf

        return [-10e10]  # initial state

    def calc_mean_abs_error(self):
        total_err = 0
        num_cameras = len(self.optimizer.initial_cameras_locations)
        projected_cameras_data = list(self.optimizer.survs.blocks.values())
        for i in range(num_cameras):
            try:
                if len(projected_cameras_data) > i + 1:
                    rot_proj_i, tr_proj_i = (
                        projected_cameras_data[i * 2].arg,  # shouldn't fail!
                        projected_cameras_data[i * 2 + 1].arg,
                    )
                else:
                    continue
            except Exception:
                continue
            proj_cam_location = get_camera_real_world_cord_via_quat_inverse(
                camera_quat=rot_proj_i, camera_tr=tr_proj_i
            )
            
            local_err = np.asarray(
                proj_cam_location.values - self.optimizer.initial_cameras_locations[i]
            )
            local_err = np.sum(np.abs(local_err))
            total_err += local_err
        mean_error = total_err / num_cameras
        return mean_error

    def get_relative_error_between_cameras_locations(self):
        orig_cam_locations = self.optimizer.initial_cameras_locations
        predicated_cameras_location = []
        num_cameras = len(self.optimizer.initial_cameras_locations)
        projected_cameras_data = list(self.optimizer.survs.blocks.values())
        for i in range(num_cameras):
            try:
                if len(projected_cameras_data) > i + 1:
                    rot_proj_i, tr_proj_i = (
                        projected_cameras_data[i * 2].arg,
                        projected_cameras_data[i * 2 + 1].arg,
                    )

                else:
                    continue
            except Exception:
                continue
            proj_cam_location = get_camera_real_world_cord_via_quat_inverse(
                camera_quat=rot_proj_i, camera_tr=tr_proj_i
            )
            predicated_cameras_location.append(proj_cam_location)


        r, t, scale, err = calc_relative_error(
            predicated_cameras_location, orig_cam_locations
        )
        return err

    def step(self, action: np.ndarray) -> Tuple[float, float, bool, Dict[Any, Any]]:
        """
        The agent takes a step in the environment.

        Parameters
        ----------
        action : float - this is a number in the range of [0,1] which represents the lambda value of the BA.

        Returns
        -------
        ob, reward, episode_over, info : tuple
            ob : float
                an environment-specific object representing your observation of
                the environment.
            reward : float
                amount of reward achieved by the previous action. The scale
                varies between environments, but the goal is always to increase
                your total reward.
            episode_over : bool
                whether it's time to reset the environment again. Most (but not
                all) tasks are divided up into well-defined episodes, and done
                being True indicates the episode has terminated. (For example,
                perhaps the pole tipped too far, or you lost your last life.)
            info : Dict
                 diagnostic information useful for debugging. It can sometimes
                 be useful for learning (for example, it might contain the raw
                 probabilities behind the environment's last state change).
                 However, official evaluations of your agent are not allowed to
                 use this for learning.
        """
        # Would change only if the stopping condition is met
        start = time.time()
        done = False
        self.mult_damping = action[0]

        # making sure that Lambda is within the range of [0,1]
        if self.mult_damping < 0:
            self.mult_damping = 1e-15
        
        calc_hessian_flag = True
        if self.mult_damping > self.calc_hessian_thrs:
            calc_hessian_flag = False

        cost = self.optimizer.compute_gradient_hessian(
            calc_hessian_flag=calc_hessian_flag
        )
        if calc_hessian_flag is True:
            # apply damping to D, compute invD inverting one block at a time
            for arg_info in self.optimizer.elims.blocks.values():
                s = arg_info.param_slice
                D_block = self.optimizer.D[s, s]
                D_damped_block = D_block.todense() + np.diag(
                    D_block.diagonal() * self.mult_damping
                )
                self.optimizer.D[s, s] = D_damped_block
                self.optimizer.invD[s, s] = np.linalg.inv(D_damped_block)

            # apply damping to A
            self.optimizer.A.setdiag(
                self.optimizer.A.diagonal() * (1 + self.mult_damping)
            )

            # compute Schur complement
            S = self.optimizer.A - (
                self.optimizer.B @ (self.optimizer.invD @ self.optimizer.B.T)
            )

            # compute error for reduced linear system
            reducedErrorX = self.optimizer.errorX - self.optimizer.B @ (
                self.optimizer.invD @ self.optimizer.errorP
            )
            self.optimizer.deltaX = sparse_solve(S, reducedErrorX)
            self.optimizer.deltaP = self.optimizer.invD @ (
                self.optimizer.errorP - self.optimizer.B.T @ self.optimizer.deltaX
            )

        else:  # The hessian was not calculated
            self.optimizer.deltaX = self.optimizer.deltaX / self.mult_damping
            self.optimizer.deltaP = self.optimizer.deltaP / self.mult_damping

        # expected cost improvement, according to the quadratic model
        expected_cost_improvement = 0.5 * (
            np.dot(self.optimizer.deltaX, self.optimizer.errorX)
            + np.dot(self.optimizer.deltaP, self.optimizer.errorP)
        )

        for b in self.optimizer.survs.blocks.values():
            b.backup()
            apply_parametrized_step(b.arg, -self.optimizer.deltaX[b.param_slice])

        for b in self.optimizer.elims.blocks.values():
            b.backup()
            apply_parametrized_step(b.arg, -self.optimizer.deltaP[b.param_slice])

        new_cost = self.optimizer.compute_gradient()
        new_expected_cost_improvement = 0.5 * (
            np.dot(self.optimizer.deltaX, self.optimizer.errorX)
            + np.dot(self.optimizer.deltaP, self.optimizer.errorP)
        )

        cost_improvement = cost - new_cost
        improvement_ratio = cost_improvement / expected_cost_improvement
        if new_cost > cost:  # cost got worse? that's BAD!
            mood = Optimizer.MOODS["crying"]
            self.last_problematic_iteration = self.iteration

            # restore previous state
            for b in self.optimizer.survs.blocks.values():
                b.restore()

            for b in self.optimizer.elims.blocks.values():
                b.restore()

        else:  # cost improved? check how well it matches with quadratic model

            # at least some degree of agreement with quadratic model
            if improvement_ratio > 0.4:
                mood = Optimizer.MOODS["excited"]
            else:
                mood = Optimizer.MOODS["worried"]

        # print status
        rel_change_percentage = (new_cost - cost) * 100.0 / cost

        if self.print_flag is True:
            print(
                f"{mood} [{self.iteration}]"
                + f" {cost:.03e} -> {new_cost:.03e} ({rel_change_percentage:.02f}%)"
                + f"  q-rel-\u03b4: {improvement_ratio:.02f}"
                + f"  \u03bb: {self.mult_damping:.02e}"
            )

        # check stopping condition (they are all sloppy...)
        if new_cost < cost * 0.999:
            self.last_improvement = self.iteration

        # Decide what's the relevant stopping condition
        cameras_projection_mean_abs_error = self.calc_mean_abs_error()
        rot_tr_scale_error = self.get_relative_error_between_cameras_locations()
        cameras_projection_mean_abs_error = rot_tr_scale_error
        if self.use_cameras_error_only is True:
            stopping_thrs, stopping_error = (
                self.mean_camera_error_thrs,
                cameras_projection_mean_abs_error,
            )
        else:
            stopping_thrs, stopping_error = self.points_and_cameras_error_thrs, new_cost

        end = time.time()
        reward = (start - end) * 1.0
        if self.iteration > 2 and (
            stopping_error < stopping_thrs or self.iteration > self.max_iter_num
        ):
            mood = Optimizer.MOODS["shrug"]
            if self.print_flag is True:
                print(f"{mood} converged or no improvement")
            done = True

            if stopping_error < stopping_thrs:
                reward += self.finish_reward

        next_state = expected_cost_improvement
        self.iteration += 1
        return_dict = {
            "iter_num": self.iteration,
            "lambda": self.mult_damping,
            "total_error": new_cost,
            "cam_error": cameras_projection_mean_abs_error,
        }
        if done is True:
            return_dict["convergence_reason"] = (
                "converged" if stopping_error < stopping_thrs else "no improvment"
            )
        return ([next_state], reward, done, return_dict)

    def enable_printing(self, enable=False) -> None:
        self.print_flag = enable

    def set_cameras_error_only(self, use_cameras_error_only=False) -> None:
        self.use_cameras_error_only = use_cameras_error_only

    def _render(self, mode: str = "human", close: bool = False) -> None:
        return None

    def seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed

    def while_step(self, action: float) -> Tuple[float, float, bool, Dict[Any, Any]]:
        """
        The agent takes a step in the environment.
        Runs the entire BA process without the action (which is not used) all at once with a while loop.
        For future comparison.

        Parameters
        ----------
        action : float - this is a number in the range of [0,1] which represents the lambda value of the BA.

        Returns
        -------
        ob, reward, episode_over, info : tuple
            ob : float
                an environment-specific object representing your observation of
                the environment.
            reward : float
                amount of reward achieved by the previous action. The scale
                varies between environments, but the goal is always to increase
                your total reward.
            episode_over : bool
                whether it's time to reset the environment again. Most (but not
                all) tasks are divided up into well-defined episodes, and done
                being True indicates the episode has terminated. (For example,
                perhaps the pole tipped too far, or you lost your last life.)
            info : Dict
                 diagnostic information useful for debugging. It can sometimes
                 be useful for learning (for example, it might contain the raw
                 probabilities behind the environment's last state change).
                 However, official evaluations of your agent are not allowed to
                 use this for learning.
        """
        # Would change only if the stopping condition is met
        start = time.time()
        done = False
        self.optimizer.mult_damping = 3e-2
        self.mult_damping = 3e-2
        stopping_error_list = []
        while done is False:
            cost = self.optimizer.compute_gradient_hessian()

            # apply damping to D, compute invD inverting one block at a time
            for arg_info in self.optimizer.elims.blocks.values():
                s = arg_info.param_slice
                D_block = self.optimizer.D[s, s]
                D_damped_block = D_block.todense() + np.diag(
                    D_block.diagonal() * self.optimizer.mult_damping
                )
                self.optimizer.D[s, s] = D_damped_block
                self.optimizer.invD[s, s] = np.linalg.inv(D_damped_block)

            # apply damping to A
            self.optimizer.A.setdiag(
                self.optimizer.A.diagonal() * (1 + self.optimizer.mult_damping)
            )

            # compute Schur complement
            S = self.optimizer.A - (
                self.optimizer.B @ (self.optimizer.invD @ self.optimizer.B.T)
            )

            # compute error for reduced linear system
            reducedErrorX = self.optimizer.errorX - self.optimizer.B @ (
                self.optimizer.invD @ self.optimizer.errorP
            )
            self.optimizer.deltaX = sparse_solve(S, reducedErrorX)
            self.optimizer.deltaP = self.optimizer.invD @ (
                self.optimizer.errorP - self.optimizer.B.T @ self.optimizer.deltaX
            )

            # expected cost improvement, according to the quadratic model
            expected_cost_improvement = 0.5 * (
                np.dot(self.optimizer.deltaX, self.optimizer.errorX)
                + np.dot(self.optimizer.deltaP, self.optimizer.errorP)
            )

            for b in self.optimizer.survs.blocks.values():
                b.backup()
                apply_parametrized_step(b.arg, -self.optimizer.deltaX[b.param_slice])

            for b in self.optimizer.elims.blocks.values():
                b.backup()
                apply_parametrized_step(b.arg, -self.optimizer.deltaP[b.param_slice])

            new_cost = self.optimizer.compute_gradient()
            new_expected_cost_improvement = 0.5 * (
                np.dot(self.optimizer.deltaX, self.optimizer.errorX)
                + np.dot(self.optimizer.deltaP, self.optimizer.errorP)
            )

            cost_improvement = cost - new_cost
            improvement_ratio = cost_improvement / expected_cost_improvement
            # print("new cost: ", new_cost, " cost: ", cost)
            if new_cost > cost:  # cost got worse? that's BAD!
                self.mult_damping = self.mult_damping * 2
                # self.add_damping = self.add_damping * 3
                mood = Optimizer.MOODS["crying"]
                self.last_problematic_iteration = self.iteration

                # restore previous state
                for b in self.optimizer.survs.blocks.values():
                    b.restore()

                for b in self.optimizer.elims.blocks.values():
                    b.restore()

            else:  # cost improved? check how well it matches with quadratic model

                # at least some degree of agreement with quadratic model
                if improvement_ratio > 0.4:
                    self.mult_damping = self.mult_damping * 0.5
                    mood = Optimizer.MOODS["excited"]
                else:
                    self.mult_damping = self.mult_damping * 2
                    mood = Optimizer.MOODS["worried"]

            # print status
            rel_change_percentage = (new_cost - cost) * 100.0 / cost
            print(
                f"{mood} [{self.iteration}]"
                + f" {cost:.03e} -> {new_cost:.03e} ({rel_change_percentage:.02f}%)"
                + f"  q-rel-\u03b4: {improvement_ratio:.02f}"
                + f"  \u03bb: {self.mult_damping:.02e}"
            )

            # check stopping condition (they are all sloppy...)
            if new_cost < cost * 0.999:
                self.last_improvement = self.iteration

            # Decide what's the relevant stopping condition
            cameras_projection_mean_abs_error = self.calc_mean_abs_error()
            rot_tr_scale_error = self.get_relative_error_between_cameras_locations()
            cameras_projection_mean_abs_error = rot_tr_scale_error
            if self.use_cameras_error_only is True:
                stopping_thrs, stopping_error = (
                    self.mean_camera_error_thrs,
                    cameras_projection_mean_abs_error,
                )
            else:
                stopping_thrs, stopping_error = (
                    self.points_and_cameras_error_thrs,
                    new_cost,
                )

            end = time.time()
            if self.iteration > 2 and (
                stopping_error < stopping_thrs or self.iteration > self.max_iter_num
            ):
                mood = Optimizer.MOODS["shrug"]
                if self.print_flag is True:
                    print(f"{mood} converged or no improvement")
                done = True
                reward = (start - end) * 1.0
                if stopping_error < stopping_thrs:
                    reward += self.finish_reward

            next_state = expected_cost_improvement
            self.iteration += 1
            stopping_error_list.append(stopping_error)

            return_dict = {"iter_num": self.iteration, "lambda": self.mult_damping, "error": stopping_error_list}
            if done is True:
                return_dict["convergence_reason"] = (
                    "converged" if stopping_error < stopping_thrs else "no improvment"
                )
                return return_dict
