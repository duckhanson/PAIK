from torch import Tensor
import torch
from typing import Optional
from attr import dataclass
from klampt import WorldModel, IKSolver
from pprint import pprint
import math
import numpy as np
from klampt.model import ik
from klampt.math import so3
import jrl.robots as jrlib


class Robot:
    def __init__(self, name, active_joint_idx):
        assert name in ["atlas", "baxter", "robonaut2"]
        self.name = name
        self._world = WorldModel()
        robot_dir_path = "/home/luca/Klampt-examples/data/robots"

        self._world.loadElement(
            f"{robot_dir_path}/{name}.rob"
        )  # pr2, atlas, baxter, robonaut2
        self._robot = self._world.robot(0)
        _ik_solver = IKSolver(self._robot)
        self._all_joint_limits = np.array(_ik_solver.getJointLimits()).T
        self._active_joint_idx = active_joint_idx
        self.set_active_dofs(active_joint_idx)
        self.n_dofs = len(active_joint_idx)
        self.ndof = self.n_dofs
        self._klampt_ee_link = self._robot.link(
            active_joint_idx[-1]
        )  # end effector link
        
    def clamp_to_joint_limits(self, x: np.ndarray) -> np.ndarray:
        """Not used in the current implementation"""
        return x

    def set_active_dofs(self, active_dofs):
        self.active_joint_names = [self._robot.link(ji).getName() for ji in active_dofs]

        self.active_joint_min = self._all_joint_limits[active_dofs, 0]
        self.active_joint_max = self._all_joint_limits[active_dofs, 1]
        self.actuated_joints_limits = self._all_joint_limits[active_dofs, :]

    def get_active_joint_limits(self):
        return self.active_joint_min, self.active_joint_max

    def get_active_joint_names(self):
        return self.active_joint_names

    def get_all_joint_names(self):
        return [self._robot.link(ji).getName() for ji in range(self._robot.numLinks())]

    def get_all_joint_limits(self):
        return self._all_joint_limits
    
    def set_klampt_robot_config(self, x: np.ndarray):
        """Set the internal klampt robots config with the given joint angle vector"""
        assert x.shape == (self.n_dofs,), f"Expected x to be of shape ({self.n_dofs},) but got {x.shape}"
        q = self._x_to_qs(np.resize(x, (1, self.n_dofs)))[0]
        self._robot.setConfig(q)

    def _get_klampt_active_driver_idxs(self):
        """We need to know which indexes of the klampt driver vector are from user specified active joints.

        Returns:
            List[int]: The indexes of the klampt driver vector which correspond to the(user specified) active joints
        """

        # Get the names of all the child links for each active joint.
        # Note: driver.getName() returns the joints' child link for some god awful reason. See L1161 in Robot.cpp
        # (https://github.com/krishauser/Klampt/blob/master/Cpp/Modeling/Robot.cpp#L1161)
        actuated_joint_child_names = self._get_actuated_joint_child_names()
        all_drivers = [
            self._klampt_robot.driver(i) for i in range(self._klampt_robot.numDrivers())
        ]
        driver_vec_tester = [
            1 if (driver.getName() in actuated_joint_child_names) else -1
            for driver in all_drivers
        ]
        active_driver_idxs = list(locate(driver_vec_tester, lambda x: x == 1))

        assert (
            len(active_driver_idxs) == self.ndof
        ), f"Error - the number of active drivers != ndof ({len(active_driver_idxs)} != {self.ndof})"
        return active_driver_idxs

    def _driver_vec_from_x(self, x: np.ndarray):
        """Format a joint angle vector into a klampt driver vector. Non user specified joints will have a value of 0.

        Args:
            x (np.ndarray): (self.ndof,) joint angle vector

        Returns:
            List[float]: A list with the joint angle vector formatted in the klampt driver format. Note that klampt
                            needs a list of floats when recieving a driver vector.
        """
        assert (
            x.size == self.n_dofs
        ), f"x doesn't have {self.n_dofs} (n_dofs) elements ({self.n_dofs} != {x.size})"
        assert x.shape == (
            self.n_dofs,
        ), f"x.shape must be (n_dofs,) - ({(self.n_dofs,)}) != {x.shape}"
        x = x.tolist()

        # return x as a list if there are no additional active joints in the urdf
        if len(x) == self._klampt_driver_vec_dim:
            return x

        # TODO(@jstm): Consider a non iterative implementation for this
        driver_vec = [0.0] * self._klampt_driver_vec_dim

        j = 0
        for i in self._active_joint_idx:
            driver_vec[i] = x[j]
            j += 1

        return driver_vec

    def get_n_drivers(self):
        return self._klampt_driver_vec_dim

    def _x_to_qs(self, x: np.ndarray):
        """Return a list of klampt configurations (qs) from an array of joint angles (x)

        Args:
            x: (n x ndof) array of joint angle settings

        Returns:
            A list of configurations representing the robots state in klampt
        """
        assert (
            x.ndim == 2 and x.shape[1] == self.ndof
        ), f"x.shape: {x.shape}, ndof: {self.ndof}"
        n = x.shape[0]
        qs = []
        for i in range(n):
            driver_vec = self._driver_vec_from_x(x[i])
            qs.append(self._klampt_robot.configFromDrivers(driver_vec))
        return qs

    def _Q_to_Qvec(self, Q: np.ndarray):
        assert (
            Q.ndim == 2 and Q.shape[1] == self.ndof
        ), f"Q.shape: {Q.shape}, ndof: {self.ndof}"
        Q_vec = np.zeros((Q.shape[0], len(self._robot.getConfig())))
        Q_vec[:, self._active_joint_idx] = Q
        return Q_vec

    def _Q_from_Qvec(self, Qvec: np.ndarray) -> np.ndarray:
        assert Qvec.ndim == 2 and Qvec.shape[1] == len(
            self._robot.getConfig()
        ), f"Qvec.shape: {Qvec.shape}, robot_config: {len(self._robot.getConfig())}"
        Q = Qvec[:, self._active_joint_idx]
        return Q

    def forward_kinematics(self, Q: np.ndarray, solver=None) -> np.ndarray:
        """Forward kinematics using the klampt library"""
        if isinstance(Q, Tensor):
            Q_numpy = Q.detach().cpu().numpy()
            _device = Q.device
        else:
            Q_numpy = Q
            _device = "cpu"
        P = np.zeros((len(Q_numpy), 7))  # x, y, z, qx, qy, qz, qw
        Qvec = self._Q_to_Qvec(Q_numpy)
        for i, q in enumerate(Qvec):
            self._robot.setConfig(q)
            R, t = self._klampt_ee_link.getTransform()  # type ignore
            P[i, 0:3] = np.array(t)
            P[i, 3:] = np.array(so3.quaternion(R))
        
        # turn into torch tensor
        return torch.from_numpy(P).float().to(_device)
    
    def sample_joint_angles(self, n: int):
        """Randomly sample n joint angles within the joint limits"""
        Q = np.zeros((n, self.ndof))
        for j in range(self.ndof):
            Q[:, j] = np.random.uniform(
                self.active_joint_min[j], self.active_joint_max[j], n
            )

        return Q
    
    def forward_kinematics_klampt(self, Q: np.ndarray, solver=None) -> np.ndarray:
        """Forward kinematics using the klampt library"""
        return self.forward_kinematics(Q)

    def sample_joint_angles_and_poses(self, n: int):
        """Randomly sample n joint angles within the joint limits"""
        Q = self.sample_joint_angles(n)
        P = self.forward_kinematics(Q)
        return Q, P

    def inverse_kinematics_klampt(
        self, p: np.ndarray, num_trails: int = 50, max_iterations: int = 150, positional_tolerance: float = 1e-3, seed: Optional[np.ndarray] = None, return_iterations: bool = False
    ) -> Optional[np.ndarray]:
        """Inverse kinematics using the klampt library"""
        assert p.ndim == 1 and p.shape[0] == 7, f"p.shape: {p.shape}"
        R = so3.from_quaternion(p[3:])  # type: ignore
        
        obj = ik.objective(self._klampt_ee_link, t=p[0:3].tolist(), R=R)

        for _ in range(num_trails):
            solver = IKSolver(self._robot)
            solver.add(obj)
            solver.setActiveDofs(self._active_joint_idx)
            solver.setMaxIters(max_iterations)
            solver.setTolerance(positional_tolerance)
            if seed is None:
                solver.sampleInitial()
            else:
                self._robot.setConfig(self._Q_to_Qvec(np.asarray([seed]))[0])
            
            res = solver.solve()

            if res:
                q = self._Q_from_Qvec(np.asarray([self._robot.getConfig()]))[0]
                if return_iterations:
                    return q, solver.lastSolveIters() # type: ignore
                
                return q

        # print(f"[INFO] Failed to find a solution for p: {p}")
        return None

    def __repr__(self):
        # print robot name, ndof, and active joint names, and joint limits
        return f"Robot: {self.name}, ndof: {self.ndof}, active joints: {self.active_joint_names}, joint limits: {self.get_active_joint_limits()}"


class AtlasArm(Robot):
    def __init__(self):
        super().__init__("atlas", [9, 10, 11, 12, 13, 14])
        self.name = "atlas_arm"


class AtlasWaistArm(Robot):
    def __init__(self):
        super().__init__("atlas", [6, 7, 8, 9, 10, 11, 12, 13, 14])


class BaxterArm(Robot):
    def __init__(self):
        super().__init__("baxter", [15, 16, 17, 18, 19, 21, 22])

def get_robot(name: str):
    try:
        return jrlib.get_robot(name)
    except ValueError:
        if name == "atlas_arm":
            return AtlasArm()
        elif name == "atlas_waist_arm":
            return AtlasWaistArm()
        elif name == "baxter_arm":
            return BaxterArm()
        else:
            raise ValueError(f"Unknown robot name: {name}")

class PR2(Robot):
    def __init__(self):
        # arm from 51 to 61, palm 63
        super().__init__("pr2", [51, 52, 53, 55, 56, 60, 61, 63])


if __name__ == "__main__":
    # Test the Robot class
    robot = AtlasWaistArm()
 
    