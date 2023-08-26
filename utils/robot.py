import numpy as np
import roboticstoolbox as rtb
import swift

# from numpy import linalg as LA
from spatialmath import SE3
from spatialmath.base import r2q
from pyquaternion import Quaternion
from tqdm import tqdm
from utils.settings import config, ets_table
from utils.utils import create_robot_dirs, data_collection, denormalize


class Robot:
    def __init__(self, verbose, robot_name: str = config.robot_name):
        self.verbose = verbose
        support_robots = list(ets_table.keys())
        if robot_name not in support_robots:
            raise NotImplementedError()

        self.robot = self._get_robot_module(robot_name)

        self.joint_min, self.joint_max = self.robot.qlim

        if self.verbose:
            print(self.robot)

        create_robot_dirs()
        
        _, P_tr = data_collection(robot=self, N=config.N_train)
        self.pos_min, self.pos_max = P_tr.min(axis=0), P_tr.max(axis=0)
        

    def _get_robot_module(self, robot_name):
        if robot_name == "panda":
            robot = rtb.models.Panda()
        elif robot_name == "al5d":
            robot = rtb.models.AL5D()
        elif robot_name == "fetchcamera":
            robot = rtb.models.FetchCamera()
        elif robot_name == "frankie":
            robot = rtb.models.Frankie()
        elif robot_name == "frankieomni":
            robot = rtb.models.FrankieOmni()
        elif robot_name == "lbr":
            robot = rtb.models.LBR()
        elif robot_name == "mico":
            robot = rtb.models.Mico()
        elif robot_name == "puma":
            robot = rtb.models.Puma560()
        elif robot_name == "ur10":
            robot = rtb.models.UR10()
        elif robot_name == "valkyrie":
            robot = rtb.models.Valkyrie()
        elif robot_name == "yumi":
            robot = rtb.models.YuMi()
        elif robot_name == "fetch":
            robot = rtb.models.Fetch()
        else:
            raise NotImplementedError()
        return robot

    def forward_kinematics(self, q: np.ndarray):
        """
        Given a joint configuration q to get end effector position ee_pos.
        Parameters
        ----------
        q : np.ndarray
            joint configuration.
        Returns
        -------
        ee_pos : np.ndarray
            end effector postion given a joint configuration
        """
        T = self.robot.fkine(q)  # forward kinematics
        # output is a SE3 matrix, use t to get position.
        ee_pos = T.t
        if self.verbose:
            print(f"Given {q}, via fkine, get {ee_pos}")
        return np.array(ee_pos)
    
    def forward_kinematics_quaternion(self, q: np.ndarray):
        """
        _summary_

        Given a joint configuration q to get end effector position and quaternion.
        ----------
        q : np.ndarray
            joint configuration.

        Returns
        -------
        (t, r) : np.ndarray(7)
            position and quaternion (x, y, z, qw, qx, qy, qz)
        """
        T = self.robot.fkine(q)  # forward kinematics
        # output is a SE3 matrix, use t to get position.
        r = r2q(T.R)
        t = T.t
        if self.verbose:
            print(f"Given {q}, via fkine, get {t}")
        return np.concatenate((t, r))

    def inverse_kinematics(self, ee_pos: np.ndarray, q0: np.ndarray = None):
        """
        Given end effector position ee_pos to get a joint configuration q. Can use q0 to
        get a solid solution.

        Parameters
        ----------
        ee_pos : np.ndarray
            end effector position
        q0 : np.ndarray, optional
            a possible joint configuration, by default None

        Returns
        -------
        q : np.ndarray
            a joint configuartion solved from analytic inverse kinematics.
        """
        if ee_pos.shape == (4, 4):
            T = ee_pos
        else:
            T = SE3(*ee_pos)

        # sol = self.robot.ikine_min(T, q0=q0)
        # ikine_XX
        # sol = self.robot.ikine_LM(T, q0=q0, ilimit=500, L=0.01)  # 0.045
        # sol = self.robot.ikine_LM(T) # 0.045
        # sol = self.robot.ikine_mmc(T, q0=q0) # not work
        # sol = self.robot.ikine_min(T, q0=q0) # not work
        # sol = self.robot.ikine_LMS(T, q0=q0) # 0.011
        # the fastest and the most accuracy.
        sol = self.robot.ikine_LMS(T, q0=q0, wN=1e-7)  # 1.4106036252062574e-06
        # get solution
        q, success = sol[0], sol[1]
        if self.verbose:
            print(f"Given {ee_pos} and {q0}, via ik, get {q}.")
        return np.array(q), success

    def uniform_sample_J(self, num_samples: int) -> np.ndarray:
        assert config.m == 3 # pos (3)
        print(f"Start sample {num_samples} data.")
        rand = np.random.rand(num_samples, config.n)
        q_samples = (self.joint_max - self.joint_min) * rand + self.joint_min

        ee_samples = np.zeros((len(q_samples), config.m))
        tq = tqdm(q_samples)
        step = 0
        for q in tq:
            ee_samples[step] = self.forward_kinematics(q=q)
            step += 1
        ee_samples = ee_samples.reshape((-1, config.m))

        return q_samples
    
    def uniform_sample_J_quaternion(self, num_samples: int):
        assert config.m == 3 + 4 # pos (3) + qua (4)
        print(f"Start sample {num_samples} data.")
        rand = np.random.rand(num_samples, config.n)
        q_samples = (self.joint_max - self.joint_min) * rand + self.joint_min
        
        ee_samples = np.zeros((len(q_samples), config.m))
        tq = tqdm(q_samples)
        step = 0
        for q in tq:
            ee_samples[step] = self.forward_kinematics_quaternion(q=q)
            step += 1
        ee_samples = ee_samples.reshape((-1, config.m))

        return q_samples, ee_samples

    def path_generate_via_stable_joint_traj(self, dist_ratio: float = 0.4, t: int = 10):
        rand = np.random.rand(2, config.n) * dist_ratio
        q_samples = (self.joint_max - self.joint_min) * rand + self.joint_min

        qs = rtb.tools.trajectory.jtraj(q_samples[0], q_samples[1], t=t)
        qs = qs.q

        ee = np.zeros((len(qs), config.m))
        step = 0
        for q in qs:
            ee[step] = self.forward_kinematics(q)
            step += 1

        return ee, qs
    
    def path_generate_via_stable_joint_traj_quaternion(self, dist_ratio: float = 0.4, t: int = 10):
        rand = np.random.rand(2, config.n) * dist_ratio
        q_samples = (self.joint_max - self.joint_min) * rand + self.joint_min

        qs = rtb.tools.trajectory.jtraj(q_samples[0], q_samples[1], t=t)
        qs = qs.q

        ee = np.zeros((len(qs), 7))
        step = 0
        for q in qs:
            ee[step] = self.forward_kinematics_quaternion(q)
            step += 1

        return ee, qs

    def position_errors_Single_Input(self, q: np.ndarray, ee_pos: np.ndarray):
        """
        Compute the Euclidean Distance between the value of forward_kinematics with joint_confg
        and ee_pos.

        Returns
        -------
        Distance : np.float
            the distance between forward(q) and ee_pos.
        """
        com_pos = self.forward_kinematics(q)
        diff = np.linalg.norm(com_pos - ee_pos)
        if self.verbose:
            print(f"Com_pos: {com_pos}, EE_pos: {ee_pos}, diff: {diff}")
        return diff

    def position_errors_Arr_Inputs(self, qs: np.ndarray, ee_pos: np.ndarray):
        """
        array version of position_errors_Single_Input

        :param qs: _description_
        :type qs: np.ndarray
        :param ee_pos: _description_
        :type ee_pos: np.ndarray
        :return: _description_
        :rtype: _type_
        """
        if config.enable_normalize:
            J = denormalize(qs, self.joint_min, self.joint_max)
            P = denormalize(ee_pos, self.pos_min, self.pos_max)
        
        com_pos = np.zeros((len(J), P.shape[-1]))
        for i, q in enumerate(J):
            com_pos[i] = self.forward_kinematics(q)
        diff = np.linalg.norm(com_pos - P, axis=1)
        return diff
    
    
    def calculate_orientation_errors_Arr_Inputs(self, preds: np.ndarray, target: np.ndarray):
        """
        Handle array prediction inputs with the same task target (position + orientation)

        Parameters
        ----------
        preds : np.ndarray
            FK(preds)
        target : np.ndarray
            task point

        Returns
        -------
        float32
            mean of orientation errors
        """
        ori_err = np.zeros((len(preds)))
        # position, orientation of target
        ot = target[3:]
        ot = Quaternion(array=ot)
        
        # position, orientation of preds
        ops = preds[:, 3:]
        # orientation error
        for i, op in enumerate(ops):
            op = Quaternion(array=op)
            ori_err[i] = Quaternion.distance(op, ot)
        
        return ori_err.mean()        

    def position_orientation_errors_Arr_Inputs(self, qs: np.ndarray, ee_pos: np.ndarray):
        """
        array version of position_errors_Single_Input

        :param qs: _description_
        :type qs: np.ndarray
        :param ee_pos: _description_
        :type ee_pos: np.ndarray
        :return: _description_
        :rtype: _type_
        """
        if config.enable_normalize:
            J = denormalize(qs, self.joint_min, self.joint_max)
            P = denormalize(ee_pos, self.pos_min, self.pos_max)
        
        preds = np.zeros((len(J), P.shape[-1]))

        for i, q in enumerate(J):
            preds[i] = self.forward_kinematics_quaternion(q)
        
        pos_err = np.linalg.norm(preds[:, :3] - P[:3], axis=1)
        ori_err = self.calculate_orientation_errors_Arr_Inputs(preds=preds, target=P)
        return pos_err, ori_err

    def plot(
        self,
        q: np.ndarray = None,
        p: np.ndarray = None,
        qs: np.ndarray = None,
        dt: float = 0.1,
    ):
        """
        _summary_
        example of use

        Given init and final
        robot.plot(q=q, p=p, dt=0.01)

        Given jtraj
        robot.plot(qs=qs, dt=0.1)

        :param q: _description_, defaults to None
        :type q: np.ndarray, optional
        :param p: _description_, defaults to None
        :type p: np.ndarray, optional
        :param qs: _description_, defaults to None
        :type qs: np.ndarray, optional
        :param dt: _description_, defaults to 0.05
        :type dt: float, optional
        :raises ValueError: _description_
        """
        if qs is None:
            if q is None or p is None:
                raise ValueError("Input error: should give init and final or jtraj.")

            t = int(1 / dt)
            qt = rtb.tools.trajectory.jtraj(q, p, t=t)
            qs = qt.q

        env = swift.Swift()
        env.launch(realtime=True)
        env.add(self.robot)

        for q in qs:
            self.robot.q = q
            env.step(dt)
        # self.robot.plot(qs, dt=dt)
