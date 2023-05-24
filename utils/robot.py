import numpy as np
from utils.settings import config
from numpy import linalg as LA
from tqdm import tqdm
import roboticstoolbox as rtb
from spatialmath import SE3


class Robot:
    def __init__(self, verbose, robot_name: str=config.robot_name, backend: str="swift"):
        self.verbose = verbose
        robots = ["panda"]
        if robot_name not in robots:
            raise NotImplementedError()
            
        backends = ["swift", "pyplot"]
        if backend not in backends:
            raise NotImplementedError()
        
        self.robot = rtb.models.DH.Panda()
        self.backend = backend
        if backend == "swift":
            self.robot = rtb.models.Panda()
        self.joint_min, self.joint_max = self.robot.qlim
        self.dof = len(self.joint_min)

        if self.verbose:
            print(self.robot)

    def forward_kinematics(self, q: np.ndarray):
        '''
        Given a joint configuration q to get end effector position ee_pos.
        Parameters
        ----------
        q : np.ndarray
            joint configuration.
        Returns
        -------
        ee_pos : np.ndarray
            end effector postion given a joint configuration
        '''
        T = self.robot.fkine(q)  # forward kinematics
        # output is a SE3 matrix, use t to get position.
        ee_pos = T.t
        if self.verbose:
            print(f"Given {q}, via fkine, get {ee_pos}")
        return np.array(ee_pos)

    def inverse_kinematics(self, ee_pos: np.ndarray, q0: np.ndarray = None):
        '''
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
        '''
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

    def random_sample_joint_config(self, num_samples: int, return_ee: bool = False) -> np.ndarray:
        # samples = np.array([])

        rand = np.random.rand(num_samples, self.dof)
        q_samples = (self.joint_max - self.joint_min) * rand + self.joint_min
        

        if return_ee:
            ee_samples = np.zeros((len(q_samples), 3))
            tq = tqdm(q_samples)
            step = 0
            for q in tq:
                ee_samples[step] = self.forward_kinematics(q=q)
                step += 1
            ee_samples = ee_samples.reshape((-1, 3))

            return q_samples, ee_samples

        return q_samples

    def path_generate_via_stable_joint_traj(self, dist_ratio: float = 0.4, t: int = 10):
        rand = np.random.rand(2, self.dof) * dist_ratio
        q_samples = (self.joint_max - self.joint_min) * rand + self.joint_min

        qs = rtb.tools.trajectory.jtraj(q_samples[0], q_samples[1], t=t)
        qs = qs.q

        ee = np.zeros((len(qs), 3))
        step = 0
        tq = tqdm(qs)
        for q in tq:
            ee[step] = self.forward_kinematics(q)
            step += 1

        return ee, qs

    def l2_err_func(self, q: np.ndarray, ee_pos: np.ndarray):
        '''
        Compute the Euclidean Distance between the value of forward_kinematics with joint_confg
        and ee_pos.

        Returns
        -------
        Distance : np.float
            the distance between forward(q) and ee_pos.
        '''
        com_pos = self.forward_kinematics(q)
        diff = np.linalg.norm(com_pos - ee_pos)
        if self.verbose:
            print(f"Com_pos: {com_pos}, EE_pos: {ee_pos}, diff: {diff}")
        return diff

    def plot(self, q: np.ndarray=None, p: np.ndarray=None, qs: np.ndarray=None, dt: float = 0.05):
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
            
            qt = rtb.tools.trajectory.jtraj(q, p, t=t)
            qs = qt.q
        self.robot.plot(qs, dt=dt)
