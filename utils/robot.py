import numpy as np
from numpy import linalg as LA
from tqdm import tqdm
import roboticstoolbox as rtb
from spatialmath import SE3


class Robot:
    def __init__(self, verbose, backend="swift"):
        self.verbose = verbose
        self.robot = rtb.models.DH.Panda()
        backends = ["swift", "pyplot"]
        if backend not in backends:
            raise NotImplementedError()
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
        if len(ee_pos) == 3:
            T = SE3(*ee_pos)
        else:
            T = ee_pos

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

    def path_generate_via_stable_joint_traj(self):
        rand = np.random.rand(2, self.dof) * 0.4
        q_samples = (self.joint_max - self.joint_min) * rand + self.joint_min

        qs = self.jtraj(q=q_samples[0], p=q_samples[1])

        ee = np.zeros((len(qs), 3))
        step = 0
        tq = tqdm(qs)
        for q in tq:
            ee[step] = self.forward_kinematics(q)
            step += 1

        return ee, qs

    def dist_fk(self, q: np.ndarray, ee_pos: np.ndarray):
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

    def jacobian(self, q: np.ndarray):
        '''
        Compute the jacobain based on joint configuration q.

        Parameters
        ----------
        q : np.ndarray
            joint configuration q.

        Returns J
        -------
        The manipulator Jacobian in the end-effector frame

        Return type
            ndarray(6,n)
        '''
        return self.robot.jacobe(q)

    def q_dot(self, q: np.ndarray, p: np.ndarray):
        '''
        Compute the q_dot (joint velocity) starts from q and ends to p.

        Parameters
        ----------
        q : np.ndarray
            start joint configuration
        p : np.ndarray
            end joint configuration

        Returns q_dot
        -------
        the joint velocity of 10/200 timestamp in the given trajectory.

        Return type
            np.ndarray
        '''
        qt = rtb.tools.trajectory.jtraj(q, p, t=100)

        return qt.qd[10]

    def jtraj_batch(self, qs: np.ndarray, ps: np.ndarray, t: int = 100, p_ratio: float = 0.3):
        q_batch = []
        for q, p in zip(qs, ps):
            q_m = self.jtraj(q, p, t, p_ratio)
            q_batch.append(q_m)
        return np.array(q_batch)

    def jtraj(self, q: np.ndarray, p: np.ndarray, t: int = 100):
        '''
        Return joint configuration between real joint configuration q and generated joint
        configuration p.

        Parameters
        ----------
        q : np.ndarray
            real joint configuration
        p : np.ndarray
            generated joint configuration
        t : int, optional
            total time slice, by default 100

        Returns 
        -------
        np.ndarray
            joint configuration
        '''
        qt = rtb.tools.trajectory.jtraj(q, p, t=100)
        return qt.q

    def x_dot_norm(self, q: np.ndarray, p: np.ndarray):
        '''
        Compute L1_norm(x_dot) where x_dot = jacobian(q) * diff(p, q).

        Parameters
        ----------
        q : np.ndarray
            joint configuration q as base configuration.
        p : np.ndarray
            joint configuration p as "to" configuration.

        Returns Norm
        -------
        L1_norm(x_dot)

        Return type
            int
        '''
        x_dot = self.jacobian(q) * self.q_dot(q, p)
        x_dot = x_dot[:3]
        return LA.norm(x_dot, ord=1)

    def traj_err(self, q: np.ndarray, p: np.ndarray, t: int, threshold: float):
        '''
        Compute the distance error of the joint trajectory from q to p.

        Parameters
        ----------
        q : np.ndarray
            joint configuration q as base configuration.
        p : np.ndarray
            joint configuration p as "to" configuration.
        t : int
            number of steps

        Returns
        -------
        Total distance between fkine(q) and fkine(qi) where qi is the joint snapshots of 
        the trajectory.

        Return type
            float
        '''
        qt = rtb.tools.trajectory.jtraj(q, p, t=t)

        target = self.forward_kinematics(q)

        for qi in qt.q:
            if self.dist_fk(qi, target) > threshold:
                return False

        return True

    def plot(self, q: np.ndarray, p: np.ndarray, movie='move.gif', t: int = 30, backend='swift'):
        qt = rtb.tools.trajectory.jtraj(q, p, t=t)
        if backend == 'swift':
            self.robot.plot(qt.q, dt=1/t, backend=backend)
        else:
            self.robot.plot(qt.q, dt=1/t, backend=backend,
                            movie=movie, jointaxes=False)

    def plot_qs(self, qs, movie='move.gif', dt: float = 0.05, mean=None, var=None):
        qs = np.array(qs)
        if mean is not None and var is not None:
            mean = np.array(mean)
            var = np.array(var)

            qs = qs * var + mean

        if self.backend == 'swift':
            self.robot.plot(qs, dt=dt, backend=self.backend)
        else:
            self.robot.plot(qs, dt=dt, backend=self.backend,
                            movie=movie, jointaxes=False)
