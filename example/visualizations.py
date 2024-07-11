from typing import List, Callable, Any
from time import sleep
from dataclasses import dataclass

from jrl.robot import Robot
from jrl.robots import Panda, Fetch, FetchArm
from klampt.math import so3
from klampt.model import coordinates, trajectory
from klampt import IKSolver, vis, WorldModel, Geometry3D
import numpy as np
import torch
import torch.optim

# from ikflow.ikflow_solver import IKFlowSolver
# from ikflow.config import device, DEFAULT_TORCH_DTYPE
# from ikflow.evaluation_utils import solution_pose_errors
np.random.seed(47) # 37 stable, 

# Import required packages
import numpy as np
from tqdm import trange

from paik.solver import Solver
from paik.settings import (
    PANDA_NSF,
    PANDA_PAIK,
)

_OSCILLATE_LATENT_TARGET_POSES = {
    Panda.name: np.array([0.25, 0.35, 0.6, 0.5, -0.5,  0.5, 0.5]),
    Fetch.name: np.array([0.45, 0.65, 0.55, 1.0, 0.0, 0.0, 0.0]),
}

_TARGET_POSE_FUNCTIONS = {
    Panda.name: lambda counter: np.array(
        [0.4 * np.sin(counter / 50), 0.6, 0.75, 0.7071068, -0.7071068, 0.0, 0.0]
    ),
    Fetch.name: lambda counter: np.array(
        [0.25 * np.sin(counter / 50) + 0.5, 0.5, 0.75, 1.0, 0.0, 0.0, 0.0]
    ),
    FetchArm.name: lambda counter: np.array(
        [0.6, 0.15 * np.sin(counter / 50) + 0.5, 0.75, 1.0, 0.0, 0.0, 0.0]
    ),
}

PI = np.pi


def _plot_pose(name: str, pose: np.ndarray, hide_label: bool = False):
    vis.add(
        name,
        coordinates.Frame(name=name, worldCoordinates=(so3.from_quaternion(pose[3:]), pose[0:3])),  # type: ignore
        hide_label=hide_label,
    )


def _run_demo(
    robot: Robot,
    n_worlds: int,
    setup_fn: Callable[[List[WorldModel]], None],
    loop_fn: Callable[[List[WorldModel], Any], None],
    viz_update_fn: Callable[[List[WorldModel], Any], None],
    demo_state: Any = None,
    time_p_loop: float = 2.5,
    title: str = "Anonymous demo",
):
    """Internal function for running a demo."""
    worlds = [robot.klampt_world_model]
    if n_worlds > 1:
        for _ in range(n_worlds - 1):
            worlds.append(worlds[0].copy())

    vis.init()
    vis.add("coordinates", coordinates.manager())
    background_color = (1, 1, 1, 0.7)
    vis.setBackgroundColor(background_color[0], background_color[1], background_color[2], background_color[3])
    size = 5
    for x0 in range(-size, size + 1):
        for y0 in range(-size, size + 1):
            vis.add(
                f"floor_{x0}_{y0}",
                trajectory.Trajectory([1, 0], [(-size, y0, 0), (size, y0, 0)]),
                color=(0.75, 0.75, 0.75, 1.0),
                width=2.0,
                hide_label=True,
                pointSize=0,
            )
            vis.add(
                f"floor_{x0}_{y0}2",
                trajectory.Trajectory([1, 0], [(x0, -size, 0), (x0, size, 0)]),
                color=(0.75, 0.75, 0.75, 1.0),
                width=2.0,
                hide_label=True,
                pointSize=0,
            )
    setup_fn(worlds)

    vis.setWindowTitle(title)
    vis.show()
    while vis.shown():
        # Modify the world here. Do not modify the internal state of any visualization items outside of the lock
        vis.lock()

        loop_fn(worlds, demo_state)
        vis.unlock()

        # Outside of the lock you can use any vis.X functions, including vis.setItemConfig() to modify the state of objects
        viz_update_fn(worlds, demo_state)
        sleep(time_p_loop)
    vis.kill()


@dataclass
class Config:
    n_worlds: int = 2
    time_p_loop: float = 0.01
    time_dilation: float = 0.75

def oscillate_latent(ik_solver: Solver, show_mug: bool = False, change_F_per_cnt: int = 1000):
    """Fixed end pose, oscillate through the latent space"""
    config = Config()
    title = "Fixed end pose with oscillation through the latent space"
    robot = ik_solver.robot
    target_pose = _OSCILLATE_LATENT_TARGET_POSES[ik_solver.param.robot_name]  # type: ignore
    F_min, F_max = solver.F.min(), solver.F.max()
    
    if show_mug:
        mug_path = "/home/luca/paik/data/visualization_resources/objects/mug.obj"
        mug = Geometry3D()
        mug.loadFile(mug_path)
    

    def setup_fn(worlds):
        del worlds
        vis.add(f"robot", robot._klampt_robot)
        if show_mug:
            vis.add("mug",mug)

        # Axis
        vis.add("coordinates", coordinates.manager())
        _plot_pose("target_pose.", target_pose, hide_label=True)

        # Configure joint angle plot
        vis.addPlot("joint_vector")
        vis.setPlotDuration("joint_vector", 5)
        vis.setPlotRange("joint_vector", -PI, PI)

        # Configure joint angle plot
        vis.addPlot("latent_vector")
        vis.setPlotDuration("latent_vector", 5)
        vis.setPlotRange("latent_vector", -0.644, 0.644) # confidence interval of 99% for a normal distribution with std=0.25 and mean=0, 0.644

        # Configure locality plot
        vis.addPlot("locality_vector")
        vis.setPlotDuration("locality_vector", 5)
        vis.setPlotRange("locality_vector", F_min-0.05, F_max+0.05)
        

        # Add target pose errors plot
        vis.addPlot("solution_error")
        vis.addPlot("solution_error")
        vis.logPlot("solution_error", "l2 (mm)", 0)
        vis.logPlot("solution_error", "angular (deg)", 0)
        vis.setPlotDuration("solution_error", 5)
        vis.setPlotRange("solution_error", 0, 8)

    @dataclass
    class DemoState:
        counter: int
        last_joint_vector: np.ndarray
        last_latent: np.ndarray
        last_locality: np.ndarray
        ave_l2_error: float
        ave_angular_error: float
    
    def solve_latent(solver, P, F, latent):
        if len(P.shape) == 1:
            P = P.reshape(1, -1)
        
        J_hat = solver.generate_ik_solutions(P, F, num_sols=1, std=0.0, latent=latent)
        # (1, 1, solver.n)
        return J_hat

    def loop_fn(worlds, _demo_state):
        latent = np.zeros(ik_solver.n)
        for i in range(ik_solver.n):
            counter = config.time_dilation * _demo_state.counter
            offset = 2 * PI * i / ik_solver.n
            latent[i] = (
                0.25 * np.cos(counter / 25 + offset)
                - 0.25 * np.cos(counter / 100 + offset)
                + 0.25 * np.sin(counter / (100 + i * 100))
            ) - 0.1
        _demo_state.last_latent = latent

        # Get solutions to pose of random sample
        ik_solution = solve_latent(ik_solver, target_pose, _demo_state.last_locality, latent)
        l2_errors, ang_errors = ik_solver.evaluate_pose_error_J3d_P2d(ik_solution, target_pose.reshape(-1, solver.m), return_all=True)
        _demo_state.ave_l2_error = l2_errors.mean() * 1000
        _demo_state.ave_ang_error = np.rad2deg(ang_errors.mean())
        # qs = robot._x_to_qs(solutions)
        ik_solution = ik_solution.reshape(ik_solver.n)
        robot.set_klampt_robot_config(ik_solution)

        # Update _demo_state
        _demo_state.counter += 1
        
        if _demo_state.counter % change_F_per_cnt == 0:
            print(f"[INFO] counter={_demo_state.counter}")
            F = solver.select_reference_posture(target_pose, "knn", num_sols=50)
            F = F[np.random.randint(0, F.shape[0])]
            print(f"[INFO] chaning F={F}")
            _demo_state.last_locality = F
        _demo_state.last_joint_vector = ik_solution
        
    def viz_update_fn(worlds, _demo_state):
        del worlds
        if show_mug:
            R, t = robot._klampt_ee_link.getTransform() # type: ignore
            R, t = np.array(R), np.array(t)
            distance_to_move = 0.15
            t += R[-3:] * distance_to_move
            # rotate the mug around the z-axis of the end effector with 90 degrees
            R = so3.mul(R, so3.from_axis_angle(([0, 0, 1], PI / 2)))  # type: ignore
            # rotate the mug around the y-axis of the end effector with -120 degrees
            R = so3.mul(R, so3.from_axis_angle(([0, 1, 0], PI * -120 / 180)))  # type: ignore
            mug.setCurrentTransform(R, t)  # type: ignore
            
        for i in range(ik_solver.n):
            vis.logPlot(f"joint_vector", f"joint_{i}", _demo_state.last_joint_vector[i])
        for i in range(ik_solver.n):
            vis.logPlot(f"latent_vector", f"latent_{i}", _demo_state.last_latent[i])
        for i in range(ik_solver.r):
            vis.logPlot(f"locality_vector", f"locality_{i}", _demo_state.last_locality[i])
        vis.logPlot("solution_error", "l2 (mm)", _demo_state.ave_l2_error)
        vis.logPlot("solution_error", "angular (deg)", _demo_state.ave_ang_error)

    demo_state = DemoState(
        counter=0, last_joint_vector=np.zeros(ik_solver.n), last_latent=np.zeros(ik_solver.n), last_locality=solver.select_reference_posture(target_pose, "knn"), ave_l2_error=0, ave_angular_error=0
    )

    _run_demo(
        robot, config.n_worlds, setup_fn, loop_fn, viz_update_fn, demo_state=demo_state, time_p_loop=config.time_p_loop, title=title
    )
    
def oscillate_locality(ik_solver: Solver, show_mug: bool = False, from_nn: bool = False):
    """Fixed end pose, oscillate through the locality space"""
    config = Config()
    title = "Fixed end pose with oscillation through the locality space"
    robot = ik_solver.robot
    target_pose = _OSCILLATE_LATENT_TARGET_POSES[ik_solver.param.robot_name]  # type: ignore
    locality_nn = solver.select_reference_posture(target_pose, "knn", num_sols=100)
    locality_nn = np.sort(locality_nn, axis=0) # type: ignore
    F_min, F_max = solver.F.min(), solver.F.max()
    print(f"[INFO] F_min={F_min}, F_max={F_max}")
    # -0.277 < F < 0.263
        
    if show_mug:
        mug_path = "/home/luca/paik/data/visualization_resources/objects/mug.obj"
        mug = Geometry3D()
        mug.loadFile(mug_path)

    def setup_fn(worlds):
        del worlds
        vis.add(f"robot", robot._klampt_robot)
        if show_mug:
            vis.add("mug",mug)

        # Axis
        vis.add("coordinates", coordinates.manager())
        _plot_pose("target_pose.", target_pose, hide_label=True)

        # Configure joint angle plot
        vis.addPlot("joint_vector")
        vis.setPlotDuration("joint_vector", 5)
        vis.setPlotRange("joint_vector", -PI, PI)

        # Configure joint angle plot
        vis.addPlot("locality_vector")
        vis.setPlotDuration("locality_vector", 5)
        vis.setPlotRange("locality_vector", F_min-0.05, F_max+0.05)
        
        # Add target pose errors plot
        vis.addPlot("solution_error")
        vis.addPlot("solution_error")
        vis.logPlot("solution_error", "l2 (mm)", 0)
        vis.logPlot("solution_error", "angular (deg)", 0)
        vis.setPlotDuration("solution_error", 5)
        vis.setPlotRange("solution_error", 0, 30)

    @dataclass
    class DemoState:
        counter: int
        last_joint_vector: np.ndarray
        last_locality: np.ndarray
        ave_l2_error: float
        ave_angular_error: float
    
    def solve_locality(solver, P, locality):
        if len(P.shape) == 1:
            P = P.reshape(1, -1)
        if len(locality.shape) == 1:
            locality = locality.reshape(1, -1)
        assert locality.shape == (1, solver.r), locality.shape
        J_hat = solver.generate_ik_solutions(P, locality, num_sols=1, std=0.0, latent=np.zeros(solver.n))
        # (1, 1, solver.n)
        return J_hat

    def loop_fn(worlds, _demo_state):
        if from_nn:
            locality = locality_nn[_demo_state.counter % locality_nn.shape[0]]
            print(f"[INFO] locality={locality}, counter={_demo_state.counter}")
        else:
            locality = np.zeros((ik_solver.r))
            for i in range(ik_solver.r):
                counter = config.time_dilation * _demo_state.counter
                offset = 2 * PI * i / ik_solver.r
                locality[i] = (
                    F_max * np.sin(counter / 25 + offset)
                )
        
        _demo_state.last_locality = locality

        # Get solutions to pose of random sample
        ik_solutions = solve_locality(ik_solver, target_pose, locality)
        
        l2_errors, ang_errors = ik_solver.evaluate_pose_error_J3d_P2d(ik_solutions, target_pose.reshape(-1, solver.m), return_all=True)
        # print(f"l2_errors.shape: {l2_errors.shape}")
        _demo_state.ave_l2_error = l2_errors.mean().item() * 1000
        _demo_state.ave_ang_error = np.rad2deg(ang_errors.mean().item())
        # qs = robot._x_to_qs(solutions)
        ik_solutions = ik_solutions.reshape(ik_solver.n)
        robot.set_klampt_robot_config(ik_solutions)

        # Update _demo_state
        _demo_state.counter += 1
        _demo_state.last_joint_vector = ik_solutions
        
    def viz_update_fn(worlds, _demo_state):
        del worlds
        if show_mug:
            R, t = robot._klampt_ee_link.getTransform() # type: ignore
            R, t = np.array(R), np.array(t)
            distance_to_move = 0.15
            t += R[-3:] * distance_to_move
            # rotate the mug around the z-axis of the end effector with 90 degrees
            R = so3.mul(R, so3.from_axis_angle(([0, 0, 1], PI / 2)))  # type: ignore
            # rotate the mug around the y-axis of the end effector with -120 degrees
            R = so3.mul(R, so3.from_axis_angle(([0, 1, 0], PI * -120 / 180)))  # type: ignore
            mug.setCurrentTransform(R, t)  # type: ignore
        
        for i in range(3):
            vis.logPlot(f"joint_vector", f"joint_{i}", _demo_state.last_joint_vector[i])
        for i in range(ik_solver.r):
            vis.logPlot(f"locality_vector", f"locality_{i}", _demo_state.last_locality[i])
        
        vis.logPlot("solution_error", "l2 (mm)", _demo_state.ave_l2_error)
        vis.logPlot("solution_error", "angular (deg)", _demo_state.ave_ang_error)

    demo_state = DemoState(
        counter=0, last_joint_vector=np.zeros(ik_solver.n), last_locality=np.zeros(ik_solver.r), ave_l2_error=0, ave_angular_error=0
    )
    _run_demo(
        robot, config.n_worlds, setup_fn, loop_fn, viz_update_fn, demo_state=demo_state, time_p_loop=config.time_p_loop, title=title
    )


def oscillate_target(ik_solver: Solver, nb_sols=5):
    """Oscillating target pose"""
    config = Config()
    title = "Solutions for oscillating target pose"
    
    robot = ik_solver.robot
    target_pose_fn = _TARGET_POSE_FUNCTIONS[robot.name] # type: ignore

    def setup_fn(worlds):
        vis.add("coordinates", coordinates.manager())
        for i in range(len(worlds)):
            vis.add(f"robot_{i}", worlds[i].robot(0))

        # Add target pose plot
        vis.addPlot("target_pose")
        vis.logPlot("target_pose", "target_pose x", 0)
        vis.setPlotDuration("target_pose", 5)
        vis.addPlot("solution_error")
        vis.addPlot("solution_error")
        vis.logPlot("solution_error", "l2 (mm)", 0)
        vis.logPlot("solution_error", "angular (deg)", 0)
        vis.setPlotDuration("solution_error", 5)
        vis.setPlotRange("solution_error", 0, 8)

        # update the cameras pose
        # vp = vis.getViewport()
        # camera_tf = vp.get_transform()
        # vis.setViewport(vp)
        # vp.fit((0, 0.25, 0.5), 1.5)

    @dataclass
    class DemoState:
        counter: int
        target_pose: np.ndarray
        ave_l2_error: float
        ave_angular_error: float
        
    def solve_pose(solver, P):
        if len(P.shape) == 1:
            P = P.reshape(1, -1)
        # num_sols from locality.
        F = solver.select_reference_posture(P, "knn", num_sols=nb_sols)
        P = np.repeat(P, nb_sols, axis=0)
        assert F.shape[0] == P.shape[0], (F.shape, P.shape)
        J_hat = solver.generate_ik_solutions(P, F, num_sols=1, std=0.0, latent=np.zeros(solver.n))
        # (1, 1, solver.n)
        return J_hat.reshape(nb_sols, 1, solver.n)

    def loop_fn(worlds, _demo_state):
        # Update target pose
        _demo_state.target_pose = target_pose_fn(_demo_state.counter)

        # Get solutions to pose of random sample
        ik_solutions = solve_pose(ik_solver, _demo_state.target_pose)
        l2_errors, ang_errors = ik_solver.evaluate_pose_error_J3d_P2d(ik_solutions, _demo_state.target_pose.reshape(-1, solver.m), return_all=True)
        # print(f"l2_errors.shape: {l2_errors.shape}")
        _demo_state.ave_l2_error = l2_errors.mean().item() * 1000
        _demo_state.ave_ang_error = np.rad2deg(ang_errors.mean().item())
        ik_solutions = ik_solutions.reshape(nb_sols, solver.n)

        # Update viz with solutions
        qs = robot._x_to_qs(ik_solutions)
        for i in range(nb_sols):
            worlds[i].robot(0).setConfig(qs[i])

        # Update _demo_state
        _demo_state.counter += 1

    def viz_update_fn(worlds, _demo_state):
        del worlds
        _plot_pose("target_pose.", _demo_state.target_pose)
        vis.logPlot("target_pose", "target_pose x", _demo_state.target_pose[0])
        vis.logPlot("solution_error", "l2 (mm)", _demo_state.ave_l2_error)
        vis.logPlot("solution_error", "angular (deg)", _demo_state.ave_ang_error)

    demo_state = DemoState(counter=0, target_pose=target_pose_fn(0), ave_l2_error=0, ave_angular_error=0)
    _run_demo(
        robot, nb_sols, setup_fn, loop_fn, viz_update_fn, demo_state=demo_state, time_p_loop=config.time_p_loop, title=title
    )


def random_target_pose(ik_solver: Solver, nb_sols=20):
    """Set the end effector to a randomly drawn pose. Generate and visualize `nb_sols` solutions for the pose"""
    config = Config()
    config.time_p_loop = 5
    title = "Solutions for random target pose"
    
    robot = ik_solver.robot
    
    def generate_random_pose():
        J, P = robot.sample_joint_angles_and_poses(n=1)
        return P.reshape(-1)

    def setup_fn(worlds):
        vis.add("coordinates", coordinates.manager())
        for i in range(len(worlds)):
            vis.add(f"robot_{i}", worlds[i].robot(0))

        # Add target pose plot
        vis.addPlot("target_pose")
        vis.logPlot("target_pose", "target_pose x", 0)
        vis.setPlotDuration("target_pose", 5)
        vis.addPlot("solution_error")
        vis.addPlot("solution_error")
        vis.logPlot("solution_error", "l2 (mm)", 0)
        vis.logPlot("solution_error", "angular (deg)", 0)
        vis.setPlotDuration("solution_error", 5)
        vis.setPlotRange("solution_error", 0, 15)

        # update the cameras pose
        vp = vis.getViewport()
        camera_tf = vp.get_transform()
        vis.setViewport(vp)
        vp.fit((0, 0.25, 0.5), 1.5)

    @dataclass
    class DemoState:
        counter: int
        target_pose: np.ndarray
        ave_l2_error: float
        ave_angular_error: float
        
    def solve_pose(solver, P):
        if len(P.shape) == 1:
            P = P.reshape(1, -1)
        # num_sols from locality.
        F = solver.select_reference_posture(P, "knn", num_sols=nb_sols)
        P = np.repeat(P, nb_sols, axis=0)
        assert F.shape[0] == P.shape[0], (F.shape, P.shape)
        J_hat = solver.generate_ik_solutions(P, F, num_sols=1, std=0.0, latent=np.zeros(solver.n))
        # (1, 1, solver.n)
        return J_hat.reshape(nb_sols, 1, solver.n)

    def loop_fn(worlds, _demo_state):
        # Update target pose
        _demo_state.target_pose = generate_random_pose()
        # Get solutions to pose of random sample
        ik_solutions = solve_pose(ik_solver, _demo_state.target_pose)
        l2_errors, ang_errors = ik_solver.evaluate_pose_error_J3d_P2d(ik_solutions, _demo_state.target_pose.reshape(-1, solver.m), return_all=True)
        # print(f"l2_errors.shape: {l2_errors.shape}")
        _demo_state.ave_l2_error = l2_errors.mean().item() * 1000
        _demo_state.ave_ang_error = np.rad2deg(ang_errors.mean().item())
        ik_solutions = ik_solutions.reshape(nb_sols, solver.n)

        # Update viz with solutions
        qs = robot._x_to_qs(ik_solutions)
        for i in range(nb_sols):
            worlds[i].robot(0).setConfig(qs[i])

        # Update _demo_state
        _demo_state.counter += 1

    def viz_update_fn(worlds, _demo_state):
        del worlds
        _plot_pose("target_pose.", _demo_state.target_pose)
        vis.logPlot("target_pose", "target_pose x", _demo_state.target_pose[0])
        vis.logPlot("solution_error", "l2 (mm)", _demo_state.ave_l2_error)
        vis.logPlot("solution_error", "angular (deg)", _demo_state.ave_ang_error)

    demo_state = DemoState(counter=0, target_pose=generate_random_pose(), ave_l2_error=0, ave_angular_error=0)
    _run_demo(
        robot, nb_sols, setup_fn, loop_fn, viz_update_fn, demo_state=demo_state, time_p_loop=config.time_p_loop, title=title
    )


def oscillate_joints(robot: Robot):
    """Set the end effector to a randomly drawn pose. Generate and visualize `nb_sols` solutions for the pose"""
    config = Config()
    config.time_p_loop = 1 / 60  # 60Hz, in theory
    title = "Oscillate joint angles"
    
    inc = 0.01

    class DemoState:
        def __init__(self):
            self.q = np.array([lim[0] for lim in robot.actuated_joints_limits]) # type: ignore
            self.increasing = True

    def setup_fn(worlds):
        vis.add("robot", worlds[0].robot(0))
        assert len(worlds) == 1

    def loop_fn(worlds, _demo_state):
        no_change = True
        for i in range(robot.n_dofs):
            joint_limits = robot.actuated_joints_limits[i]
            if _demo_state.increasing:
                if _demo_state.q[i] < joint_limits[1]:
                    _demo_state.q[i] += inc
                    no_change = False
                    break
            else:
                if _demo_state.q[i] > joint_limits[0]:
                    _demo_state.q[i] -= inc
                    no_change = False
                    break
        if no_change:
            _demo_state.increasing = not _demo_state.increasing

        q = robot._x_to_qs(np.array([_demo_state.q])) # type: ignore
        worlds[0].robot(0).setConfig(q[0])

    

    def viz_update_fn(worlds, _demo_state):
        return

    demo_state = DemoState()
    _run_demo(
        robot,
        1,
        setup_fn,
        loop_fn,
        viz_update_fn,
        time_p_loop=config.time_p_loop,
        title=title,
        demo_state=demo_state,
    )
    

def mug_moving(ik_solver: Solver, nb_sols=5, from_locality=True):
    """Oscillating target pose"""
    config = Config()
    config.time_p_loop = 0.01
    title = "Solutions for oscillating target pose"
    
    robot = ik_solver.robot
    target_pose_fn = _TARGET_POSE_FUNCTIONS[robot.name] # type: ignore
    
    mug_path = "/home/luca/paik/data/visualization_resources/objects/mug.obj"
    mug = Geometry3D()
    mug.loadFile(mug_path)

    
    def setup_fn(worlds):
        vis.add("coordinates", coordinates.manager())
        for i in range(len(worlds)):
            vis.add(f"robot_{i}", worlds[i].robot(0))

        vis.add("mug",mug)
        # Add target pose plot
        vis.addPlot("target_pose")
        vis.logPlot("target_pose", "target_pose x", 0)
        vis.setPlotDuration("target_pose", 5)
        vis.addPlot("solution_error")
        vis.addPlot("solution_error")
        vis.logPlot("solution_error", "l2 (mm)", 0)
        vis.logPlot("solution_error", "angular (deg)", 0)
        vis.setPlotDuration("solution_error", 5)
        vis.setPlotRange("solution_error", 0, 8)

        # update the cameras pose
        # vp = vis.getViewport()
        # camera_tf = vp.get_transform()
        # vis.setViewport(vp)
        # vp.fit((0, 0.25, 0.5), 1.5)

    @dataclass
    class DemoState:
        counter: int
        target_pose: np.ndarray
        ave_l2_error: float
        ave_angular_error: float
        
    def solve_pose(solver, P):
        if len(P.shape) == 1:
            P = P.reshape(1, -1)
        # num_sols from locality.
        F = solver.select_reference_posture(P, "knn", num_sols=nb_sols)
        P = np.repeat(P, nb_sols, axis=0)
        assert F.shape[0] == P.shape[0], (F.shape, P.shape)
        J_hat = solver.generate_ik_solutions(P, F, num_sols=1, std=0.0, latent=np.zeros(solver.n))
        # (1, 1, solver.n)
        return J_hat.reshape(nb_sols, 1, solver.n)

    def loop_fn(worlds, _demo_state):
        # Update target pose
        _demo_state.target_pose = target_pose_fn(_demo_state.counter)

        # Get solutions to pose of random sample
        ik_solutions = solve_pose(ik_solver, _demo_state.target_pose)
        l2_errors, ang_errors = ik_solver.evaluate_pose_error_J3d_P2d(ik_solutions, _demo_state.target_pose.reshape(-1, solver.m), return_all=True)
        # print(f"l2_errors.shape: {l2_errors.shape}")
        _demo_state.ave_l2_error = l2_errors.mean().item() * 1000
        _demo_state.ave_ang_error = np.rad2deg(ang_errors.mean().item())
        ik_solutions = ik_solutions.reshape(nb_sols, solver.n)

        # Update viz with solutions
        qs = robot._x_to_qs(ik_solutions)
        for i in range(nb_sols):
            worlds[i].robot(0).setConfig(qs[i])

        # Update _demo_state
        _demo_state.counter += 1

    def viz_update_fn(worlds, _demo_state):
        del worlds
        R, t = robot._klampt_ee_link.getTransform() # type: ignore
        R, t = np.array(R), np.array(t)
        distance_to_move = 0.1
        t += R[-3:] * distance_to_move + np.array([0.1, 0, 0])

        mug.setCurrentTransform(R, t) # type: ignore
        _plot_pose("target_pose.", _demo_state.target_pose)
        vis.logPlot("target_pose", "target_pose x", _demo_state.target_pose[0])
        vis.logPlot("solution_error", "l2 (mm)", _demo_state.ave_l2_error)
        vis.logPlot("solution_error", "angular (deg)", _demo_state.ave_ang_error)

    demo_state = DemoState(counter=0, target_pose=target_pose_fn(0), ave_l2_error=0, ave_angular_error=0)
    _run_demo(
        robot, nb_sols, setup_fn, loop_fn, viz_update_fn, demo_state=demo_state, time_p_loop=config.time_p_loop, title=title
    )


if __name__ == "__main__":
    solver = Solver(solver_param=PANDA_PAIK, load_date="0707-1713", work_dir="/home/luca/paik")
    # solver = Solver(solver_param=PANDA_NSF, load_date="0115-0234", work_dir="/home/luca/paik")
    oscillate_latent(solver, show_mug=True, change_F_per_cnt=200)
    # oscillate_locality(solver, show_mug=True, from_nn=True)
    # oscillate_target(solver)
    # random_target_pose(solver)
    # oscillate_joints(solver.robot)
    # mug_moving(solver, nb_sols=10, from_locality=True)