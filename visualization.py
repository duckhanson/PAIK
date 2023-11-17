from typing import List, Callable, Any
from time import sleep
from dataclasses import dataclass

from pprint import pprint
from jrl.robot import Robot
from jrl.robots import Panda, Fetch, FetchArm
from jrl.evaluation import solution_pose_errors

import klampt
from klampt.math import so3
from klampt.model import coordinates, trajectory
from klampt import vis, GeometricPrimitive, Geometry3D, Point
from klampt import WorldModel
import numpy as np
import torch
import torch.optim
from paik.settings import config
from paik.solver import Solver, DEFAULT_SOLVER_PARAM_M3, DEFAULT_SOLVER_PARAM_M7
from paik.model import get_robot


class Visualizer(Solver):
    def __init__(self, robot: Robot, solver_param: dict) -> None:
        super().__init__(robot, solver_param)

    def _plot_pose(self, name: str, pose: np.ndarray, hide_label: bool = False):
        vis.add(
            name,
            coordinates.Frame(
                name=name, worldCoordinates=(so3.from_quaternion(pose[3:]), pose[0:3])
            ),  # type: ignore
            hide_label=hide_label,
        )

    def _run_demo(
        self,
        n_worlds: int,
        setup_fn: Callable[[List[WorldModel]], None],
        loop_fn: Callable[[List[WorldModel], Any], None],
        viz_update_fn: Callable[[List[WorldModel], Any], None],
        demo_state: Any = None,
        time_p_loop: float = 2.5,
        title: str = "Anonymous demo",
        load_terrain: bool = True,
        load_box: bool = False,
    ):
        """Internal function for running a demo."""

        worlds = [self.robot.klampt_world_model.copy() for _ in range(n_worlds)]

        # TODO: Adjust terrain height for each robot
        if load_terrain:
            terrain_filepath = "visualization_resources/terrains/plane.off"
            res = worlds[0].loadTerrain(terrain_filepath)
            assert res, f"Failed to load terrain '{terrain_filepath}'"
            vis.add("terrain", worlds[0].terrain(0))

        if load_box:
            box_filepath = "visualization_resources/objects/thincube.off"
            res = worlds[0].loadRigidObject(box_filepath)
            assert res, f"Failed to load obj '{box_filepath}'"
            vis.add("box", worlds[0].rigidObject(0))

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

    def sample_latent_space(self, num_samples: int = 5):
        self._random_target_pose(num_samples=num_samples, k=1)

    def sample_posture_space(self, k: int = 5):
        old_shrink_ratio = self.shrink_ratio
        self.shrink_ratio = 0
        self._random_target_pose(num_samples=1, k=k)
        self.shrink_ratio = old_shrink_ratio

    def _random_target_pose(self, num_samples: int = 5, k: int = 1):
        """Set the end effector to a randomly drawn pose. Generate and visualize `nb_sols` solutions for the pose"""
        if k > 1:
            assert (
                self.shrink_ratio == 0
            ), "Shrink ratio must be 0 for k > 1 (sweep posture, fix latent)"

        nb_sols = num_samples * k

        def setup_fn(worlds):
            vis.add(f"robot_goal", worlds[0].robot(0))
            vis.setColor(f"robot_goal", 0.5, 1, 1, 0)
            vis.setColor(
                (f"robot_goal", self.robot.end_effector_link_name), 0, 1, 0, 0.7
            )

            for i in range(1, nb_sols + 1):
                vis.add(f"robot_{i}", worlds[i].robot(0))
                vis.setColor(f"robot_{i}", 1, 1, 1, 1)
                vis.setColor(
                    (f"robot_{i}", self.robot.end_effector_link_name), 1, 1, 1, 0.71
                )

        def loop_fn(worlds, _demo_state):
            # Get random sample
            random_sample = self.robot.sample_joint_angles(1)
            random_sample_q = self.robot._x_to_qs(random_sample)
            worlds[0].robot(0).setConfig(random_sample_q[0])
            target_pose = self.robot.forward_kinematics_klampt(random_sample)[0]

            # Get solutions to pose of random sample
            ik_solutions = self.solve_set_k(
                target_pose, num_samples, k=k, return_numpy=True
            )
            qs = self.robot._x_to_qs(ik_solutions)  # type: ignore
            for i in range(nb_sols):
                worlds[i + 1].robot(0).setConfig(qs[i])

        time_p_loop = 2.5
        title = "Solutions for randomly drawn poses - Green link is the target pose"

        def viz_update_fn(worlds, _demo_state):
            return

        n_worlds = nb_sols + 1
        self._run_demo(
            n_worlds,
            setup_fn,
            loop_fn,
            viz_update_fn,
            time_p_loop=time_p_loop,
            title=title,
        )

    # TODO(@jeremysm): Add/flesh out plots. Consider plotting each solutions x, or error
    def visualize_path_following(
        self,
        load_time: str = "",
        num_traj: int = 5,
        shrink_ratio: float = 0,
        enable_box: bool = False,
        seed=47,
    ):
        P_path, J_traj, ref_F = self.path_following(
            load_time=load_time,
            num_traj=num_traj,
            shrink_ratio=shrink_ratio,
            enable_plot=True,
            seed=seed,
        )  # type: ignore
        P_path = np.tile(P_path, (num_traj, 1))
        Qs = np.empty((num_traj, J_traj.shape[1], 17))
        for i, J in enumerate(J_traj):
            qs = np.asarray(self.robot._x_to_qs(J))  # type: ignore
            Qs[i] = qs
        P_path = P_path.reshape(-1, P_path.shape[-1])
        Qs = Qs.reshape(-1, Qs.shape[-1])

        if enable_box:
            self._visualize_box(Qs, P_path)
        else:
            self._oscillate_target(Qs, P_path)

    def _oscillate_target(self, Qs, P_path):
        """Oscillating target pose"""

        time_p_loop = 0.01
        title = "Solutions for oscillating target pose"

        def target_pose_fn(counter: int):
            return P_path[max(counter, len(P_path) - 1)]

        def setup_fn(worlds):
            vis.add("coordinates", coordinates.manager())
            for i in range(len(worlds)):
                vis.add(f"robot_{i}", worlds[i].robot(0))
                vis.setColor(f"robot_{i}", 1, 1, 1, 1)
                vis.setColor(
                    (f"robot_{i}", self.robot.end_effector_link_name), 1, 1, 1, 0.71
                )

            # # Axis
            # vis.add("x_axis", trajectory.Trajectory([1, 0], [[1, 0, 0], [0, 0, 0]]))
            # vis.add("y_axis", trajectory.Trajectory([1, 0], [[0, 1, 0], [0, 0, 0]]))

            # # Add target pose plot
            # vis.addPlot("target_pose")
            # vis.logPlot("target_pose", "target_pose x", 0)
            # vis.setPlotDuration("target_pose", 5)
            # vis.addPlot("solution_error")
            # vis.addPlot("solution_error")
            # vis.logPlot("solution_error", "l2 (mm)", 0)
            # vis.logPlot("solution_error", "angular (deg)", 0)
            # vis.setPlotDuration("solution_error", 5)
            # vis.setPlotRange("solution_error", 0, 25)

        @dataclass
        class DemoState:
            counter: int
            target_pose: np.ndarray
            direction: bool

        def _update_demo_state(_demo_state):
            # Update _demo_state
            if _demo_state.direction:
                _demo_state.counter += 1
            else:
                _demo_state.counter -= 1

            if _demo_state.counter == len(P_path) - 1:
                _demo_state.direction = False
            elif _demo_state.counter == 0:
                _demo_state.direction = True
            return _demo_state

        def loop_fn(worlds, _demo_state):
            # Update target pose
            _demo_state.target_pose = target_pose_fn(_demo_state.counter)

            # Get solutions to pose of random sample
            # ik_solutions = self.solve(_demo_state.target_pose, nb_sols, k=1)
            # l2_errors, ang_errors = solution_pose_errors(self.robot, ik_solutions, _demo_state.target_pose)

            # _demo_state.ave_l2_error = np.mean(l2_errors) * 1000
            # _demo_state.ave_ang_error = np.rad2deg(np.mean(ang_errors))

            # Update viz with solutions
            worlds[0].robot(0).setConfig(Qs[_demo_state.counter])

            _demo_state = _update_demo_state(_demo_state)

        def viz_update_fn(worlds, _demo_state):
            # _plot_pose("target_pose.", _demo_state.target_pose)
            # vis.logPlot("target_pose", "target_pose x", _demo_state.target_pose[0])
            # vis.logPlot("solution_error", "l2 (mm)", _demo_state.ave_l2_error)
            # vis.logPlot("solution_error", "angular (deg)", _demo_state.ave_ang_error)
            pass

        demo_state = DemoState(counter=0, target_pose=target_pose_fn(0), direction=True)

        self._run_demo(
            1,
            setup_fn,
            loop_fn,
            viz_update_fn,
            demo_state=demo_state,
            time_p_loop=time_p_loop,
            title=title,
        )

    def _visualize_box(self, Qs, P_path):
        def target_pose_fn(counter: int):
            return P_path[counter]

        """Shows how to pop up a visualization window with a world"""

        # add the world to the visualizer
        def setup_fn(worlds):
            vis.add("coordinates", coordinates.manager())
            for i in range(len(worlds)):
                vis.add(f"robot_{i}", worlds[i].robot(0))
                vis.setColor(f"robot_{i}", 1, 1, 1, 1)
                vis.setColor(
                    (f"robot_{i}", self.robot.end_effector_link_name), 1, 1, 1, 0.71
                )

                vis.add(f"box_{i}", worlds[i].rigidObject(0))
                vis.setColor(f"box_{i}", 0.95, 0.95, 0.95, 0.8)

        @dataclass
        class DemoState:
            counter: int
            target_pose: np.ndarray
            direction: bool

            def update(self):
                # Update _demo_state
                if self.direction:
                    self.counter += 1
                else:
                    self.counter -= 1

                if self.counter == len(P_path) - 1:
                    self.direction = False
                elif self.counter == 0:
                    self.direction = True
                self.target_pose = target_pose_fn(self.counter)

        def loop_fn(worlds, _demo_state):
            # Update target pose
            q = Qs[_demo_state.counter]
            target_pose = _demo_state.target_pose
            t = target_pose[:3]
            R = so3.from_quaternion(target_pose[3:])

            for i in range(len(worlds)):
                worlds[i].rigidObject(0).setTransform(R, t)
                worlds[i].robot(0).setConfig(q)

            _demo_state.update()

        def viz_update_fn(worlds, _demo_state):
            self._plot_pose("target_pose.", _demo_state.target_pose)

        demo_state = DemoState(counter=0, target_pose=target_pose_fn(0), direction=True)
        time_p_loop = 0.01
        title = "Solutions for randomly drawn poses - Green link is the target pose"

        self._run_demo(
            1,
            setup_fn,
            loop_fn,
            viz_update_fn,
            demo_state=demo_state,
            time_p_loop=time_p_loop,
            title=title,
            load_terrain=False,
            load_box=True,
        )


# =========================
# Main function
# =========================


def main():
    visualizer = Visualizer(robot=get_robot(), solver_param=DEFAULT_SOLVER_PARAM_M7)
    # visualizer.sample_latent_space(num_samples=5)
    # visualizer.sample_posture_space(k=5)
    visualizer.visualize_path_following(
        load_time="1111215818", num_traj=3, shrink_ratio=0, enable_box=True, seed=37
    )


if __name__ == "__main__":
    main()
