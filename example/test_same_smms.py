# Import required packages
import numpy as np
from tqdm import trange
from sklearn.neighbors import NearestNeighbors
from collections import deque

from paik.solver import Solver
from paik.settings import (
    PANDA_NSF,
    PANDA_PAIK,
)


def is_collision_free(solver, q1, q2, pose_esp=1e-3, num_steps=10):
    """
    Check if a straight-line path between two configurations is collision-free.

    Parameters:
    - q1: start configuration (list or array).
    - q2: goal configuration (list or array).
    - pose_esp: Pose error threshold for self-motion checking.
    - num_steps: Number of steps to check along the path.

    Returns:
    - True if the path is collision-free, False otherwise.
    """
    
    # Compute end-effector poses
    ee = solver.robot.forward_kinematics(q1.reshape(-1, solver.n))[0, :3]

    # Compute the straight line between q1 and q2
    q_line = np.linspace(q1, q2, num=num_steps)
    ee_line = solver.robot.forward_kinematics(q_line)[:, :3]
    
    for ql, eel in zip(q_line, ee_line):
        if solver.robot.config_self_collides(ql) or np.linalg.norm(eel - ee) > pose_esp:
            # print(f"Collision detected at pose={eel}, demo_p={ee}, distance={np.linalg.norm(eel - ee)}")
            return False
    return True


def sample_joint_angles_within_bounds_of_q1q2(solver, num_interval_samples, q1, q2):
    """
    Sample joint angles within the bounds of q1 and q2 with consideration of joint limits.
    
    sample_joint_angles_within_bounds_of_q1q2(solver, 10, q, q+qr)

    Parameters:
    - num_interval_samples: Number of samples to generate.
    - q1: First configuration (list or array).
    - q2: Second configuration (list or array).

    Returns:
    - List of samples.
    """
    boundaries = np.array([[min(q1[i], q2[i]), max(q1[i], q2[i])] for i in range(solver.n)])
    
    samples = np.zeros((num_interval_samples+2, solver.n))
    
    samples[0] = q1
    samples[1] = q2
    
    for i in range(num_interval_samples):
        samples[i+2] = np.random.uniform(boundaries[:, 0], boundaries[:, 1])
    
    assert np.all(samples >= boundaries[:, 0]) and np.all(samples <= boundaries[:, 1])
    return samples

def build_roadmap(solver, q_start, q_goal, num_interval_samples, num_nearest_neighbors, pose_esp=5e-3):
    """
    Build a roadmap connecting a start and goal configuration.

    Parameters:
    - q_start: Start configuration (list or array).
    - q_goal: Goal configuration (list or array).
    - num_interval_samples: Number of samples to generate.
    - num_nearest_neighbors: Number of nearest neighbors to consider.
    - pose_esp: Pose error threshold for self-motion checking.

    Returns:
    - List of samples.
    """
    num_total_samples = num_interval_samples + 2
    samples = sample_joint_angles_within_bounds_of_q1q2(solver, num_interval_samples, q_start, q_goal)
    tree = NearestNeighbors(n_neighbors=num_nearest_neighbors, algorithm='kd_tree').fit(samples)
    indices = tree.kneighbors(samples, return_distance=False)
    
    graph = {i: [] for i in range(num_total_samples)}
    
    # print(f"sample size: {samples.shape}")
    # print(f"neighbors.shape: {indices.shape}")
    
    for i, sample in enumerate(samples):
        for j in indices[i]:
            if i != j and j not in graph[i] and is_collision_free(solver, sample, samples[j], pose_esp=pose_esp):
                graph[i].append(j)
                graph[j].append(i)
                
    return graph

def check_reachable(solver, q_start, q_goal, num_interval_samples, num_nearest_neighbors, pose_esp=1e-3):
    
    # Base case
    if is_collision_free(solver, q_start, q_goal, pose_esp=pose_esp, num_steps=10):
        return True
    
    graph = build_roadmap(solver, q_start, q_goal, num_interval_samples=num_interval_samples, num_nearest_neighbors=num_nearest_neighbors, pose_esp=pose_esp)
    
    s, d = 0, 1 # Start and goal indices
    
    # Mark all the vertices as not visited
    visited = [False for i in range(num_interval_samples+2)]
 
    # Create a queue for BFS
    queue = deque()
 
    # Mark the current node as visited and enqueue it
    visited[s] = True
    queue.append(s)
 
    while (len(queue) > 0):
       
        # Dequeue a vertex from queue and print
        s = queue.popleft()
        # queue.pop_front()
 
        # Get all adjacent vertices of the dequeued vertex s
        # If a adjacent has not been visited, then mark it
        # visited  and enqueue it
        for i in graph[s]:
 
            # If this adjacent node is the destination node,
            # then return true
            if (i == d):
                return True
 
            # Else, continue to do BFS
            if (not visited[i]):
                visited[i] = True
                queue.append(i)
    # If BFS is complete without visiting d
    return False

def paik_same_SMM_solve(solver, std, J, P, num_sols):
    solver.base_std = std
    num_poses = J.shape[0]
    P = np.expand_dims(P, 1).repeat(num_sols, axis=1)
    assert P.shape[:2] == (num_poses, num_sols)

    F = solver.F[
            solver.P_knn.kneighbors(
                np.atleast_2d(P[:, 0]), n_neighbors=1, return_distance=False
            ).flatten()
        ]
    F = np.expand_dims(F, 1).repeat(num_sols, axis=1)
    F = F.flatten()

    # shape: (num_poses * num_sols, n)
    P = P.reshape(-1, P.shape[-1])
    J_hat = solver.solve_batch(P, F, 1)  # (1, num_poses * num_sols, n)
    assert J_hat.shape == (
        1,
        num_poses * num_sols,
        solver.n,
    ), f"Expected: {(1, num_poses * num_sols, solver.n)}, Got: {J_hat.shape}"
    return J_hat

def paik_diverse_SMM_solve(solver, std, J, P, num_sols):
    solver.base_std = std
    num_poses = J.shape[0]
    P = np.expand_dims(P, 1).repeat(num_sols, axis=1)
    assert P.shape[:2] == (num_poses, num_sols)

    F = solver.F[
            solver.P_knn.kneighbors(
                np.atleast_2d(P[:, 0]), n_neighbors=num_sols, return_distance=False
            ).flatten()
        ]

    # shape: (num_poses * num_sols, n)
    P = P.reshape(-1, P.shape[-1])
    J_hat = solver.solve_batch(P, F, 1)  # (1, num_poses * num_sols, n)
    assert J_hat.shape == (
        1,
        num_poses * num_sols,
        solver.n,
    ), f"Expected: {(1, num_poses * num_sols, solver.n)}, Got: {J_hat.shape}"
    return J_hat


if __name__ == "__main__":
    solver = Solver(solver_param=PANDA_PAIK, load_date="0703-0717", work_dir="/home/luca/paik")
    
    num_poses = 10 # 20
    num_sols = 10 # 50
    stds = np.linspace(0.01, 0.5, 10)
    J, P = solver.robot.sample_joint_angles_and_poses(n=num_poses)
    count_same_SMM = {std: np.zeros(len(P)) for std in stds}
    root_SMMs = {std: np.array([[i for i in range(num_sols)] for _ in range(len(P))]) for std in stds}
    
    for std in stds:
        J_hat = paik_same_SMM_solve(solver, std, J, P, num_sols=num_sols)
        l2 , ang = solver.evaluate_pose_error_J3d_P2d(J_hat, np.expand_dims(P, axis=1).repeat(num_sols, axis=1).reshape(-1, P.shape[-1]), return_all=True)
        J_hat = J_hat.reshape(len(P), num_sols, solver.n)

        for i in trange(len(P)):
            for j in range(num_sols):
                for k in range(j+1, num_sols):
                    if check_reachable(solver, J_hat[i, j], J_hat[i, k], num_interval_samples=50, num_nearest_neighbors=5, pose_esp=1e-3):
                        # print(f"Same SMM found at {i}, {j}, {k}, count={count_same_SMM[i]}")
                        root_SMMs[std][i, k] = root_SMMs[std][i, j]
            count_same_SMM[std][i] = np.max(np.bincount(root_SMMs[std][i]))

    print("Same SMM")
    for std in stds:
        print(f"std={std:.2f}, count_SMM={np.mean(count_same_SMM[std]):.2f}, rate/num_sols={round(100 * np.mean(count_same_SMM[std]) / num_sols)}%")
    
    count_same_SMM = {std: np.zeros(len(P)) for std in stds}
    root_SMMs = {std: np.array([[i for i in range(num_sols)] for _ in range(len(P))]) for std in stds}

    for std in stds:
        J_hat = paik_diverse_SMM_solve(solver, std, J, P, num_sols=num_sols)
        l2 , ang = solver.evaluate_pose_error_J3d_P2d(J_hat, np.expand_dims(P, axis=1).repeat(num_sols, axis=1).reshape(-1, P.shape[-1]), return_all=True)
        J_hat = J_hat.reshape(len(P), num_sols, solver.n)

        for i in trange(len(P)):
            for j in range(num_sols):
                for k in range(j+1, num_sols):
                    if check_reachable(solver, J_hat[i, j], J_hat[i, k], num_interval_samples=50, num_nearest_neighbors=5, pose_esp=1e-3):
                        # print(f"Same SMM found at {i}, {j}, {k}, count={count_same_SMM[i]}")
                        root_SMMs[std][i, k] = root_SMMs[std][i, j]
            count_same_SMM[std][i] = np.max(np.bincount(root_SMMs[std][i]))
    
    print("Diverse SMM")
    for std in stds:
        print(f"std={std:.2f}, count_SMM={np.mean(count_same_SMM[std]):.2f}, rate/num_sols={round(100 * np.mean(count_same_SMM[std]) / num_sols)}%")
    