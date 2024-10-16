# Import required packages
from tabnanny import verbose
import time
import numpy as np
import pandas as pd
from tqdm import trange
import torch
# from paik.solver import Solver
# from paik.settings import PANDA_NSF, PANDA_PAIK

from ikflow.utils import set_seed
from ikflow.model_loading import get_ik_solver
from jkinpylib.evaluation import solution_pose_errors

from common.config import Config_IKP
from common.display import display_ikp

from common.config import Config_Posture
from common.display import display_posture
from common.evaluate import mmd_J3d_J3d

from common.config import Config_Diversity
from common.file import save_diversity, load_poses_and_numerical_ik_sols

def solve_batch(solver, P: np.ndarray, num_sols: int, std: float):
    num_poses = len(P)
    J = torch.empty((num_sols, num_poses, solver.robot.n_dofs),
                    dtype=torch.float32, device="cpu")
    # cast P to torch tensor
    P = torch.tensor(P, dtype=torch.float32, device="cuda")
    
    for i in trange(num_poses):
        J[:, i, :] = solver.generate_ik_solutions(
            P[i],
            n=num_sols,
            latent_scale=std,
            refine_solutions=False,
            return_detailed=False,
        ).clone().detach().cpu()  # type: ignore
    return J

def get_ikflow_solver(robot_name: str):
    if robot_name == "panda":
        return get_ik_solver("panda__full__lp191_5.25m")
    elif robot_name == "fetch":
        return get_ik_solver("fetch_full_temp_nsc_tpm")
    elif robot_name == "fetch_arm":
        return get_ik_solver("fetch_arm__large__mh186_9.25m")
    else:
        raise ValueError(f"Unknown robot name: {robot_name}")

def test_random_ikp_with_mmd(robot_name: str, num_poses: int, num_sols: int, std: float, record_dir: str, verbose: bool=False):
    
    ikflow_solver, _ = get_ikflow_solver(robot_name)
    
    _, P = ikflow_solver.robot.sample_joint_angles_and_poses(num_poses)
    
    results = {
        "num": random_ikp(ikflow_solver, P, num_sols, numerical_inverse_kinematics_batch, verbose=False),
        "ikflow": random_ikp(ikflow_solver, P, num_sols, solve_batch, std=std, verbose=True)
    }
    
    mmd_results = {
        "num": 0,
        "ikflow": mmd_J3d_J3d(results["ikflow"][0], results["num"][0], num_poses)
    }

    if verbose:
        print_out_format = "l2: {:1.2f} mm, ang: {:1.2f} deg, time: {:1.1f} ms"
        print("="*70)
        print(f"Robot: {robot_name} computes {num_sols} solutions and average over {num_poses} poses")
        # print the results of the random IKP with l2_mm, ang_deg, and solve_time_ms
        for key in results.keys():
            print(f"{key.upper()}:  {print_out_format.format(*results[key][1:])}", f", MMD: {mmd_results[key]}")
        print("="*70)
    
    # get count results keys
    num_solvers = len(results.keys())
    
    df = pd.DataFrame({
        "robot": [robot_name] * num_solvers,
        "solver": list(results.keys()),
        "mmd": [mmd_results[key] for key in mmd_results.keys()],
        "l2_mm": [results[key][1] for key in results.keys()],
        "ang_deg": [results[key][2] for key in results.keys()],
        "solve_time_ms": [results[key][3] for key in results.keys()],
    }) 
    
    # save the results to a csv file
    df_file_path = f"{record_dir}/ikp_ikflow_{robot_name}_{num_poses}_{num_sols}_{std}.csv"
    df.to_csv(df_file_path, index=False)
    print(f"Results are saved to {df_file_path}")
    

# def posture(config: Config_Posture):
#     set_seed()
#     # Build IKFlowSolver and set weights
#     ik_solver, _ = get_ik_solver("panda__full__lp191_5.25m")
#     J, P = ik_solver.robot.sample_joint_angles_and_poses(n=config.num_poses)
#     l2 = np.zeros((config.num_sols, config.num_poses))
#     ang = np.zeros((config.num_sols, config.num_poses))
#     J_hat = torch.empty(
#         (config.num_sols, config.num_poses, ik_solver.robot.n_dofs),
#         dtype=torch.float32,
#         device="cpu",
#     )
#     if config.num_poses < config.num_sols:
#         for i in trange(config.num_poses):
#             J_hat[:, i, :] = ik_solver.solve(
#                 P[i],
#                 n=config.num_sols,
#                 latent_scale=config.std,
#                 refine_solutions=False,
#                 return_detailed=False,
#             ).cpu()  # type: ignore

#             l2[:, i], ang[:, i] = solution_pose_errors(
#                 ik_solver.robot, J_hat[:, i, :], P[i]
#             )
#     else:
#         for i in trange(config.num_sols):
#             J_hat[i] = ik_solver.solve_n_poses(
#                 P,
#                 latent_scale=config.std,
#                 refine_solutions=False,
#                 return_detailed=False,
#             ).cpu()
#             l2[i], ang[i] = solution_pose_errors(ik_solver.robot, J_hat[i], P)
#     # J_hat.shape = (num_sols, num_poses, num_dofs or n)
#     # J.shape = (num_poses, num_dofs or n)
#     distance_J = compute_distance_J(J_hat, J)
#     display_posture(
#         config.record_dir,
#         "ikflow",
#         l2.flatten(),
#         ang.flatten(),
#         distance_J.flatten(),
#         config.success_distance_thresholds,
#     )


# def diversity(config: Config_Diversity, solver: Any, std: float, P: np.ndarray):
#     assert P.shape[:2] == (config.num_poses, config.num_sols)
#     P = P.reshape(-1, P.shape[-1])
#     P = make_batches(P, config.batch_size)  # type: ignore
#     J_hat = batches_back_to_array(
#         [
#             solver.solve_n_poses(batch_P, latent_scale=std).cpu().numpy()
#             for batch_P in tqdm(P)
#         ]
#     )
#     J_hat = np.expand_dims(J_hat, axis=0)
#     assert J_hat.shape == (
#         1,
#         config.num_poses * config.num_sols,
#         J_hat.shape[-1],
#     ), f"Expected: {(1, config.num_poses * config.num_sols, J_hat.shape[-1])}, Got: {J_hat.shape}"
#     return J_hat


# def ikflow(config: Config_Diversity, solver: Solver):
#     set_seed()
#     # Build IKFlowSolver and set weights
#     ik_solver, _ = get_ik_solver("panda__full__lp191_5.25m")
#     iterate_over_base_stds(config, "ikflow", ik_solver, solver, ikflow_solve)


# def iterate_over_base_stds(
#     config: Config_Diversity,
#     iksolver_name: str,
#     solver: Any,
#     paik_solver: Solver,
#     solve_fn,
# ):
#     l2_mean = np.empty((len(config.base_stds)))
#     ang_mean = np.empty((len(config.base_stds)))
#     mmd_mean = np.empty((len(config.base_stds)))
#     P, J_ground_truth = load_poses_and_numerical_ik_sols(config.record_dir)
#     P = np.expand_dims(P, axis=1).repeat(config.num_sols, axis=1)
#     J_hat_base_stds = np.empty(
#         (
#             len(config.base_stds),
#             config.num_poses,
#             config.num_sols,
#             J_ground_truth.shape[-1],
#         )
#     )

#     for i, std in enumerate(config.base_stds):
#         J_hat = solve_fn(config, solver, std, P)

#         l2, ang = paik_solver.evaluate_pose_error_J3d_P2d(
#             J_hat, P.reshape(-1, P.shape[-1]), return_all=True
#         )
#         J_hat = J_hat.reshape(config.num_poses, config.num_sols, -1)
#         l2_mean[i] = l2.mean()
#         ang_mean[i] = ang.mean()
#         mmd_mean[i] = mmd_evaluate_multiple_poses(
#             J_hat, J_ground_truth, num_poses=config.num_poses
#         )
#         J_hat_base_stds[i] = J_hat
#         assert not np.isnan(mmd_mean[i])

#     save_diversity(
#         config.record_dir,
#         iksolver_name,
#         J_hat_base_stds,
#         l2_mean,
#         ang_mean,
#         mmd_mean,
#         config.base_stds,
#     )


if __name__ == "__main__":
    config = Config_IKP()
    robot_name = "panda" # "panda", "fetch", "fetch_arm"
    test_random_ikp_with_mmd(robot_name=robot_name, num_poses=config.num_poses, num_sols=config.num_sols, std=config.std, record_dir=config.record_dir, verbose=True)
