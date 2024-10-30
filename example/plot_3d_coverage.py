# Import required packages
import os
import pickle
import torch
import numpy as np
from paik.solver import Solver, get_solver
from paik.file import save_pickle, load_pickle

from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from common.config import Config_Diversity
from ikp import get_number_of_distinct_solutions, solver_batch, random_ikp, numerical_inverse_kinematics_batch
from _visualize import visualize_ik_solutions

# set random seeds for numpy and torch for reproducibility
np.random.seed(11)
torch.manual_seed(0)

import OpenGL.GLUT as glut
glut.glutInit()

BASE_DISTRIBUTION = "box_uniform" # "diag_normal" or "box_uniform"

def plot_random_3d_joints_scatter(ax, keys, J: dict, c: dict, marker: dict, label: dict, joint_nums=None):
    if joint_nums is None:
        joint_nums = np.random.choice(J[0].shape[-1], 3, replace=False)
    x, y, z = joint_nums
    
    j = 0
    colors = plt.cm.tab20.colors
    markers = ['o', 's', 'd', '^', 'v', '<', '>', 'p', 'P', '*', 'h', 'H', '+', 'x', 'X', '|', '_', '1', '2', '3', '4']
        
    for i in keys:
        Ji = J[i].reshape(-1, J[i].shape[-1])
        
        # if c[i], marker[i], label[i] (dict) is not exist, use a default color, marker, label
        if i not in c:
            # color is a tuple of RGB (0-1)
            c[i] = np.atleast_2d(colors[j])
            j += 1

        if i not in marker:
            marker[i] = markers[j]
            j += 1

        if i not in label:
            label[i] = i.upper()
                        
        if i == 'num':
            # alpha is used to make the NUM solution more transparent
            ax.scatter(Ji[:, x], Ji[:, y], Ji[:, z], c=c[i], marker=marker[i], label=label[i], alpha=0.8, s=70)
        else:
            ax.scatter(Ji[:, x], Ji[:, y], Ji[:, z], c=c[i], marker=marker[i], label=label[i], s=70)
            
    

    ax.set_xlabel(f'Joint {x}')
    ax.set_ylabel(f'Joint {y}')
    ax.set_zlabel(f'Joint {z}')
    
    # return legend handles
    return ax # ax.get_legend_handles_labels()

def get_solvers(robot_name: str, work_dir: str) -> dict[str, Solver]:
    """
    Get solvers for the robot

    Args:
        robot_name (str): robot name
        work_dir (str): the working directory for the solver

    Returns:
        dict: solvers for the robot
    """
    
    solvers = {
        'nsf': get_solver(arch_name="nsf", robot_name=robot_name, load=True, work_dir=work_dir),
        'paik': get_solver(arch_name="paik", robot_name=robot_name, load=True, work_dir=work_dir),
    }
    
    for key, solver in solvers.items():
        solver.base_name = BASE_DISTRIBUTION
    print(f"Using Base distribution: {BASE_DISTRIBUTION}")
    
    return solvers

def get_J_map(solvers: dict, num_poses: int, num_sols: int, std: float, record_dir: str, load: bool=True):
    """
    Generate J_map for 3D scatter plot

    Args:
        robot_name (str): robot name
        num_poses (int): number of poses
        num_sols (int): number of solutions per pose to generate
        std (float): the standard deviation for the solver
        record_dir (str): the record directory for the solver

    Returns:
        dict: J_map for 3D scatter plot
    """
    file_path = record_dir + f"/J_map_{num_poses}_{num_sols}_{std}.pkl"
    if load:
        return load_pickle(file_path)
    
    _, P = solvers['nsf'].robot.sample_joint_angles_and_poses(num_poses)

    # Experiment shows the first execute of random_ikp is slow, so we execute a dummy one.
    random_ikp(solvers['nsf'], P, num_sols, solver_batch, std=std, verbose=False)
    
    J_map = {
        'num': random_ikp(solvers['nsf'], P, num_sols, numerical_inverse_kinematics_batch, verbose=False)[0],
        **{f'{key}_{std}': random_ikp(solver, P, num_sols, solver_batch, std=std, verbose=False)[0] for key, solver in solvers.items()},
        **{f'{key}_0.25': random_ikp(solver, P, num_sols, solver_batch, std=0.25, verbose=False)[0] for key, solver in solvers.items()},
    }
    
    save_pickle(file_path, J_map)
    
    return J_map

def visualize_3d_joints_scatter(J_map: dict, num_x_sub_plots: int, num_y_sub_plots: int, random_joint_nums: np.ndarray, record_dir: str):
    """
    Visualize 3D Joints Scatter

    Args:
        J_map (dict): the J_map for 3D scatter plot
        num_x_sub_plots (int): the number of subplots in x-axis
        num_y_sub_plots (int): the number of subplots in y-axis
        record_dir (str): the directory to save the results
    """
    
    
    # All colors are np.atleast_2d is used to make it 2D
    # num is gray and alpha is 0.5, 
    # nsf_0.01 and nsf_0.25 are from color of red series, but nsf_0.25 is ligher than nsf_0.01
    # paik_0.01 and paik_0.25 are from color of blue series, but paik_0.25 is ligher than paik_0.01
    _lighter_color = lambda color: np.clip(np.atleast_2d([c + 0.2 for c in color]), 0, 1)
    
    solver_color = {
        'num': np.atleast_2d((0.5, 0.5, 0.5, 0.5)),
        'nsf': np.atleast_2d((0.8, 0.1, 0.1)),
        'paik': np.atleast_2d((0.1, 0.1, 0.8)),
        'pick_up': 'orange',
        'not_pick_up': np.atleast_2d((0.1, 0.1, 0.8)),
    }
    
    color_map = {}
    for key in J_map.keys():
        if key == 'num' or key == 'pick_up' or key == 'not_pick_up':
            color_map[key] = solver_color[key]
        # if key with _0.25, use lighter color
        elif key.endswith('_0.25'):
            solver_name = key.split('_')[0]
            color_map[key] = _lighter_color(solver_color[solver_name])
        else:
            solver_name = key.split('_')[0]
            color_map[key] = solver_color[solver_name]

    marker_map = {}
    for key in color_map.keys():
        if key == 'num':
            marker_map[key] = 'x'
        else:
            marker_map[key] = 's'

    # num is NUM, pick_up is Pick up, not_pick_up is Not pick up, else key_{std} is upper key (std={std})
    label_map = {}
    for key in color_map.keys():
        if key == 'num':
            label_map[key] = 'NUM'
        elif key == 'pick_up':
            label_map[key] = 'Pick up'
        elif key == 'not_pick_up':
            label_map[key] = 'Not pick up'
        else:
            label_map[key] = f'{key.upper()}'
            
    fig = plt.figure(figsize=(15, 12))
    gs = GridSpec(num_x_sub_plots, num_y_sub_plots, figure=fig)
    handles_list = {}
    keys_list = [(key, 'num') for key in J_map.keys() if key != 'num']

    # Initialize lists to store ticks and limits
    x_min, y_min, z_min = float('inf'), float('inf'), float('inf')
    x_max, y_max, z_max = float('-inf'), float('-inf'), float('-inf')

    axes = []
    # First pass to determine the common ticks and limits
    for x in range(num_x_sub_plots):
        for y in range(num_y_sub_plots):
            ax = fig.add_subplot(gs[x, y], projection='3d')
            count = x * num_y_sub_plots + y
            ax = plot_random_3d_joints_scatter(ax, keys_list[count], J_map, color_map, marker_map, label_map, random_joint_nums)
            axes.append(ax)

            x_min = min(x_min, ax.get_xlim()[0])
            x_max = max(x_max, ax.get_xlim()[1])
            y_min = min(y_min, ax.get_ylim()[0])
            y_max = max(y_max, ax.get_ylim()[1])
            z_min = min(z_min, ax.get_zlim()[0])
            z_max = max(z_max, ax.get_zlim()[1])
            
            handles, _ = ax.get_legend_handles_labels()
            handles_list.update(dict(zip(keys_list[count], handles)))
            
    for x in range(num_x_sub_plots):
        for y in range(num_y_sub_plots):
            count = x * num_y_sub_plots + y
            ax = axes[count]
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_zlim(z_min, z_max)
            
            # show num_ticks ticks and round to 1 decimal
            num_ticks = 3
            ax.set_xticks(np.round(np.linspace(x_min, x_max, num_ticks), 1))
            ax.set_yticks(np.round(np.linspace(y_min, y_max, num_ticks), 1))
            ax.set_zticks(np.round(np.linspace(z_min, z_max, num_ticks), 1))
            
            # Set font sizes for axis labels and tick labels
            fontsize = 14
            labelpad = 10  # Adjust this value to move the labels further away from the ticks
            ax.set_xlabel(ax.get_xlabel(), fontsize=fontsize, labelpad=labelpad)
            ax.set_ylabel(ax.get_ylabel(), fontsize=fontsize, labelpad=labelpad)
            ax.set_zlabel(ax.get_zlabel(), fontsize=fontsize, labelpad=labelpad)
            ax.tick_params(axis='both', which='major', labelsize=12)
    # Adjust the spacing between subplots
    plt.subplots_adjust(wspace=0.2, hspace=0.2)
    fig.suptitle(f'3D Joints Scatter with Random {config.num_poses} poses and {config.num_sols} solutions')
    fig.legend(handles=handles_list.values(), loc='upper center', bbox_to_anchor=(0.9, 0.9))
    plt.savefig(record_dir + f"/scatter.png")
    plt.show()
    
def _visualize_3d_scatter_w_highlight_distinct_clusters(ax, J_map, key, eps: float):
    """
    Visualize 3D scatter plot with highlighting distinct clusters

    Args:
        ik_solutions (np.ndarray): the IK solutions
        record_dir (str): the directory to save the results
    """
    ik_solutions = J_map[key]
    
    if len(ik_solutions.shape) == 3:
        ik_solutions = ik_solutions.reshape(-1, ik_solutions.shape[-1])
    
    num_clusters, labels = get_number_of_distinct_solutions(ik_solutions[:, 4:], eps=eps, min_samples=1)
    
    # plot the 3D scatter plot with highlighting distinct clusters
    scatter = ax.scatter(ik_solutions[:, 4], ik_solutions[:, 5], ik_solutions[:, 6], c=labels, cmap='viridis')
    ax.set_xlabel('Joint 5')
    ax.set_ylabel('Joint 6')
    ax.set_zlabel('Joint 7')
    ax.set_title(f"{key.upper()}/'s Plot w/ {num_clusters} clusters")
    return scatter

def visualize_3d_scatter_w_highlight_distinct_clusters(J_map: dict, record_dir: str, eps: float):
    """
    Visualize 3D scatter plot with highlighting distinct clusters

    Args:
        J_map (dict): the J_map for 3D scatter plot
        record_dir (str): the directory to save the results
    """
    scale = 2
    figsize = (5, 4)
    scaled_figsize = tuple([i * scale for i in figsize])
    fig = plt.figure(figsize=scaled_figsize)
    num_solvers = len(J_map.keys())
    
    # Determine global min and max for each axis
    # get the min and max values for each axis
    all_ik_solutions = np.vstack(list(J_map.values()))
    all_ik_solutions = all_ik_solutions.reshape(-1, all_ik_solutions.shape[-1])
    # filter out nan values
    all_ik_solutions = all_ik_solutions[~np.isnan(all_ik_solutions).any(axis=1)]
    # joint 5, 6, 7
    x_min, x_max = all_ik_solutions[:, 4].min(), all_ik_solutions[:, 4].max()
    y_min, y_max = all_ik_solutions[:, 5].min(), all_ik_solutions[:, 5].max()
    z_min, z_max = all_ik_solutions[:, 6].min(), all_ik_solutions[:, 6].max()
    
    # visualize 3d scatter plot with highlighting distinct clusters for each solver
    for i, key in enumerate(J_map.keys()):
        ax = fig.add_subplot(2, (num_solvers + 1) // 2, i + 1, projection='3d')
        scatter = _visualize_3d_scatter_w_highlight_distinct_clusters(ax, J_map, key, eps)
        
        # Set consistent axis limits and ticks
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)
        num_ticks = 5
        ax.set_xticks(np.round(np.linspace(x_min, x_max, num_ticks), 1))
        ax.set_yticks(np.round(np.linspace(y_min, y_max, num_ticks), 1))
        ax.set_zticks(np.round(np.linspace(z_min, z_max, num_ticks), 1))
    
    cbar = fig.colorbar(scatter, ax=ax, pad=0.1, shrink=0.5, aspect=5)
    cbar.set_label('Cluster Labels')
    plt.tight_layout()
    file_path = record_dir + f"/scatter_subplots_eps{eps}.png"
    plt.savefig(file_path)
    plt.show()
    print(f"3D Scatter Plot with {num_solvers} solvers and highlighting distinct clusters is saved at {file_path}")


if __name__ == "__main__":
    
    config = Config_Diversity()
    config.num_poses = 1
    config.num_sols = 100
    std = 0.1
    robot_name = "panda"
    num_x_sub_plots = 2
    num_y_sub_plots = 2
    pick_up_solver = f'paik_{std}'
    pick_up_idxs = [83, 74, 67, 63, 39] # 83, 74, 67, 63, 39
    
    solvers = get_solvers(robot_name, config.workdir)
    
    J_map = get_J_map(solvers, config.num_poses, config.num_sols, std, config.record_dir, load=False)
    
    for key, ik_solutions in J_map.items():
        print(f"{key}: {ik_solutions.shape}")

    # epss = [0.01, 0.05, 0.1, 0.15]
    # for eps in epss:
    #     visualize_3d_scatter_w_highlight_distinct_clusters(J_map, config.record_dir, eps=eps)

    random_joint_list = np.array([[4, 5, 6]]) # [[1, 3, 5], [2, 4, 6]]
    for random_joint_nums in random_joint_list:
        print(f"Pick up index: {pick_up_idxs}, Random joint numbers: {random_joint_nums}")
        J_map['pick_up'] = J_map[pick_up_solver][pick_up_idxs]
        J_map['not_pick_up'] = np.delete(J_map[pick_up_solver], pick_up_idxs, axis=0)
        visualize_3d_joints_scatter(J_map, num_x_sub_plots, num_y_sub_plots, random_joint_nums, config.record_dir)
        visualize_ik_solutions(robot=solvers['nsf'].robot, ik_solutions=J_map['pick_up']) # type: ignore
    
    
    # random pick up 
    # num_random_pick_up = 30
    # for _ in range(num_random_pick_up):
    #     random_idx = np.random.choice(J_map[pick_up_solver].shape[0], 1)
    #     print(f"Random pick up index: {random_idx}")
    #     J_map['pick_up'] = J_map[pick_up_solver][random_idx]
    #     J_map['not_pick_up'] = np.delete(J_map[pick_up_solver], random_idx, axis=0)
    #     visualize_3d_joints_scatter(J_map, num_x_sub_plots, num_y_sub_plots, random_joint_nums, config.record_dir) 
        