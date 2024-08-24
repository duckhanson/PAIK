# Import required packages
from cProfile import label
import time
from turtle import color
from typing import Any
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from paik.solver import get_solver

from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from common.config import Config_Diversity
from common.file import save_diversity, load_poses_and_numerical_ik_sols
from common.evaluate import (
    mmd_evaluate_multiple_poses,
    make_batches,
    batches_back_to_array,
)

from solution_space_coverage import random_ikp, numerical_inverse_kinematics_batch, nsf_batch, paik_batch

# set random seeds for numpy and torch for reproducibility
np.random.seed(0)
torch.manual_seed(0)

if __name__ == "__main__":
    config = Config_Diversity()
    config.num_poses = 5
    config.num_sols = 50
    std = 0.01
    
    nsf_solver = get_solver(arch_name="nsf", robot_name="panda", load=True, work_dir=config.workdir)
    paik_solver = get_solver(arch_name="paik", robot_name="panda", load=True, work_dir=config.workdir)

    _, P = nsf_solver.robot.sample_joint_angles_and_poses(config.num_poses)
    
    # Experiment shows the first execute of random_ikp is slow, so we execute a dummy one.
    _ = random_ikp(paik_solver, P, config.num_sols, paik_batch, std=std, verbose=False)
    
    J_nsf, l2_nsf, l2_std_nsf, ang_nsf, ang_std_nsf = random_ikp(nsf_solver, P, config.num_sols, nsf_batch, std=std, verbose=True)
    J_paik, l2_paik, l2_std_paik, ang_paik, ang_std_paik = random_ikp(paik_solver, P, config.num_sols, paik_batch, std=std, verbose=True)
    J_nsf_025, l2_nsf_025, l2_std_nsf_025, ang_nsf_025, ang_std_nsf_025 = random_ikp(nsf_solver, P, config.num_sols, nsf_batch, std=0.25, verbose=True)

    # # filter out the solutions with l2 > 10 or ang > 10
    # J_nsf = J_nsf[(l2_nsf < 10) & (ang_nsf < 10)]
    # J_paik = J_paik[(l2_paik < 10) & (ang_paik < 10)]
    # J_nsf_025 = J_nsf_025[(l2_nsf_025 < 10) & (ang_nsf_025 < 10)]
    
    def plot_random_3d_joints_scatter(ax, J_nsf, J_paik, c, label, joint_nums=None):
        if joint_nums is None:
            joint_nums = np.random.choice(J_nsf.shape[-1], 3, replace=False)
        x, y, z = joint_nums
        ax.scatter(J_nsf[:, x], J_nsf[:, y], J_nsf[:, z], c=c[0], marker='o', label=label[0])
        ax.scatter(J_paik[:, x], J_paik[:, y], J_paik[:, z], c=c[1], marker='^', label=label[1])
        ax.set_xlabel(f'Joint {x}')
        ax.set_ylabel(f'Joint {y}')
        ax.set_zlabel(f'Joint {z}')
        
        # # set legend
        # ax.legend(loc='right')
        
        # return legend handles
        return ax.get_legend_handles_labels()
        
    
    def plot_random_2d_joints_scatter(ax, J_nsf, J_paik, J_nsf_025, c, label):
        x, y = np.random.choice(J_nsf.shape[-1], 2, replace=False)
        ax.scatter(J_nsf[:, x], J_nsf[:, y], c=c[0], marker='o', label=label[0])
        ax.scatter(J_paik[:, x], J_paik[:, y], c=c[1], marker='^', label=label[1])
        ax.scatter(J_nsf_025[:, x], J_nsf_025[:, y], c=c[2], marker='o', label=label[2])
        ax.set_xlabel(f'Joint {x}')
        ax.set_ylabel(f'Joint {y}')
        
    # plot and save 6 scatter of J_nsf, J_paik in randomly 3 dimensions (under solver.n)
    fig = plt.figure()
    gs = GridSpec(2, 2, figure=fig)
    num_x_sub_plots = 1
    num_y_sub_plots = 2
    
    colors = ['r', 'b', 'violet']
    labels = [f'NSF (std={std})', f'PAIK (std={std})', 'NSF (std=0.25)']
    
    handles_list = []
    for x in range(num_x_sub_plots):
        random_joint_nums = np.random.choice(J_nsf.shape[-1], 3, replace=False)
        for y in range(num_y_sub_plots):
            ax = fig.add_subplot(gs[x, y], projection='3d')
            if y == 0:
                c = colors[:2]
                l = labels[:2]
                handles, _ = plot_random_3d_joints_scatter(ax, J_nsf, J_paik, c=c, label=l, joint_nums=random_joint_nums)
            else:
                # select the last 2 colors and labels and reverse the order
                c = colors[1:]
                l = labels[1:]
                
                c.reverse()
                l.reverse()
                handles, _ = plot_random_3d_joints_scatter(ax, J_nsf_025, J_paik, c=c, label=l, joint_nums=random_joint_nums)
                handles.pop() # only need the first one (J_nsf_025)
            handles_list += handles
            
    # set title
    fig.suptitle(f'3D Joints Scatter with Random {config.num_poses} poses and {config.num_sols} solutions')
    
    # plot legend
    fig.legend(handles=handles_list, loc='upper right')
    
    # plot table of l2 and ang
    ax = fig.add_subplot(gs[-1, :])
    ax.axis('off')
    
    table_data = [
        ['Solver', 'Base std', 'L2 (mm)', 'Ang (°)'],
        ['NSF', f'{std:.2f}', f'{l2_nsf:.1f}±{l2_std_nsf:.1f}', f'{ang_nsf:.1f}±{ang_std_nsf:.1f}'],
        ['PAIK', f'{std:.2f}', f'{l2_paik:.1f}±{l2_std_paik:.1f}', f'{ang_paik:.1f}±{ang_std_paik:.1f}'],
        ['NSF', 0.25, f'{l2_nsf_025:.1f}±{l2_std_nsf_025:.1f}', f'{ang_nsf_025:.1f}±{ang_std_nsf_025:.1f}'],
    ]
    # plot table location to right
    table = ax.table(cellText=table_data, loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)

    # save and show
    plt.savefig(config.record_dir + f"/scatter.png")
    plt.show()