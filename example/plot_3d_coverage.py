# Import required packages
import torch
import numpy as np
from paik.solver import get_solver

from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from common.config import Config_Diversity
from ikp import solver_batch, random_ikp, numerical_inverse_kinematics_batch

# set random seeds for numpy and torch for reproducibility
np.random.seed(0)
torch.manual_seed(0)

def plot_random_3d_joints_scatter(ax, keys, J, c, marker, label, joint_nums=None):
    if joint_nums is None:
        joint_nums = np.random.choice(J[0].shape[-1], 3, replace=False)
    x, y, z = joint_nums
    
    for i in keys:
        Ji = J[i].reshape(-1, J[i].shape[-1])
        if i == 'num':
            # alpha is used to make the NUM solution more transparent
            ax.scatter(Ji[:, x], Ji[:, y], Ji[:, z], c=c[i], marker=marker[i], label=label[i], alpha=0.8)
        else:
            ax.scatter(Ji[:, x], Ji[:, y], Ji[:, z], c=c[i], marker=marker[i], label=label[i])
    ax.set_xlabel(f'Joint {x}')
    ax.set_ylabel(f'Joint {y}')
    ax.set_zlabel(f'Joint {z}')
    
    # return legend handles
    return ax.get_legend_handles_labels()

if __name__ == "__main__":
    config = Config_Diversity()
    config.num_poses = 1
    config.num_sols = 70
    std = 0.01
    robot_name = "panda"
    num_x_sub_plots = 2
    num_y_sub_plots = 2
    
    nsf_solver = get_solver(arch_name="nsf", robot_name=robot_name, load=True, work_dir=config.workdir)
    paik_solver = get_solver(arch_name="paik", robot_name=robot_name, load=True, work_dir=config.workdir)

    _, P = nsf_solver.robot.sample_joint_angles_and_poses(config.num_poses)
    
    # Experiment shows the first execute of random_ikp is slow, so we execute a dummy one.
    random_ikp(paik_solver, P, config.num_sols, solver_batch, std=std, verbose=False)
    
    num_results = random_ikp(nsf_solver, P, config.num_sols, numerical_inverse_kinematics_batch, verbose=False)
    nsf_001_results = random_ikp(nsf_solver, P, config.num_sols, solver_batch, std=std, verbose=True)
    paik_001_results = random_ikp(paik_solver, P, config.num_sols, solver_batch, std=std, verbose=True)
    nsf_025_results = random_ikp(nsf_solver, P, config.num_sols, solver_batch, std=0.25, verbose=True)
    paik_025_results = random_ikp(paik_solver, P, config.num_sols, solver_batch, std=0.25, verbose=True)

        
    # plot and save scatter of J_nsf, J_paik in randomly 3 dimensions (under solver.n)
    fig = plt.figure()
    gs = GridSpec(num_x_sub_plots, num_y_sub_plots, figure=fig)
    
    color_map = {
        'num': 'orange',
        'nsf_001': 'r',
        'paik_001': 'b',
        'nsf_025': 'violet',
        'paik_025': 'g',
    }
    
    marker_map = {
        'num': 'x',
        'nsf_001': 'o',
        'paik_001': '^',
        'nsf_025': 's',
        'paik_025': 'd',
    }

    label_map = {
        'num': 'NUM',
        'nsf_001': f'NSF (std={std})',
        'paik_001': f'PAIK (std={std})',
        'nsf_025': f'NSF (std=0.25)',
        'paik_025': f'PAIK (std=0.25)',
    }
    
    J_map = {
        'num': num_results[0],
        'nsf_001': nsf_001_results[0],
        'paik_001': paik_001_results[0],
        'nsf_025': nsf_025_results[0],
        'paik_025': paik_025_results[0],
    }
    
    handles_list = {}

    random_joint_nums = np.random.choice(J_map['nsf_001'].shape[-1], 3, replace=False)
    for x in range(num_x_sub_plots):
        for y in range(num_y_sub_plots):
            ax = fig.add_subplot(gs[x, y], projection='3d')
            if x == 0 and y == 0:
                keys = ['nsf_001', 'num']
            elif x == 0 and y == 1:
                keys = ['nsf_025', 'num']
            elif x == 1 and y == 0:
                keys = ['paik_001', 'num']
            elif x == 1 and y == 1:
                keys = ['paik_025', 'num']
            else:
                raise ValueError(f"Invalid y value: {y}")
            
            handles, _ = plot_random_3d_joints_scatter(ax, keys, J_map, color_map, marker_map, label_map, random_joint_nums)
            # each key is the label, and each value is the handle
            handles_list.update(dict(zip(keys, handles)))
    
    # set title
    fig.suptitle(f'3D Joints Scatter with Random {config.num_poses} poses and {config.num_sols} solutions')
    
    # plot legend, and set location to upper right but not overlap with the plot
    fig.legend(handles=handles_list.values(), loc='upper right', bbox_to_anchor=(0.9, 0.9))
    
    # save and show
    plt.savefig(config.record_dir + f"/scatter.png")
    plt.show()