# Import required packages
import torch
import numpy as np
from paik.solver import get_solver

from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from common.config import Config_Diversity
from ikp import solver_batch, random_ikp

# set random seeds for numpy and torch for reproducibility
np.random.seed(0)
torch.manual_seed(0)

def plot_random_3d_joints_scatter(ax, J, c, label, joint_nums=None):
    left_J = J[0]
    right_J = J[1]
    
    # reshape input to (num_poses*num_sols, n)
    left_J = left_J.reshape(-1, left_J.shape[-1])
    right_J = right_J.reshape(-1, right_J.shape[-1])
    
    if joint_nums is None:
        joint_nums = np.random.choice(left_J.shape[-1], 3, replace=False)
    x, y, z = joint_nums
    ax.scatter(left_J[:, x], left_J[:, y], left_J[:, z], c=c[0], marker='o', label=label[0])
    ax.scatter(right_J[:, x], right_J[:, y], right_J[:, z], c=c[1], marker='^', label=label[1])
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
    num_x_sub_plots = 4
    num_y_sub_plots = 2
    
    nsf_solver = get_solver(arch_name="nsf", robot_name=robot_name, load=True, work_dir=config.workdir)
    paik_solver = get_solver(arch_name="paik", robot_name=robot_name, load=True, work_dir=config.workdir)

    _, P = nsf_solver.robot.sample_joint_angles_and_poses(config.num_poses)
    
    # Experiment shows the first execute of random_ikp is slow, so we execute a dummy one.
    random_ikp(paik_solver, P, config.num_sols, solver_batch, std=std, verbose=False)
    
    nsf_001_results = random_ikp(nsf_solver, P, config.num_sols, solver_batch, std=std, verbose=True)
    paik_001_results = random_ikp(paik_solver, P, config.num_sols, solver_batch, std=std, verbose=True)
    nsf_025_results = random_ikp(nsf_solver, P, config.num_sols, solver_batch, std=0.25, verbose=True)

        
    # plot and save scatter of J_nsf, J_paik in randomly 3 dimensions (under solver.n)
    fig = plt.figure()
    gs = GridSpec(num_x_sub_plots, num_y_sub_plots, figure=fig)
    
    color_map = {
        'nsf_001': 'r',
        'paik_001': 'b',
        'nsf_025': 'violet',
    }
    
    label_map = {
        'nsf_001': f'NSF (std={std})',
        'paik_001': f'PAIK (std={std})',
        'nsf_025': f'NSF (std=0.25)',
    }
    
    J_map = {
        'nsf_001': nsf_001_results[0],
        'paik_001': paik_001_results[0],
        'nsf_025': nsf_025_results[0],
    }
    
    handles_list = {}
    
    for x in range(num_x_sub_plots):
        random_joint_nums = np.random.choice(J_map['nsf_001'].shape[-1], 3, replace=False)
        for y in range(num_y_sub_plots):
            ax = fig.add_subplot(gs[x, y], projection='3d')
            
            if y == 0:
                left_model = 'nsf_001'
                right_model = 'paik_001'
            elif y == 1:
                left_model = 'nsf_025'
                right_model = 'paik_001'
            else:
                raise ValueError(f"Invalid y value: {y}")
            
            c = (color_map[left_model], color_map[right_model])
            l = (label_map[left_model], label_map[right_model])
            J = (J_map[left_model], J_map[right_model])
            handles, _ = plot_random_3d_joints_scatter(ax, J, c, l, random_joint_nums)
            handles_list.update({left_model: handles[0], right_model: handles[1]})
    
    # set title
    fig.suptitle(f'3D Joints Scatter with Random {config.num_poses} poses and {config.num_sols} solutions')
    
    # plot legend
    fig.legend(handles=handles_list.values(), loc='upper right')
    
    # save and show
    plt.savefig(config.record_dir + f"/scatter.png")
    plt.show()