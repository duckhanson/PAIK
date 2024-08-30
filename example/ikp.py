from paik.solver import get_solver
from common.config import Config_IKP
from common.display import save_ikp

if __name__ == "__main__":
    solver_names = ["nsf", "paik"]
    robot_names = ["panda"] # ["panda", "fetch", "fetch_arm", "iiwa7", "atlas_arm", "atlas_waist_arm", "baxter_arm"]
    config = Config_IKP()
    
    for robot_name in robot_names:
        for solver_name in solver_names:
            solver = get_solver(arch_name=solver_name, robot_name=robot_name)
            
            # dummy run
            solver.random_ikp(num_poses=config.num_poses, num_sols=config.num_sols, std=config.std, verbose=False)
        
            # real run
            results = solver.random_ikp(num_poses=config.num_poses, num_sols=config.num_sols, std=config.std, verbose=True)
            
            result_name = f"{robot_name}_{solver_name}"
            save_ikp(config.record_dir, result_name, *results)
    
    
    
