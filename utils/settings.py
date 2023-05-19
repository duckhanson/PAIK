from torch.nn import LeakyReLU


class Config:
    def __init__(self):
        # robot
        self.dof = 7
        
        # data
        self.x_data_path = './data/feature.npy' # joint configuration
        self.y_data_path = './data/target.npy' # end-effector position
        
        # hnee parameter
        self.reduced_dim = 4
        self.num_neighbors = 1000
        
        
        # flow parameter
        self.architecture = 'nsf'
        self.device = 'cuda'
        self.num_features = 7
        self.num_conditions = 3 + 4 + 1 # position + posture + noise = 3-dim + 4-dim + 1-dim 
        self.num_transforms = 14
        self.subnet_shape = [1024] * 4
        self.activation = LeakyReLU
        
        # nflow parameter
        self.shrink_ratio = .30
        
        
        # training
        self.lr = 4e-3
        self.lr_decay = 3e-1
        self.batch_size = 256
        self.noise_esp = 1e-3
        self.num_epochs = 10
        self.num_steps_save = 2000
        self.num_test_data = 60
        self.num_test_samples = 40
        self.save_path = './weights/nsf.pth'
        
        # log
        self.err_his_path = './log/err_his.npy'
        self.train_loss_his_path = './log/train_loss_his.npy'
        
        # experiment
        self.show_pose_features_path = './data/show_pose/features.npy'
        self.show_pose_pidxs_path = './data/show_pose/pidxs.npy'
        self.show_pose_errs_path = './data/show_pose/errs.npy'
        self.show_pose_log_probs_path = './data/show_pose/log_probs.npy'
        self.traj_dir = './data/trajectory/'
        
        
    
    def __repr__(self):
        return str(self.__dict__)

config = Config()