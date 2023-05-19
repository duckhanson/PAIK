import torch

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'

param = {
    'degree_interval': 60,
    'device': device,

    ## franka panda ##
    'panda_dof': 7,
    'panda_urdf': './src/panda_arm_hand_fixed.urdf',
    'goal_length': 3,

    ## Training VAE ##
    'layer_length': 2048,
    'latent_length': 4,
    'lr': 9e-4,
    'kl_ratio': 2e-4,
    'weight_dir': './weights',
    'image_dir': './images',
    'batch_size': 256 * 8,
    'epochs': 300,
    'patience': 30,


    ## Data csv ##
    'data_dir': './data',
    'tables': ['degree', 'psik', 'local', 'global', 'num_p', 'num_e', 'esik']
}

degree_database = {
    ## database ##
    # database table name
    'degree_table': f"degree{param['degree_interval']}",
    # columns defined inside database table
    # remove end_ori_.
    'degree_cols': ['j1', 'j2', 'j3', 'j4', 'j5', 'j6', 'j7', 'ee_px', 'ee_py', 'ee_pz'],

    'z_cols': ['z1', 'z2', 'z3', 'z4'],
}

psik_database = {
    'psik_table': f"p_{degree_database['degree_table']}",
    'psik_cols': degree_database['degree_cols'],
}

local_database = {
    'local_table': f"local_{degree_database['degree_table']}",
    'local_cols': degree_database['degree_cols'] + ['ml'],
}

global_database = {
    'global_table': f"global_{degree_database['degree_table']}",
    'global_cols': local_database['local_cols'] + ['mg'],
}

num_p_database = {
    'num_p_table': f"num_p_{degree_database['degree_table']}",
    'num_p_cols': degree_database['degree_cols'],
}

esik_database = {
    'esik_table': f"e_{degree_database['degree_table']}",
    'esik_cols': local_database['local_cols'],
}

num_e_database = {
    'num_e_table': f"num_e_{degree_database['degree_table']}",
    'num_e_cols': esik_database['esik_cols'],
}

ee_path = {
    'ee_path_file': f"{param['data_dir']}/ee_path.p"
}

param.update(degree_database)
param.update(psik_database)
param.update(local_database)
param.update(global_database)
param.update(num_p_database)
param.update(num_e_database)
param.update(esik_database)
param.update(ee_path)
