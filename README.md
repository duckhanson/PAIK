# pafik: Posture-Aware Inverse Kinematics

# install steps:
0. conda create pafik python=3.9.12
1. conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
2. python install .
3. pip install hnne
4. python -m ipykernel install --user --name pafik --display-name "pafik"

# possible install issues:
1. ERROR: Could not find a version that satisfies the requirement torch==2.0.1+cu118 (from pafik) (from versions: 1.7.1, 1.8.0, 1.8.1, 1.9.0, 1.9.1, 1.10.0, 1.10.1, 1.10.2, 1.11.0, 1.12.0, 1.12.1, 1.13.0, 1.13.1, 2.0.0, 2.0.1, 2.1.0, 2.1.1, 2.1.2, 2.2.0)
ERROR: No matching distribution found for torch==2.0.1+cu118

- try conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia

2. AssertionError: Torch not compiled with CUDA enabled

# possible run issues:
1. RuntimeError: "addmm_cuda" not implemented for 'Int'
- your gpu may not support int operations, so replace zuko/nn.py:181
    precedence = adjacency.int() @ adjacency.int().t() == adjacency.sum(dim=-1)
    with 
    precedence = adjacency.float() @ adjacency.float().t() == adjacency.sum(dim=-1)
