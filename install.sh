conda create paik python=3.9.12
conda activate paik
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install .
pip install hnne
python -m ipykernel install --user --name paik --display-name "paik"