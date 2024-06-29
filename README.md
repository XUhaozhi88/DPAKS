# DPAKS
DPAKS: DETR Guided by Prior Auxiliary Knowledge for Small Object Detection

The code will be presented when the article of this code is accepted.

# Needed Environment
CUDA 11.8
Python 3.9

# Install
python
conda create --name dpaks python=3.9 -y

torch
plan a:
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
plan b:
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia -y

mmengine
plan a:
pip install -U openmim
mim install mmengine
plan b:
git clone https://github.com/open-mmlab/mmengine.git
cd mmengine
pip install -e . -v

mmcv
git clone https://github.com/open-mmlab/mmcv.git
cd mmcv
pip install -r requirements/optional.txt
pip install -e .
python .dev_scripts/check_installation.py

dpaks (based on mmdetection)
cd dpaks
pip install -v -e .
