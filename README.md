# DPAKS
DPAKS: DETR Guided by Prior Auxiliary Knowledge for Small Object Detection  
_**The code will be presented after the article is accepted.**_
# 1. Needed Environment
CUDA 11.8   
Python 3.9
# 2. Install
## 2.1 python
    conda create --name dpaks python=3.9 -y
## 2.2 torch
### plan a:  
    pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
### plan b:  
    conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia -y
## 2.3 mmengine 0.9.0
### plan a:  
    pip install -U openmim
    mim install mmengine
### plan b:  
    git clone https://github.com/open-mmlab/mmengine.git
    cd mmengine
    pip install -e . -v 
## 2.4 mmcv 2.1.0
    git clone https://github.com/open-mmlab/mmcv.git
    cd mmcv
    pip install -r requirements/optional.txt
    pip install -e .
    python .dev_scripts/check_installation.py
## 2.5 dpaks (based on mmdetection 3.2.0)
    cd dpaks
    pip install -v -e .
# 3 Train
    python tools/train.py config/dpaks/dpaks-4scale_r50_channelmap-retinanet_5scale_r50_fpn-12e_visdrone.py \
	--work-dir results/dpaks/
