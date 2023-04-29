# Feature Shrinkage Pyramid for Camouflaged Object Detection with Transformers (CVPR2023)

<p align="left">
<a href="https://arxiv.org/abs/2303.14816"><img src="https://img.shields.io/badge/Paper-arXiv-green"></a>
<a href="https://tzxiang.github.io/project/COD-FSPNet/index.html"><img src="https://img.shields.io/badge/Page-Project-blue"></a>
</p>

## Usage

> The training and testing experiments are conducted using PyTorch with 8 Tesla V100 GPUs of 36 GB Memory.

### 1. Prerequisites

> Note that FSPNet is only tested on Ubuntu OS with the following environments.

- Creating a virtual environment in terminal: `conda create -n FSPNet python=3.8`.
- Installing necessary packages: `pip install -r requirements.txt`

### 2. Downloading Training and Testing Datasets

- Download the [training set](https://anu365-my.sharepoint.com/:u:/g/personal/u7248002_anu_edu_au/EQ75AD2A5ClIgqNv6yvstSwBQ1jJNC6DNbk8HISuxPV9QA?e=UhHKSD) (COD10K-train) used for training 
- Download the [testing sets](https://anu365-my.sharepoint.com/:u:/g/personal/u7248002_anu_edu_au/EVI0Bjs7k_VIvz4HmSVV9egBo48vjwX7pvx7deXBtooBYg?e=FjGqZZ) (COD10K-test + CAMO-test + CHAMELEON + NC4K ) used for testing

### 3. Training Configuration

- The pretrained model is stored in [Google Drive](https://drive.google.com/file/d/1OmE2vEegPPTB1JZpj2SPA6BQnXqiuD1U/view?usp=share_link) and [Baidu Drive](https://pan.baidu.com/s/1Dqo5VnL_1z7HwViOftXKGQ?pwd=xuwb 
  ) (xuwb). After downloading, please change the file path in the corresponding code.
- Run `train.sh` or `slurm_train.sh` as needed to train.

### 4. Testing Configuration

Our well-trained model is stored in [Google Drive](https://drive.google.com/file/d/1yh5hKNvFSt9v65Av6ybQ-aVmIc_hpOd5/view?usp=share_link) and [Baidu Drive](https://pan.baidu.com/s/1JuH9ED95f0M1VVLnVNsqIg?pwd=otz5) (otz5). After downloading, please change the file path in the corresponding code.



### 5. Evaluation

- Matlab code: One-key evaluation is written in [MATLAB code](https://github.com/DengPingFan/CODToolbox), please follow this the instructions in `main.m` and just run it to generate the evaluation results.
- Python code: After configuring the test dataset path, run `slurm_eval.py` in the `run_slurm` folder for evaluation.

### 6. Results download

The prediction results of our FSPNet are stored on [Google Drive](https://drive.google.com/file/d/1vgIk5EN0DSPvFclhiYD185lP_XH4ZAWw/view?usp=share_link) and [Baidu Drive](https://pan.baidu.com/s/1yncX2Ct7oh3dWXPCwKupJQ?pwd=ryzg) (ryzg) please check.



## Citation

```
@inproceedings{Huang2023Feature,
title={Feature Shrinkage Pyramid for Camouflaged Object Detection with Transformers},
author={Huang, Zhou and Dai, Hang and Xiang, Tian-Zhu and Wang, Shuo and Chen, Huai-Xin and Qin, Jie and Xiong, Huan},
booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
year={2023}
}
```

Thanks to [Deng-Ping Fan](https://dengpingfan.github.io/), [Ge-Peng Ji](https://scholar.google.com/citations?user=oaxKYKUAAAAJ&hl=en), *et al.* for a series of efforts in the field of [COD](https://github.com/visionxiang/awesome-camouflaged-object-detection#COD).

