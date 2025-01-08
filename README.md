# Selective Spectral-Spatial Aggregation Transformer for Hyperspectral and LiDAR Classification (IEEE GRSL)

## Abstract
Convolutional neural networks (CNNs) and transformer have achieved excellent classification performances in HSI and LiDAR land cover classification. However, for complex land covers, effectively characterizing the contextual information and spectral-spatial interaction features of HSI and LiDAR is crucial for improving classification accuracy. Motivated by this, this paper is dedicated to selective convolutional kernel mechanisms and spectral-spatial interactive transformer feature learning style, proposing a selective spectral-spatial aggregation transformer network, named S2ATNet. A convolution feature selected module (CFSM), which can dynamically capture its contextual features of various land covers, is firstly utilized into both of HSI and LiDAR branches. Afterwards, a cascaded spatial-spectral learning and interactive fusion block (CSLIF) is designed for acquiring the non-local spatial-spectral characteristics in an interactive feature learning style. The learned features are fed into max-average classification head (\rewrite{MACH}) to obtain the final classification results. The effectiveness of the proposed S2ATNet is validated on two publicly available datasets.

If this codebase is helpful for you, please consider give me a star 😊.


## Usage

### Installation

* Step 1: Create a conda environment

```shell
conda create --name s2at python=3.7 -y
conda activate s2at
```

* Step 2: Install PyTorch

```shell
# CUDA 11.6, If conda cannot install, use pip
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit==11.6 -c pytorch -c conda-forge
# pip install torch==1.12.0+cu116 torchvision==0.13.0+cu116 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu116
```

* Step 3: Install OpenMMLab 2.x Codebases and Other Dependencies

```shell
# other dependencies
pip install scipy
pip install numpy==1.23.5
pip install terminaltables
pip install timm # 0.9.16
pip install einops # 0.7.0
pip install monai # 1.3.0
pip install ml_collections # 0.1.1
pip install yapf==0.40.0
pip install matplotlib
pip install hdf5storage
```

### Prepare datasets

  * Download [augsburg dataset](https://drive.google.com/drive/folders/1JApzH3UqQO73KEya-lNxdrojKi7uDvuS?usp=drive_link) and [muufl dataset](https://drive.google.com/drive/folders/1cAXTLdC-frkvJTa0HFKarqqnm8UkxKfi?usp=drive_link)
  * The file structure is as followed: 
    ```none
    S2ATNET
    ├── configs
    ├── data                                                
    │   ├── augsburg                                      
    │   │   ├── data_DSM.mat                                      
    │   │   ├── data_HS_LR.mat                                      
    │   │   ├── train_test_gt.mat                                      
    │   ├── muufl                                      
    │   │   ├── HSI.mat                                      
    │   │   ├── LiDAR2.mat                                      
    │   │   ├── muufl.mat                                      
    │   │   ├── tr_ts_gt_150samples.mat                                      
    ```


### Training & Testing

```shell
python main.py
```


## Citation 
  If you find S2ATNet is useful in your research, please consider citing:
  ```shell
  todo
  ```

<!-- ## Acknowledgement
Thanks [mmsegmentation](https://mmsegmentation.readthedocs.io/zh-cn/0.x/index.html) contribution to the community! -->
