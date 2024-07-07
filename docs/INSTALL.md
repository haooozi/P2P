## Installation


- ### Requirement
  
  Here, we list the most important part of our dependencies
  
  |Dependency|Version|
  |---|---|
  |python|3.9.0|
  |pytorch|2.0.1|
  |mmengine|0.7.4|
  |mmcv|2.0.0|
  |mmdet|3.0.0|
  |mmdet3d|1.1.0|
  |spconv|2.3.6|
  |yapf|0.40.0|

- ### Preparation
  
  Clone the repository to your local directory.

  ```shell
  git clone https://github.com/haooozi/P2P.git
  ```
  
- ### Installation
  
  We use conda to manage the environment.

  ```shell
  conda create -n p2p python=3.9
  conda activate p2p
  ```
  
  Some important dependencies.
  
  ```shell
  pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
  pip install mmcv==2.0.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html
  pip install spconv-cu118==2.3.6
  ```
