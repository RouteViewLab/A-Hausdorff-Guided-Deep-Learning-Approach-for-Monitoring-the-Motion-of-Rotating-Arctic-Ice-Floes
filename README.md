A Hausdorff-Guided Deep Learning Approach for Monitoring
the Motion of Rotating Arctic Ice Floes

-   Authors: Adan Wu, Tao Che∗, Chengzhu Ji, Xiaowen Zhu, Jinlei Chen, Qingchao Xu, Qun Gu, Rui Zhang,
Kaihui Zhang, Lei Fu, and Shengpeng Chen

-   Key Laboratory of Cryospheric Science and Frozen Soil Engineering,
Heihe Remote Sensing Experimental Research Station, Northwest
Institute of Eco-Environment and Resources, Chinese Academy of
Sciences, Lanzhou 730000, China

    The code in this toolbox implements the [[\"A Hausdorff-Guided Deep
    Learning Approach for Motion Monitoring of Rotating Arctic Ice
    Floes\"]{.underline}](https://ieeexplore.ieee.org/document/9627165).
    More specifically, it is detailed as follow.

![Flow chart](https://github.com/RouteViewLab/A-Hausdorff-Guided-Deep-Learning-Approach-for-Monitoring-the-Motion-of-Rotating-Arctic-Ice-Floes/raw/main/Flow%20chat.png)

## Overview

This repository provides the implementation of our proposed ice floe
motion monitoring framework, which integrates **geometric selection**
and **intelligent feature matching**. The method combines
Hausdorff-based geometric selection, SuperPoint feature extraction, and
SuperGlue feature matching to address challenges of texture degradation
and rotational variations in Arctic ice floe imagery.

The source code is implemented in **PyTorch**, with training and testing
pipelines designed to reproduce the results reported in our manuscript
submitted.
### **Requirements**

Please ensure the following dependencies are installed:

-   Python \>= 3.11

-   PyTorch \>= 1.6

-   NumPy, SciPy, OpenCV

-   Matplotlib (for visualization)

## Data Preparation

-   The datasets consist of optical imagery acquired from Arctic
    marginal ice zones.

-   Example data used in the paper (July 6, 2020, B4 floe sample) are
    provided in the data/ directory.

-   Users may organize their own datasets following the structure below

    datasets/ (\$DATA_DIR)

    \|\-- Dataset

    \| \|\-- train2014

    \| \| \|\-- file1.jpg

    \| \| \`\-- \...

    \| \`\-- val2014

    \| \|\-- file1.jpg

    \| \`\-- \...

## Training

The training procedure is designed for reproducibility and can be
customized with different network hyperparameters.

python train4.py train_base configs/magicpoint_shapes_pair.yaml
magicpoint_synth \--eval

python train4.py train_joint
configs/superpoint_dataset_train_heatmap.yaml superpoint_my_data \--eval
\--debug

python train.py \--feature_dim 256 \--dataset_offline_rebuild 1
\--batch_size 32 \--debug 0 \--eval \--viz

Options:

-   \--threshold_keypoint: 0.001

-   \--threshold_match: 0.1

-   \--lr: 1 × 10⁻⁴

-   \--epochs: 500

## Testing

Testing can be performed on the held-out dataset or on provided floe
samples.

python test.py \--model checkpoints/model_best.pth \--data data/test/

This will output matched pairs, matching accuracy, and visualization
results under the results/ folder.

Example Experiment (B4 Floe, July 6, 2020)

A demo script is provided to reproduce the **B4 floe matching
experiment** described in the paper.

-   this experiment demonstrates the ability of the proposed framework
    to achieve robust matching under rotation and texture loss.

-   Please note that results may slightly differ from those reported in
    the manuscript due to variations in PyTorch and related library
    versions.

  ## Results

| Method          | Matching Pairs | Matched Accuracy |
|-----------------|----------------|------------------|
| Proposed method | 50             | 100%             |
| SIFT            | 15             | 40%              |
| A-KAZE          | 19             | 68.42%           |


## **Acknowledgments**

The codes are based
on [S](https://github.com/hanyoseob/pytorch-noise2void)uperPoint and [S](https://github.com/DegangWang97/IEEE_TGRS_BS3LNet)uperGlue.
Thanks for their awesome work.

## **Contact**

If you have any questions or suggestions, feel free to contact me.\
Email: wuadan@lzb.ac.cn
