# **Preliminary Experiment on Complex Ice Floe Motion Monitoring Based on Deep Learning Architectures**

- Authors: Adan Wu, Tao Che*, Chengzhu Ji, Xiaowen Zhu, Jinlei Chen, Qingchao Xu, Qun Gu, Rui Zhang,  
Kaihui Zhang, Lei Fu, and Shengpeng Chen  

-   Key Laboratory of Cryospheric Science and Frozen Soil Engineering,
Heihe Remote Sensing Experimental Research Station, Northwest
Institute of Eco-Environment and Resources, Chinese Academy of
Sciences, Lanzhou 730000, China   

## 🧭 Overview

This repository is developed to support and demonstrate the preliminary study on complex sea ice motion monitoring based on deep learning architectures:

The project is motivated by our finding that **rotation and deformation of ice floes are primary factors leading to the failure of traditional feature matching methods**. To address this issue, we develop a framework that integrates **geometric modeling (Hausdorff distance)** with **deep feature learning (SuperPoint + SuperGlue)**.

More specifically, the deep learning component of our framework is built upon two key modules: **SuperPoint** for interest point detection and description, and **SuperGlue** for feature matching. Their network architectures are illustrated below.

<div align="center">
  <img src="https://github.com/RouteViewLab/A-Hausdorff-Guided-Deep-Learning-Approach-for-Monitoring-the-Motion-of-Rotating-Arctic-Ice-Floes/raw/main/SuperPoint%20Network.png" width="600"/>
  <p><b>Figure 1.</b> SuperPoint Network for keypoint detection and descriptor extraction.</p>
</div>

<div align="center">
  <img src="https://github.com/RouteViewLab/A-Hausdorff-Guided-Deep-Learning-Approach-for-Monitoring-the-Motion-of-Rotating-Arctic-Ice-Floes/raw/main/SuperGlue%20Network.png" width="600"/>
  <p><b>Figure 2.</b> SuperGlue Network for feature matching.</p>
</div>

In this framework, **SuperPoint** is employed to extract rotation-invariant keypoints and descriptors, while **SuperGlue** performs context-aware feature matching through graph neural network-based attention mechanisms. 

---

## 📂 Repository Structure

The repository consists of **six main folders**, covering the full workflow:

### 1. Source Data
Contains all **226 Arctic ice floe images**, which are fully utilized for the deep learning framework.

---

### 2. Ice Floe Data for the Experiment
Includes:
- Original ice floe imagery  
- Preprocessed data used in experiments  

This folder reflects the **data preparation and preprocessing pipeline**.

---

### 3. Traditional Methods Results
Provides feature extraction and matching results using:
- SIFT  
- A-KAZE  

These results serve as **baseline comparisons**, highlighting the limitations of traditional methods under rotation and deformation.

---

### 4. Deep Learning Experimental Results
This folder primarily presents the **deep learning-based experimental results** of the proposed framework.

It contains:

- Ice floe motion monitoring results between consecutive days obtained using the proposed deep learning model  
- Vector maps illustrating short-term continuous motion trajectories generated from learned feature correspondences  

This folder demonstrates the **effectiveness of the proposed method in real scenarios**.

To further demonstrate the superiority of the proposed deep learning approach, we provide a comparative analysis using the **B4 ice floe (July 6, 2020)** as a representative example.


| Method              | Matching Pairs | Matched Accuracy |
|---------------------|----------------|------------------|
| **Proposed method** | **50** | **100%** |
| SIFT                | 15             | 40%              |
| A-KAZE              | 19             | 68.42%           |

---

### 5.Influence of Ice Floe Rotation on Deep Learning-Based Monitoring
This folder focuses on the **impact of ice floe rotation on the performance of the deep learning-based monitoring framework**, including:

- Spatial visualization under **72 rotation angles (5° interval)**  
- Quantitative evaluation of **deep learning-based matching performance** under varying rotation conditions  

---

### 6. Code
Includes the implementation of the deep learning framework:
- Network architecture (SuperPoint / SuperGlue-based)  
- Data preprocessing modules  
- Training and testing pipelines

---

## 🔬 Preliminary Experiment: Impact of Rotation

### 📈 Quantitative Analysis

![HD vs Rotation Angle](https://raw.githubusercontent.com/LZUFE-Machine-Learning/A-Hausdorff-Guided-Deep-Learning-Approach-for-Monitoring-the-Motion-of-Rotating-Arctic-Ice-Floes/main/Variation%20of%20Hausdorff%20Distance%20with%20Rotation%20Angle.png)

This figure shows the variation of **Hausdorff Distance (HD)** with respect to rotation angle for ice floe **B2**.

- Minimum HD (**3.16**) occurs at approximately **40°**, indicating optimal alignment  
- Maximum HD (**118.07**) reflects severe mismatch  
- Strong nonlinearity indicates high sensitivity to rotation  

👉 **Insight:**  
Rotation drastically alters geometric similarity, making direct matching unreliable.

---

### 🖼️ Qualitative Comparison

![Rotation Comparison](https://raw.githubusercontent.com/LZUFE-Machine-Learning/A-Hausdorff-Guided-Deep-Learning-Approach-for-Monitoring-the-Motion-of-Rotating-Arctic-Ice-Floes/main/Rotation%20Comparison.png)

#### 🔹 Before Rotation Alignment
- Feature matches are **disordered and inconsistent**  
- Significant geometric deviation  
- High mismatch rate  

#### 🔹 After Rotation Alignment (~40°)
- Ice floe is **well aligned**  
- Matches become **structured and reliable**  
- Significant improvement in accuracy  

---

## 🚀 Method Overview

This repository implements a **rotation-aware deep learning framework** that integrates:

- **Hausdorff Distance** → rotation estimation  
- **SuperPoint** → feature extraction  
- **SuperGlue** → feature matching  

Key advantages:
- Robust to **large rotation**
- Handles **low-texture regions**
- Improves **matching accuracy and stability**

---


## **Acknowledgments**

The codes are based
on [SuperPoint](https://github.com/hanyoseob/pytorch-noise2void) and [SuperGlue](https://github.com/DegangWang97/IEEE_TGRS_BS3LNet).
Thanks for their awesome work.

## **Contact**

If you have any questions or suggestions, feel free to contact me.\
Email: wuadan@lzb.ac.cn
