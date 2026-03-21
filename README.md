# **Preliminary Experiment on Complex Ice Floe Motion Monitoring Based on Deep Learning Architectures**

-   Authors: Adan Wu

-   Key Laboratory of Cryospheric Science and Frozen Soil Engineering,
Heihe Remote Sensing Experimental Research Station, Northwest
Institute of Eco-Environment and Resources, Chinese Academy of
Sciences, Lanzhou 730000, China   

## 🧭 Overview

This repository is developed to support and demonstrate the preliminary study on complex sea ice motion monitoring based on deep learning architectures:

The project is motivated by our finding that **rotation and deformation of ice floes are primary factors leading to the failure of traditional feature matching methods**. To address this issue, we construct a complete experimental pipeline integrating **geometric modeling (Hausdorff distance)** and **deep feature learning (SuperPoint + SuperGlue)**.
More specifically, it is detailed as follow.

![Flow chart](https://github.com/RouteViewLab/A-Hausdorff-Guided-Deep-Learning-Approach-for-Monitoring-the-Motion-of-Rotating-Arctic-Ice-Floes/raw/main/Flow%20chat.png)

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

### 4. Experimental Results
Contains:
- Ice floe motion monitoring results between consecutive days  
- Vector maps of short-term continuous motion trajectories  

This folder demonstrates the **effectiveness of the proposed method in real scenarios**.

---

### 5. Influence of Ice Floe Rotation
Focuses on the **preliminary experiment on rotation effects**, including:
- Spatial visualization under **72 rotation angles (5° interval)**  
- Quantitative evaluation of matching performance under rotation  

---

### 6. Code
Includes the implementation of the deep learning framework:
- Network architecture (SuperPoint / SuperGlue-based)  
- Data preprocessing modules  
- Training and testing pipelines

---

## 🔬 Preliminary Experiment: Impact of Rotation

### 📈 Quantitative Analysis

![HD vs Rotation Angle](https://raw.githubusercontent.com/LZUFE-Machine-Learning/A-Hausdorff-Guided-Deep-Learning-Approach-for-Monitoring-the-Motion-of-Rotating-Arctic-Ice-Floes/main/Variation%20of%20Hausdorff%20Distance%20with%20Rotation%20Angle.png))

This figure shows the variation of **Hausdorff Distance (HD)** with respect to rotation angle for ice floe **B2**.

- Minimum HD (**3.16**) occurs at approximately **40°**, indicating optimal alignment  
- Maximum HD (**118.07**) reflects severe mismatch  
- Strong nonlinearity indicates high sensitivity to rotation  

👉 **Insight:**  
Rotation drastically alters geometric similarity, making direct matching unreliable.

---

### 🖼️ Qualitative Comparison

![Rotation Comparison](https://raw.githubusercontent.com/LZUFE-Machine-Learning/A-Hausdorff-Guided-Deep-Learning-Approach-for-Monitoring-the-Motion-of-Rotating-Arctic-Ice-Floes/main/Rotation%20Comparison%20Chart.png)

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
on [S](https://github.com/hanyoseob/pytorch-noise2void)uperPoint and [S](https://github.com/DegangWang97/IEEE_TGRS_BS3LNet)uperGlue.
Thanks for their awesome work.

## **Contact**

If you have any questions or suggestions, feel free to contact me.\
Email: wuadan@lzb.ac.cn
