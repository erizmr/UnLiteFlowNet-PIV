# Unsupervised learning of Particle Image Velocimetry

## Introduction
Particle Image Velocimetry (PIV) is a classical flow estimation problem which is widely considered and utilised, especially as a diagnostic tool in experimental fluid dynamics and the remote sensing of environmental flows. Recently, the development of deep learning based methods has inspired new approaches to tackle the PIV problem. These supervised learning based methods are driven by large volumes of data with ground truth training information. However, it is difficult to collect reliable ground truth data in large-scale, real-world scenarios. Although synthetic datasets can be used as alternatives, the gap between the training set-ups and real-world scenarios limits applicability. We present here what we believe to be the first work which takes an unsupervised learning based approach to tackle PIV problems. The proposed approach is inspired by classic optical flow methods. Instead of using ground truth data, we make use of photometric loss between two consecutive image frames, consistency loss in bidirectional flow estimates and spatial smoothness loss to construct the total unsupervised loss function. The approach shows significant potential and advantages for fluid flow estimation. Results presented here demonstrate that is outputs competitive results compared with classical PIV methods as well as supervised learning based methods for a broad PIV dataset, and even outperforms these existing approaches in some difficult flow cases.

## Unsupervised Loss

<p align="center">
  <img src="https://github.com/erizmr/UnLiteFlowNet-PIV/blob/master/images/network.png" ><br>
</p>


## Sample results
#### Samples from PIV dataset

- Backstep flow

<p align="center">
  <img src="https://github.com/erizmr/UnLiteFlowNet-PIV/blob/master/images/results/backstep_Re1000_00386.gif" />
  <img src="https://github.com/erizmr/UnLiteFlowNet-PIV/blob/master/images/results/backstep_385_un.png" width="62%" height="62%"/><br>
</p>

- Surface Quasi Geostrophic (SQG) flow

<p align="center">
  <img src="https://github.com/erizmr/UnLiteFlowNet-PIV/blob/master/images/results/SQG_01386.gif" />
  <img src="https://github.com/erizmr/UnLiteFlowNet-PIV/blob/master/images/results/SQG_1385_un.png" width="62%" height="62%"/><br>
</p>



- Particle Images from PIV challenge

Jet Flow


## Dataset

ref:


## Installation instructions


## Run test cases


## Dependencies


## License
This software and documentation
