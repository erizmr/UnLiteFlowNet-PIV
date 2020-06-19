# Unsupervised learning of Particle Image Velocimetry
This repository contains materials for ISC workshop paper Unsupervised learning of Particle Image Velocimetry.

## Introduction
Particle Image Velocimetry (PIV) is a classical flow estimation problem which is widely considered and utilised, especially as a diagnostic tool in experimental fluid dynamics and the remote sensing of environmental flows. We present here what we believe to be the first work which takes an unsupervised learning based approach to tackle PIV problems. The proposed approach is inspired by classic optical flow methods. Instead of using ground truth data, we make use of photometric loss between two consecutive image frames, consistency loss in bidirectional flow estimates and spatial smoothness loss to construct the total unsupervised loss function. The approach shows significant potential and advantages for fluid flow estimation. Results presented here demonstrate that is outputs competitive results compared with classical PIV methods as well as supervised learning based methods for a broad PIV dataset, and even outperforms these existing approaches in some difficult flow cases.

## Unsupervised Loss

<p align="center">
  <img src="https://github.com/erizmr/UnLiteFlowNet-PIV/blob/master/images/network.png" width="80%" height="80%"><br>
</p>


## Sample results
#### Syethetic data: samples from PIV dataset

- Backstep flow

<p align="center">
  <img src="https://github.com/erizmr/UnLiteFlowNet-PIV/blob/master/images/results/backstep_Re1000_00386.gif" width="24.5%" height="24.5%" />
  <img src="https://github.com/erizmr/UnLiteFlowNet-PIV/blob/master/images/results/backstep_385_un.png" width="50%" height="50%"/><br>
</p>

- Surface Quasi Geostrophic (SQG) flow

<p align="center">
  <img src="https://github.com/erizmr/UnLiteFlowNet-PIV/blob/master/images/results/SQG_01386.gif" width="24.5%" height="24.5%" />
  <img src="https://github.com/erizmr/UnLiteFlowNet-PIV/blob/master/images/results/SQG_1385_un.png" width="51%" height="51%"/><br>
</p>


#### Real experimental data: particle Images from [PIV challenge](http://www.pivchallenge.org/)

- Jet Flow

<p align="center">
  <img src="https://github.com/erizmr/UnLiteFlowNet-PIV/blob/master/images/results/jetflow_1.gif" width="85%" height="85%" /><br>
  <em>From left to right: Particle images, UnLiteFlowNet-PIV output, PIV-LiteNetFlow output</em>
</p>



## Dataset
The dataset used in this work is obtained from the work below:

- [PIV dataset](https://doi.org/10.1007/s00348-019-2717-2)(9GB)
```
Shengze Cai, Shichao Zhou, Chao Xu, Qi Gao. 
Dense motion estimation of particle images via a convolutional neural network, Exp Fluids, 2019
```
- [JHTDB](http://turbulence.pha.jhu.edu)
```
Y. Li, E. Perlman, M. Wan, Y. Yang, R. Burns, C. Meneveau, R. Burns, S. Chen, A. Szalay & G. Eyink. 
A public turbulence database cluster and applications to study Lagrangian evolution of velocity increments in turbulence. Journal of Turbulence 9, No. 31, 2008.
```

## Installation instructions


## Run test cases


## Dependencies


