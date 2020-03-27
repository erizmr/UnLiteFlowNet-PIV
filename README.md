# Unsupervised learning of Particle Image Velocimetry

## Introduction
We present the first work using un-supervised learning approach for PIV problems. The proposed approachis inspired by classic optical flow method. Instead of using ground truth,we make use of photometric loss between two consecutive frames, con-sistency loss of bidirectional flow estimates and spatial smoothness lossamong the estimates to construct the loss function. The approach showsgreat potential and advantages on fluid flow estimations. It outputs com-petitive results to classic as well as supervised learning methods on PIVdataset, and even outperforms them on some difficult flow metrics.

## Unsupervised Loss

<p align="center">
  <img src="https://github.com/erizmr/UnLiteFlowNet-PIV/blob/master/images/network.png" ><br>
</p>


## Dataset

ref:

## Sample results

- Particle Images from PIV challenge

Jet Flow




## Installation instructions
- Other package and libraries
It is recommended to use the command below to install other package and libraries.
```bash
   pip install numpy
   pip install scipy
   pip install sklearn 
   pip install matplotlib
   pip install torch
   pip install cmocean
   pip install vtk
```

## Run test cases




## Dependencies
To be able to run this software, the following packages and versions are required:

- Firedrake (2019)
- Thetis (2019)
- Pytorch (1.1.0)
- ParticleModule (2019)
- SciPy (v1.3.0)
- NumPy (v1.16.4)
- vtk (v6.0.0)


## License
This software is licensed under the MIT [license](https://github.com/msc-acse/acse-9-independent-research-project-erizmr/blob/master/License)
