# LastSAM: Sharpness-Aware Minimization at the End

This repository contains an implementation of LastSAM, a variant of Sharpness-Aware Minimization (SAM) that applies sharpness-aware optimization only after a model has been trained using standard optimization techniques. LastSAM aims to improve the generalization performance of deep learning models by finding flatter minima in the loss landscape, while being more computationally efficient than the original SAM and Adaptive SAM (ASAM) methods.

## Motivation

Sharpness-Aware Minimization (SAM) and its adaptive variant, ASAM, have shown promising results in improving the generalization performance of deep learning models by simultaneously minimizing the loss value and the loss sharpness. However, these methods can be computationally expensive and may require extensive hyperparameter tuning. LastSAM addresses these limitations by applying SAM or ASAM only during the last few epochs of training, after the model has been trained using standard optimization techniques such as Stochastic Gradient Descent (SGD).

## Implementation

This project is built upon the ASAM implementation available at [link to the original ASAM repository]. The main contributions of this repository are:

1. `lastSAM.py`: This file contains the implementation of the LastSAM optimizer, which applies SAM or ASAM during the last few epochs of training.

2. `ASAM.py`: The original ASAM implementation has been modified to support the LastSAM approach.

## Usage

To use LastSAM in your deep learning project, follow these steps:

1. Clone this repository:

``` git clone https://github.com/yourusername/LastSAM.git```

2. Install the required dependencies:

```pip install -r requirements.txt```

3.  Import the LastSAM optimizer in your training scipt and use it during training.

## Experiments
The effectiveness of LastSAM has been demonstrated on the CIFAR-10 dataset using ResNet20 and WRN-28-10 architectures. The results show that LastSAM achieves slightly better or comparable test accuracies compared to the default SAM and ASAM methods, while being more computationally efficient and robust to hyperparameter choices.

## Acknowledgements

- The original ASAM implementation: https://github.com/SamsungLabs/ASAM

- The authors of the SAM and ASAM papers for their influential work in sharpness-aware optimization.
