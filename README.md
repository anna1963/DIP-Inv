# A Test-Time Learning Approach to Reparameterize the Geophysical Inverse Problem with a Convolutional Neural Network (DIP-Inv)

This work was presented in AGU23 and been awarded the Outstanding Student Presentation Awards (OSPA). You can find poster and video presentation in the OSPA Winners's gallery: https://agu23.ipostersessions.com/Default.aspx?s=ospa-2024-winners-gallery, by searching "LEVERAGING CONVOLUTIONAL NEURAL NETWORKS FOR IMPLICIT REGULARIZATION IN DC RESISTIVITY INVERSIONS". This work was published in IEEE Transactions on Geoscience and Remote Sensing, and you can find the link to the paper below.

## Summary
We proposed a method that doesn’t require a training dataset of the subsurface models. This test-time learning method, where the CNN weights are adjusted for each test data, produces better inversion results in multiple synthetic cases. 
In this study, we examine the applicability of the implicit regularization from the CNN structure to Tikhonov-style geophysical inversions. The CNN maps an arbitrary vector to the model space (e.g., log-conductivity on the simulation mesh). 
The predicted subsurface model is then fed into a forward numerical simulation process to generate corresponding predicted measurements. 
Subsequently, the objective function value is computed by comparing these predicted measurements with the observed field measurements. 
The backpropagation algorithm is employed to update the trainable parameters of the CNN. 
Note that the CNN in our proposed method does not require training before the inversion, rather, the CNN weights are estimated in the inversion algorithm.

## Contents
There are 3 notebooks in the 'notebooks' directory:
- Generate_Synthetic_Voltages.ipynb: runs the forward simulation for the 5 cases shown in the paper. 5% Gaussion noises are added to the synthetic data. 
- Plot_Results.ipynb: gives an example about how to plot the image in the paper using SimPEG and matplotlib.
All notesbook can be run on Colab (!pip install is included in the first cell).
- Conventional_inversion_using_SimPEG.ipynb: runs the SimPEG inversion codes to get the conventional inversion results.

There are 2 python scripts in the 'train' directory:
- stage_1.py: trains a randomly initialized CNN with the lose |m-m_ref|. This stage is mainly for saving time in the experiments with the same reference model.
- train.py: using PyTorch as the deep learning frame and SimPEG for the forward simulations (details on how we connect them can be found in our paper). 
The final results (both the final predicted conductivity model and the weights) will be stored in the proper format with name defined in 'file_name' in the attributes dictionary.

## Usage

Dependencies are specified in [requirements.txt](/requirements.txt)

```
pip install -r requirements.txt
```
You can then clone this repository. From a command line, run

```
git clone https://github.com/anna1963/DIP-Inv.git
```

Then `cd` into the `DIP-Inv` directory:

```
cd DIP-Inv
```

To setup your software environment, we recommend you use the provided conda environment

```
conda env create -f environment.yml
conda activate DIP-Inv-environment
```
## Running the notebooks

For more information on running Colab notebooks, see https://colab.google.

## Citation

The preprint is available on the ArXiv: http://arxiv.org/abs/2312.04752.
The IEEE Early Access is available on: https://ieeexplore.ieee.org/document/10587271.

©2024 IEEE. Personal use of this material is permitted. Permission from IEEE must be obtained for all other uses, in any current or future media, including reprinting/republishing this material for advertising or promotional purposes, creating new collective works, for resale or redistribution to servers or lists, or reuse of any copyrighted component of this work in other works.

```
@ARTICLE{10587271,
  author={Xu, Anran and Heagy, Lindsey J.},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={A Test-Time Learning Approach to Reparameterize the Geophysical Inverse Problem with a Convolutional Neural Network}, 
  year={2024},
  volume={},
  number={},
  pages={1-1},
  keywords={Inverse problems;Conductivity;Convolutional neural networks;Training;Neural networks;Mathematical models;Geology;Convolutional neural network (CNN);deep image prior (DIP);deep learning (DL);direct-current resistivity (DCR) inversion;parameterized inversion (PI)},
  doi={10.1109/TGRS.2024.3424659}}
}
```

## License
These notebooks are licensed under the [MIT License](/LICENSE).
