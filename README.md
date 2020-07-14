# Neural Architecture Search for Image Super-Resolution Using Densely Connected Search Space: DeCoNAS

Basic implementation of DeCoNAS from [Neural Architecture Search for Image Super-Resolution Using Densely Connected Search Space: DeCoNAS](https://~~).

- Uses Tensorflow to define and train the child network / Controller network.
- `Controller` manages the training and evaluation of the Controller RNN
- `ChildNetwork` handles the training and evaluation of the Child network

# Usage
For full training details, please see `train.py`.


The metrics and results can be generated with 'evaluation.py'
 

# Implementation details

We train DeCoNAS 1000 epoch, and fintune 1000 epoch(total 2000epoch = 2000*1000 iterations)


# Result
We construct DeCoNASNet with 4 DNBs, and each DNB has 4 layers. 
The sequence of DeCoNASNet is '1 1 1 1 1 0 1 0 0 0 1 1 0 0 0 0 1 0 0 1 0 0 1 1 1 0 0 0 0 1'.

For x2 scale super-resolution task, we evaluated the performance(PSNR and SSIM) of our DeCoNASNet on four datasets(Set5, Set14, B100, Urban100).


<img src="https://github.com/titu1994/neural-architecture-search/blob/master/images/training_losses.PNG?raw=true" height=100% width=100%>

# Requirements
- Tensorflow-gpu >= 1.13
- 

# Acknowledgements
We referred the codes of ENAS.[melodyguan/enas](https://github.com/melodyguan/enas)
