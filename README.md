# Neural Architecture Search for Image Super-Resolution Using Densely Connected Search Space: DeCoNAS

Basic implementation of DeCoNAS from [Neural Architecture Search for Image Super-Resolution Using Densely Connected Search Space: DeCoNAS](https://~~).

- Uses Tensorflow to define and train the child network / Controller network.
- `Controller` manages the training and evaluation of the Controller RNN
- `ChildNetwork` handles the training and evaluation of the Child network

# Usage
At first, you should download the training dataset(DIV2K) and test datasets(Set5, Set14, B100, Urban100).

For full training details, please see `train.py`.
The metrics and results can be generated with `evaluation.py`

You can search promising networks by DeCoNAS with,
```shell
$ python ./src/DIV2K/train.py --output_dir='./outputs/x2_search' --num_epochs=200 --controller_training=True
```
Train DeCoNASNet searched by DeCoNAS with, 
```shell
$ python ./src/DIV2K/train.py --output_dir='./outputs/x2_scratch' --finetune=False --child_fixed_arc='1 1 1 1 1 0 1 0 0 0 1 1 0 0 0 0 1 0 0 1 0 0 1 1 1 0 0 0 0 1'
```

After training DeCoNANet from scratch, finetune DeCoNASNet with, 
```shell
$ python ./src/DIV2K/train.py --output_dir='./outputs/x2_finetune' --checkpoint='./outputs/x2/model.ckpt-931000' --finetune=True --child_fixed_arc='1 1 1 1 1 0 1 0 0 0 1 1 0 0 0 0 1 0 0 1 0 0 1 1 1 0 0 0 0 1'
```

Finally, evaluate DeCoNASNet with,
```shell
$ python ./src/DIV2K/evaluate.py --checkpoint='model.ckpt-931000' --checkpoint_dir='./outputs/x2'
```


# Implementation details

We train DeCoNAS 1000 epoch, and fintune 1000 epoch(total 2000epoch = 2000*1000 iterations)


# Result
We construct DeCoNASNet with 4 DNBs, and each DNB has 4 layers. 

The sequence of DeCoNASNet is 

'1 1 1 1 1 0 1 0 0 0 1 1 0 0 0 0 1 0 0 1 0 0 1 1 1 0 0 0 0 1'.

For x2 scale super-resolution task, we evaluated the performance(PSNR and SSIM) of our DeCoNASNet on four datasets(Set5, Set14, B100, Urban100).


<img src="https://github.com/titu1994/neural-architecture-search/blob/master/images/training_losses.PNG?raw=true" height=100% width=100%>

# Requirements
- Tensorflow-gpu >= 1.13
- scipy >= 1.5.0
- numpy >= 1.18
- OpenCV >= 3.4.0

# Acknowledgements
We referred the codes of ENAS([melodyguan/enas](https://github.com/melodyguan/enas)).
