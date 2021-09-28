# RGB-T-Salient-Object-Detection-via-CNN-Features-and-Result-Saliency-Maps-Fusion
This project provides the codes and results for 'RGB-T-Salient-Object-Detection-via-CNN-Features-and-Result-Saliency-Maps-Fusion.'

## Overview
![image](https://github.com/xanxuso/RGB-T-Salient-Object-Detection-via-CNN-Features-and-Result-Saliency-Maps-Fusion/blob/main/network.png)

## Code
### For the training of the proposed network:
download pretrained vgg model from [link](https://pan.baidu.com/s/1DDHhmjau01Oo775mi1wdgw),code：0000; put it in `model` directory.

change path in `train.py` to train data .

use `python train.py` to start training.
### For Result Fusion:
change paths in `resultFusion.py`.

use `python resultFusion.py` to fuse saliency maps from RGB and T modalities, there are some examples in `asserts` for testing.

## Experimental Results
Result of VT5000 dataset: [link](https://pan.baidu.com/s/1jn6Y9vi7qhnTIpHiW8anWA),&emsp;code：1234.

Result of VT1000 dataset: [link](https://pan.baidu.com/s/1jex2q55VZeSalOYWCtC4BQ),&emsp;code：4321.
